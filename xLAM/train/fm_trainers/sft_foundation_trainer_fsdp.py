import time
import math
import os
from typing import Optional
from collections import defaultdict

from fm_utils.fsdp_utils.wrapping import (
    get_llama_wrapper,
    get_mistral_wrapper,
    get_mixtral_wrapper,
    get_size_policy,
)
from fm_utils.fsdp_utils.env_check import bfloat_support
from fm_utils.fsdp_utils.mixed_precision import bfSixteen
from fm_utils.fsdp_utils.config import fsdp_config
from fm_utils.fsdp_utils.activation_checkpointing import apply_fsdp_checkpointing

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
import gc

from tqdm import tqdm
from safetensors.torch import save_file
import wandb

from torch.distributed.elastic.multiprocessing.errors import record


def prepare_policies(args):
    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    bfloat_available = bfloat_support()
    if bfloat_available:
        mixed_precision_policy = bfSixteen
    else:
        print(
            f"bFloat16 support not present. Will use FP32, and not mixed precision"
        )
    if args.fsdp_wrapper == "llama":
        wrapping_policy = get_llama_wrapper()
    elif args.fsdp_wrapper == "mixtral":
        wrapping_policy = get_mixtral_wrapper()
    elif args.fsdp_wrapper == "mistral":
        wrapping_policy = get_mistral_wrapper()
    else:
        wrapping_policy = get_size_policy()
    return mixed_precision_policy, wrapping_policy


def setup_pg(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    print(f"set up process group for rank {rank} of world size {world_size} ")

def cleanup_pg():
    dist.destroy_process_group()

def prepare_fsdp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_pg(rank, world_size)


class SFTFoundationTrainerFSDP:

    def __init__(self, args, train_dataset, eval_dataset, collator):
        self.args = args

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # setup_pg(self.rank, self.world_size)
        torch.cuda.set_device(self.local_rank)

        mixed_precision_policy, wrapping_policy = prepare_policies(self.args)
        model = self._get_model()
        self.model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=fsdp_config.limit_all_gathers,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(self.model, self.args.fsdp_wrapper)

        self.optimizer = self._prepare_optimizer()
        self.lr_scheduler = self._prepare_scheduler()

        self.train_dataloader, self.eval_dataloader = \
            self._prepare_dataset(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                collator=collator)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.logger = self.setup_logger(self.args)

        if self.rank == 0:
            print("*" * 50)
            print("Setup Ready ...")
            print("     World Size =", self.world_size)
            print()


    def _prepare_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )
        return optimizer

    def _prepare_scheduler(self):
        lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_steps,
        )
        return lr_scheduler

    def _get_model(self):
        args = self.args
        get_model_start_time = time.time()
        if self.rank == 0:
            print("*" * 50)
            print("Preparing model ...")
            print("     Model name or path =", args.model_name)
            print("     Max context length =", args.seq_length)
            print("     Weight format =", args.weight_precision)
            print()

        model_name_or_path = args.model_name
        weight_precision = args.weight_precision

        if weight_precision == 'bf16':
            if self.rank == 0:
                print("Using bf16 for model weights")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_auth_token=True,
                attn_implementation="flash_attention_2",
            )

        elif weight_precision == 'nf4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_auth_token=True,
                attn_implementation="flash_attention_2",
            )

        else:
            raise Exception(f"Only supporting weight precision to be bf16 or nf4 for now. Received {weight_precision}")

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        model.config.use_cache = False
        get_model_end_time = time.time()
        if self.rank == 0:
            print("model:")
            print(model)
            print()
            print("Loaded model -- took", get_model_end_time - get_model_start_time)
            print()
        return model

    def _prepare_dataset(self, train_dataset=None, eval_dataset=None, collator=None):
        args = self.args

        train_dataloader = None
        eval_dataloader = None

        train_dataloader_kwargs = {"batch_size": args.per_device_train_batch_size,
                             "pin_memory": True,
                             "shuffle": False,
                             }

        if isinstance(train_dataset, IterableDataset):
            train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=args.per_device_train_batch_size,
                    drop_last=True,
                    num_processes=self.world_size,
                    process_index=self.rank,
                )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collator,
                drop_last=True,
                **train_dataloader_kwargs,
            )
        else:
            raise Exception("We expect the dataloader an object of IterableDataset")

        if eval_dataset is not None:
            eval_dataloader_kwargs = {"batch_size": args.per_device_eval_batch_size,
                                       "pin_memory": True,
                                       "shuffle": False,
                                       }

            if isinstance(eval_dataset, IterableDataset):
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    drop_last=True,
                    num_processes=self.world_size,
                    process_index=self.rank,
                )
                eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=collator,
                    drop_last=True,
                    **eval_dataloader_kwargs,
                )
            else:
                raise Exception("We expect the dataloader an object of IterableDataset")
        return train_dataloader, eval_dataloader

    @record
    def train(self):
        gc.collect()
        torch.cuda.empty_cache()

        args = self.args
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        max_steps: int = int(args.max_steps)
        self.model.train()

        num_updated_steps: int = 0
        avg_loss: float = 0.
        max_retries = 3
        curr_retry = 0
        gradient_accumulation_cnt = 0

        if self.local_rank == 0:
            inner_pbar = tqdm(range(max_steps))
        for step, batch in enumerate(self.train_dataloader):
            if step > max_steps > 0:
                break
            if curr_retry > max_retries:
                break
            try:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch[
                            'labels']
                input_ids = input_ids.to(self.local_rank)
                attention_mask = attention_mask.to(self.local_rank)
                labels = labels.to(self.local_rank)

                if step == 0 and self.rank == 0:
                    print("     Input shape =", input_ids.shape)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                avg_loss += loss.item()
                loss.backward()
                gradient_accumulation_cnt += 1
                if gradient_accumulation_cnt % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    num_updated_steps += 1
                    if self.logger is not None and step % args.logging_steps == 0:
                        avg_loss = avg_loss / args.gradient_accumulation_steps
                        self.logger.log({"train_loss": avg_loss}, step=step)
                        if self.local_rank == 0:
                            print(f"step = {step} -- rank {self.rank} -- avg_loss = {avg_loss} -- num_updated_steps = {num_updated_steps}")
                    gradient_accumulation_cnt = 0
                    avg_loss = 0.0
            except Exception as e:
                print(f"step = {step} -- rank {self.rank}: error:", e)
                curr_retry += 1
            if self.rank == 0:
                inner_pbar.update(1)

            if step > 0 and step % args.save_steps == 0:
                self.save_checkpoint_fsdp(
                    output_dir=os.path.join(args.output_dir, f'step_{step}'),
                    save_full=True)

        gc.collect()
        torch.cuda.empty_cache()

        if self.local_rank == 0:
            inner_pbar.close()

        dist.barrier()
        wandb.finish()

    def eval(self):
        pass

    def save_checkpoint_fsdp(self, output_dir, save_full=True):
        """saving model via rank0 cpu streaming and full_state_dict"""

        # saving with rank0 cpu
        # if not cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
        #    print(f" unable to handle checkpoint type {cfg.checkpoint_type}, aborting")
        fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = self.model.state_dict()
            print(f"saving process: rank {self.rank} w model state_dict\n")
        if self.rank == 0:
            print(f"--> saving model ...")
            # create save path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # save model
            if save_full:
                save_file(cpu_state, f"{output_dir}/full_model.safetensors")

    def save_model(self, output_dir):
        self.save_checkpoint_fsdp(output_dir=output_dir, save_full=True)

    def store_metrics(self, metrics, train_eval="train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def setup_logger(self, args):
        if args.log_all:
            logger = wandb.init(
                project="huggingface",
                name=args.run_name + ":rank=" + str(self.rank),
                group="FSDP",
                job_type="training",
            )
        else:
            if self.local_rank == 0:
                logger = wandb.init(
                    project="huggingface",
                    name=args.run_name + ":rank=" + str(self.rank),
                )
            else:
                logger = None
        return logger

    def log(self):
        pass




