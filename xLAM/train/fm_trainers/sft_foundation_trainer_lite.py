import time
import math
import os
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import gc

from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from deepspeed.utils import set_z3_leaf_modules
from accelerate import (
    DeepSpeedPlugin,
    Accelerator,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_scheduler,
    HfArgumentParser,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)

from tqdm import tqdm
from safetensors.torch import save_file
import wandb


def prepare_accelerator(args):
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=args.ds_config_path,
        zero_stage=int(args.ds_stage),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        zero3_init_flag=(int(args.ds_stage) == 3),
    )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        mixed_precision="bf16",
        log_with=args.log_with if args.use_log else None,
    )

    if args.use_log:
    # Initialise your wandb run, passing wandb parameters and any config information
        log_with = args.log_with
        accelerator.init_trackers(
            project_name="huggingface",
            init_kwargs={
                log_with:
                    {
                        "name": args.run_name
                    }
            }
        )
    return accelerator


class SFTFoundationTrainerLite:

    def __init__(self, args, accelerator, train_dataset, eval_dataset, collator):
        self.args = args
        self.accelerator = accelerator
        model = self._get_model()
        optimizer = self._prepare_optimizer(model=model)
        lr_scheduler = self._prepare_scheduler(optimizer=optimizer)
        train_dataloader, eval_dataloader = \
            self._prepare_dataset(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                collator=collator)

        self.model, self.train_dataloader, self.eval_dataloader, self.optimizer, self.lr_scheduler = \
            self._final_cook_with_accelerator(
                model,
                train_dataloader,
                eval_dataloader,
                optimizer,
                lr_scheduler
            )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    # def __del__(self):
    #     if self.accelerator is not None:
    #         self.accelerator.end_training()

    def _prepare_optimizer(self, model):
        assert isinstance(self.accelerator, Accelerator)
        if self.accelerator.state.deepspeed_plugin is None or "optimizer" not \
                in self.accelerator.state.deepspeed_plugin.deepspeed_config:
            optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = DummyOptim

        optimizer = optimizer_cls(params=model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def _prepare_scheduler(self, optimizer):
        assert isinstance(self.accelerator, Accelerator)
        if self.accelerator.state.deepspeed_plugin is None or "scheduler" not \
                in self.accelerator.state.deepspeed_plugin.deepspeed_config:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.max_steps,
            )
        else:
            if self.accelerator.is_main_process:
                print("Using DummyScheduler --> move to DS scheduler")
            lr_scheduler = DummyScheduler(
                optimizer,
                warmup_num_steps=self.args.num_warmup_steps,
                total_num_steps=self.args.max_steps,
            )
        return lr_scheduler

    def _get_model(self):
        assert isinstance(self.accelerator, Accelerator)
        args = self.args
        get_model_start_time = time.time()
        if self.accelerator.is_main_process:
            print("*" * 50)
            print("Preparing model ...")
            print("     Model name or path =", args.model_name)
            print("     Max context length =", args.seq_length)
            print("     Weight format =", args.weight_precision)
            print("     Use LoRA =", args.use_lora)
            print()

        model_name_or_path = args.model_name
        weight_precision = args.weight_precision

        if weight_precision == 'bf16':
            if self.accelerator.is_main_process: print("Using bf16 for model weights")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_auth_token=True,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=(args.ds_stage != 3),
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
                low_cpu_mem_usage=(args.ds_stage != 3),
            )

        else:
            raise Exception(f"Only suppoprting weight precision to be bf16 or nf4 for now. Received {weight_precision}")

        if args.use_lora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing
            )

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

        model.config.use_cache = False
        if int(args.ds_stage) == 3:
            if self.accelerator.is_main_process:
                print("enabling z3 leaf for ds_03")
            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        if args.use_lora:
            target_modules = args.lora_target_modules.split(",")
            if self.accelerator.is_main_process:
                print(f"Loading LoRA with target modules = {target_modules}...")
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, peft_config)

        else:
            if self.accelerator.is_main_process:
                print("Not using LoRA")

        get_model_end_time = time.time()
        if self.accelerator.is_main_process:
            print("model:")
            print(model)
            print()
            print("Loaded model -- took", get_model_end_time - get_model_start_time)
            print()

        return model

    def _prepare_dataset(self, train_dataset, eval_dataset=None, collator=None):
        args = self.args
        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=collator,
                                      batch_size=args.per_device_train_batch_size,
                                      drop_last=True)
        if eval_dataset is not None:
            eval_dataloader = DataLoader(eval_dataset,
                                         collate_fn=collator,
                                         batch_size=args.per_device_eval_batch_size,
                                         drop_last=False)
        else:
            eval_dataloader = None
        return train_dataloader, eval_dataloader

    def _final_cook_with_accelerator(self, model, train_dataloader, eval_dataloader, optimizer, lr_scheduler):
        if eval_dataloader is not None:
            if self.accelerator.is_main_process:
                print("     USING eval set")

            model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = self.accelerator.prepare(
                model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
            )
        else:
            if self.accelerator.is_main_process:
                print("     Not using eval set")
            model, train_dataloader, optimizer, lr_scheduler = self.accelerator.prepare(
                model, train_dataloader, optimizer, lr_scheduler
            )
        return model, train_dataloader, eval_dataloader, optimizer, lr_scheduler

    def train(self):
        gc.collect()
        torch.cuda.empty_cache()

        args = self.args
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        max_steps: int = int(args.max_steps)

        if self.accelerator.is_main_process:
            print(f"Training...")

        self.model.train()

        num_updated_steps: int = 0
        avg_loss: float = 0.
        max_retries = 3
        curr_retry = 0
        with tqdm(total=max_steps) as pbar:
            for step, batch in enumerate(self.train_dataloader):
                if step > max_steps > 0:
                    break
                if curr_retry > max_retries:
                    break
                try:
                    with self.accelerator.accumulate(self.model):
                        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch[
                            'labels']

                        if step == 0 and self.accelerator.is_main_process:
                            print("     Input shape =", input_ids.shape)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        loss = outputs.loss
                        avg_loss += loss.item()

                        self.accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        # update avg_loss
                        num_updated_steps += 1
                        mean_avg_loss = sum(self.accelerator.gather(torch.tensor([avg_loss]).to(
                            device=self.accelerator.device)).detach().cpu()) / \
                            (args.gradient_accumulation_steps * self.accelerator.num_processes)
                        if args.use_log:
                            self.accelerator.log({"train_loss": mean_avg_loss}, step=step)
                        if self.accelerator.is_main_process and step % args.logging_steps == 0:
                            print(f"step = {step} -- avg_loss = {mean_avg_loss} --num_updated_steps = {num_updated_steps}")
                        avg_loss = 0.

                except Exception as e:
                    print(f"step = {step} -- rank {self.accelerator.local_process_index}: error:", e)
                    curr_retry += 1
                pbar.update(1)

                if step > 0 and step % args.save_steps == 0:
                    self.save_checkpoint_ds(
                        output_dir=os.path.join(args.output_dir, f'step_{step}'),
                        save_full=not args.use_lora)

        gc.collect()
        torch.cuda.empty_cache()

        self.accelerator.end_training()

    def eval(self):
        """
        """
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        with torch.no_grad():
            eval_losses = []

            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.detach().repeat(int(input_ids.shape[0]))

                eval_losses.append(self.accelerator.gather_for_metrics(loss).detach().cpu())

            try:
                avg_eval_loss = torch.mean(torch.cat(eval_losses))
            except:
                avg_eval_loss = -1.

        # TODO: add logging here

        return avg_eval_loss

    def save_checkpoint_ds(self, output_dir, save_full=False):
        args = self.args
        if self.accelerator.is_main_process:
            print("*" * 50 + "\nPreparing to save model ...")

        if int(args.ds_stage) == 3:
            if self.accelerator.is_main_process:
                print("     Getting DS Zero 3 consolidated state dict")
            ds_state_dict = self.model._zero3_consolidated_16bit_state_dict()
        else:
            assert int(args.ds_stage) == 2
            if self.accelerator.is_main_process:
                print("     Getting DS Zero 2 state dict")

            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            ds_state_dict = unwrapped_model.state_dict()

        if args.use_lora:
            if self.accelerator.is_main_process:
                print("     Getting LoRA state dict")

            lora_state_dict = get_peft_model_state_dict(self.model, ds_state_dict)

        print(f"    [Now saving] --- local rank = {self.accelerator.local_process_index}")
        if self.accelerator.is_main_process:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if args.use_lora:
                save_file(lora_state_dict, f"{output_dir}/adapter_model.safetensors")
            if save_full:
                save_file(ds_state_dict, f"{output_dir}/full_model.safetensors")

        self.accelerator.wait_for_everyone()
        print(f"    [Finished saving] --- local rank = {self.accelerator.local_process_index}")

    def save_model(self, output_dir):
        self.save_checkpoint_ds(output_dir=output_dir, save_full=True)

    def store_metrics(self, metrics, train_eval="train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self):
        pass




