# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from accelerate import  DeepSpeedPlugin, Accelerator
from typing import Optional
import wandb
import json
import torch
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
from huggingface_hub import login
from tqdm import tqdm

from actionstudio.src.foundation_modeling.train.train_sft import prepare_data
from actionstudio.src.foundation_modeling.trainers.sft_foundation_trainer import SFTFoundationTrainerLite
from actionstudio.src.foundation_modeling.data_handlers.any_dataset import AnyDatasetLoader
from actionstudio.src.foundation_modeling.data_handlers.derived_data_collator import DataCollatorForPromptAnswer
from actionstudio.src.foundation_modeling.data_handlers.interleave_datasets import interleave_data

from actionstudio.src.foundation_modeling.utils.seed_random import init_device_seed
from actionstudio.src.foundation_modeling.utils.common import load_yaml_file, create_sampled_ratio

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    # model
    model_name: Optional[str] = field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", metadata={"help": "the model name"})
    fc_mode: Optional[bool] = field(default=False, metadata={"help": "whether to use function mode when using Tokenizer's chat template"})    
    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    
    # logging
    use_log: Optional[bool] = field(default=True, metadata={"help": "whether to use log such as wandb"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    project_name: Optional[str] = field(default="huggingface", metadata={"help": "experiment project name for 'wandb'"})
    run_name: Optional[str] = field(default="test_run", metadata={"help": "experiment run name for 'wandb'"})

    # backend
    ds_config_path: Optional[str] = field(default="./config_trainerlite_ds_02.json", metadata={"help": "deepspeed config path"})
    ds_stage: Optional[int] = field(default=2, metadata={"help": "deepspeed stage. Must be either 2 or 3"})

    # data
    data_save_dir: Optional[str] = field(default="actionstudio/data/train/sft", metadata={"help": "the default dataset dir"})
    data_mix_recipe_yaml_config: Optional[str] = field(
        default="",
        metadata={"help": "the default yaml file for data mixed ratio recipe config"}
    )

    # access
    hf_credential_json_config: Optional[str] = field(default="", metadata={"help": "the json file for HuggingFace credential config"})
    wandb_credential_json_config: Optional[str] = field(default="", metadata={"help": "the json file for Weights and Bias credential config"})
    
    # others
    seed: Optional[int] = field(default=9120, metadata={"help": "the random seed"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer_size: Optional[int] = field(default=500000, metadata={"help": "the shuffle buffer size"})
    mask_prompt_loss: Optional[bool] = field(default=True, metadata={"help": "mask prompt loss or not"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    debug_mode: Optional[bool] = field(default=True, metadata={"help": "turning on debug mode or not"})
    max_verify_steps_per_dataset: Optional[int] = field(default=2, metadata={"help": "the maximum number of data points to verify per dataset"})


def prepare_accelerator(script_args):
    """
    Given the script args, prepare the accelerator engine with deepspeed plugin.
    
    Args:
        script_args: training script args
    
    Returns:
        accelerator: the accelerator engine
    """
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=script_args.ds_config_path,
        zero_stage=int(script_args.ds_stage),
        gradient_accumulation_steps=int(script_args.gradient_accumulation_steps),
        zero3_init_flag=(int(script_args.ds_stage) == 3),
    )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=int(script_args.gradient_accumulation_steps),
        mixed_precision="bf16",
        log_with=script_args.log_with if script_args.use_log else None,
    )

    if script_args.use_log:
        log_with = script_args.log_with
        
        if log_with == "wandb":
            if accelerator.is_main_process:
                with open(script_args.wandb_credential_json_config) as f:
                    wandb_access_content = json.load(f)
                    wandb_host = wandb_access_content["host"]
                    wandb_key = wandb_access_content["api_key"]

                wandb.login(key=wandb_key, relogin=True, host=wandb_host)
            
        accelerator.init_trackers(
            project_name=script_args.project_name,
            init_kwargs={
                log_with: {"name": script_args.run_name}
            }
        )

    return accelerator

def main(script_args) -> None:
    # spin up accelerator
    accelerator = prepare_accelerator(script_args)
        
    # login HF (remove if not use any data from HF)
    if accelerator.is_main_process:
        with open(script_args.hf_credential_json_config, 'r') as f:
            hf_token = json.load(f)["token"]
            login(token=hf_token)
    
    # random seed controller for each device
    seed = init_device_seed(seed=script_args.seed, accelerator=accelerator)
    
    # setup data
    tokenizer, train_dataset, eval_dataset, collator = prepare_data(accelerator, script_args, seed)

    print("train_dataset.dataset.features.keys =", train_dataset.dataset.features.keys())
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=1,
        drop_last=True,
    )
    
    with tqdm(total=script_args.max_verify_steps_per_dataset) as pbar:
        for step, batch in enumerate(train_dataloader):
            print(f"step {step}")
            
            if step > script_args.max_verify_steps_per_dataset: break
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            
            if "llama" in script_args.model_name.lower():
                to_replace_inp_tok = "<|finetune_right_pad_id|>"
                to_replace_tok_id = 128004
                to_replace_tok = "<|finetune_right_pad_id|>"
            else:
                to_replace_inp_tok = "<unk>"
                to_replace_tok_id = 1
                to_replace_tok = "<s>"
            labels[labels == -100] = to_replace_tok_id

            print("---------------------------------------")
            print(f"step {step}")
            print("Encoded data:")
            print("     input_ids =", input_ids)
            print("     labels =", labels)
            print("     attention_mask =", attention_mask)
            print("Decoded data:")
            print("====Input====")
            print(tokenizer.decode(input_ids[0], skip_sepcial_tokens=False).replace(to_replace_inp_tok, ""))
            print()
            print("====Labels====")
            print(tokenizer.decode(labels[0], skip_special_tokens=False).replace(to_replace_tok, ""))
            print("---------------------------------------\n")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
