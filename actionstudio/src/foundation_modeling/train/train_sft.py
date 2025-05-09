# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from accelerate import  DeepSpeedPlugin, Accelerator
from typing import Optional
import wandb
import yaml
import json
import torch
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login

from actionstudio.src.foundation_modeling.trainers.sft_foundation_trainer import SFTFoundationTrainerLite
from actionstudio.src.foundation_modeling.data_handlers.any_dataset import AnyDatasetLoader
from actionstudio.src.foundation_modeling.data_handlers.derived_data_collator import DataCollatorForPromptAnswer
from actionstudio.src.foundation_modeling.data_handlers.interleave_datasets import interleave_data

from actionstudio.src.foundation_modeling.utils.seed_random import init_device_seed
from actionstudio.src.foundation_modeling.utils.common import load_yaml_file, create_sampled_ratio, save_yaml_file

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
    weight_precision: Optional[str] = field(default="bf16", metadata={"help": "weight precision. Supporting bf16 and nf4"})
    ds_config_path: Optional[str] = field(default="./config_trainerlite_ds_02.json", metadata={"help": "deepspeed config path"})
    ds_stage: Optional[int] = field(default=2, metadata={"help": "deepspeed stage. Must be either 2 or 3"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    # lora
    use_lora: Optional[bool] = field(default=False, metadata={"help": "if True, use LoRA training, otherwise, do full training"})
    lora_alpha: Optional[float] = field(default=64, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=32, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,w1,w2,w3,gate", metadata={"help": "target modules for LoRA, separated by commas"})
    
    # training
    mask_prompt_loss: Optional[bool] = field(default=True, metadata={"help": "mask prompt loss or not"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    max_steps: Optional[int] = field(default=None, metadata={"help": "the maximum number of sgd steps, use None to allow auto step calculation"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    per_device_train_batch_size: Optional[int] = field(default=3, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    save_steps: Optional[int] = field(default=None, metadata={"help": "the checkoint saving frequency (None to save every max_steps)"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})

    # data
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    data_save_dir: Optional[str] = field(default="actionstudio/data/train/sft", metadata={"help": "the default dataset dir"})
    data_mix_recipe_yaml_config: Optional[str] = field(
        default="",
        metadata={"help": "the default yaml file path for data mixed ratio recipe config"}
    )
    is_data_verfication: Optional[bool] = field(
        default=False, metadata={"help": "whether to conduct data verification"}
    )

    # access
    hf_credential_json_config: Optional[str] = field(default="", metadata={"help": "the json file for HuggingFace credential config"})
    wandb_credential_json_config: Optional[str] = field(default="", metadata={"help": "the json file for Weights and Bias credential config"})
    
    # others
    seed: Optional[int] = field(default=9120, metadata={"help": "the random seed"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer_size: Optional[int] = field(default=500000, metadata={"help": "the shuffle buffer size"})
    debug_mode: Optional[bool] = field(default=False, metadata={"help": "turning on debug mode or not"})


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
            
            accelerator.wait_for_everyone()
            
        accelerator.init_trackers(
            project_name=script_args.project_name,
            init_kwargs={
                log_with: {"name": script_args.run_name}
            }
        )

    return accelerator

def prepare_data(accelerator, script_args, seed=9120):
    """
    Given the script args and optionally the random seed, prepare the tokenizer, train, and eval dataset and the data collator for training.
    
    Args:
        script_args: training script args
        
    Returns:
        tokenizer: tokenizer object
        train_dataset: the train dataset object
        eval_dataset: the eval dataset object
        collator: data collator
    """
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, add_bos_token=False)      # note: it is important to not add bos here, since apply_chat_template is already doing that
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.truncation_side = "left" # We want to keep the label side

    if "llama" in script_args.model_name.lower():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    elif "pad_token" in tokenizer.special_tokens_map:
        tokenizer.pad_token = tokenizer.special_tokens_map["pad_token"]
        tokenizer.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0

    # load data mix config
    yaml_data = load_yaml_file(script_args.data_mix_recipe_yaml_config)
    sample_probs, yaml_data, num_total_data = create_sampled_ratio(yaml_data)

    if not script_args.is_data_verfication:
        # Calculate the max training steps. Note the num_optimization_updates = max_steps / gradient_accumulation_steps
        calculated_steps = num_total_data // (script_args.per_device_train_batch_size * accelerator.num_processes)
        
        # If max_steps is provided, validate it matches calculated value
        if accelerator.is_main_process and not script_args.debug_mode and script_args.max_steps is not None and script_args.max_steps != calculated_steps:
            raise ValueError(
                f"❤️ Provided max_steps ({script_args.max_steps}) doesn't match calculated steps ({calculated_steps}) ❤️. "
                f"Please check your data configuration and batch settings!"
            )
        
        # Set max_steps to calculated value if not explicitly provided
        if script_args.max_steps is None:
            script_args.max_steps = calculated_steps
        # By default, save the model every half of the max_steps
        if script_args.save_steps is None:
            script_args.save_steps = script_args.max_steps // 2
        
        if accelerator.is_main_process:
            new_path = script_args.data_mix_recipe_yaml_config.replace(".yaml", "") + "--processed.yaml"
            save_yaml_file(new_path, yaml_data)

    accelerator.wait_for_everyone()
     
    data = []
    for selected_dataset_name in yaml_data:
        data.append(AnyDatasetLoader(selected_dataset_name, tokenizer, script_args))

    # load collator
    if accelerator.is_main_process: print("using fc_mode:", script_args.fc_mode)
    collator = DataCollatorForPromptAnswer(
        tokenizer=tokenizer,
        mlm=False
    )

    # load train & eval datasets
    train_dataset, eval_dataset = \
        interleave_data(
            accelerator=accelerator,
            data_objects=data,
            sample_probs=sample_probs,
            return_type="prompt_answer",
            seq_length=script_args.seq_length,
            fc_mode=script_args.fc_mode,
            mask_prompt_loss=script_args.mask_prompt_loss,
            seed=seed)

    return tokenizer, train_dataset, eval_dataset, collator

def main(script_args):
    # spin up accelerator
    accelerator = prepare_accelerator(script_args)
        
    # login HF (remove if not use any data from HF)
    if accelerator.is_main_process:
        with open(script_args.hf_credential_json_config, 'r') as f:
            hf_token = json.load(f)["token"]
            login(token=hf_token)
    accelerator.wait_for_everyone()
    
    # random seed controller for each device
    seed = init_device_seed(seed=script_args.seed, accelerator=accelerator)
    
    # setup data
    tokenizer, train_dataset, eval_dataset, collator = prepare_data(accelerator, script_args, seed)

    # setup trainer
    trainer = SFTFoundationTrainerLite(
        args=script_args,
        accelerator=accelerator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )

    # train and save
    trainer.train()

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
