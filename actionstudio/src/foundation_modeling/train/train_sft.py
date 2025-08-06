# Fine-Tune model with SFT (Supervised Fine-Tuning)
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
from actionstudio.src.foundation_modeling.utils.common import load_yaml_file, create_sampled_ratio, save_yaml_file, save_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    # Required fields (no defaults)
    # In dataclasses, you can't have a required field (without default) after optional fields (with defaults).
    model_save_id: str = field(metadata={"help": "unique model id for saving model configuration"})
    
    # model
    model_name: Optional[str] = field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", metadata={"help": "the model name"})
    fc_mode: Optional[bool] = field(default=False, metadata={"help": "whether to use function mode when using Tokenizer's chat template"})    
    enable_thinking: Optional[bool] = field(default=False, metadata={"help": "whether to enable thinking, default is False for qwen3 models"})
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
    min_learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "the minimum learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    scheduler_params: Optional[str] = field(default=None, metadata={"help": "custom scheduler params in format: 'warmup_min_lr=2e-5,warmup_type=linear,warmup_num_steps=200,total_num_steps=3000'"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw", metadata={"help": "the optimizer type"})
    max_steps: Optional[int] = field(default=None, metadata={"help": "the maximum number of sgd steps, use None to allow auto step calculation"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    num_nodes: Optional[int] = field(default=1, metadata={"help": "the number of gpu nodes"})
    num_gpus_per_node: Optional[int] = field(default=8, metadata={"help": "the number of gpus per node"})
    per_device_train_batch_size: Optional[int] = field(default=3, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    save_steps: Optional[int] = field(default=None, metadata={"help": "the checkoint saving frequency (None to save every max_steps)"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})

    # data
    output_dir: Optional[str] = field(default="./checkpoints", metadata={"help": "the output directory for saving model checkpoints"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    data_save_dir: Optional[str] = field(default="actionstudio/data/train/sft", metadata={"help": "the default dataset dir"})
    data_mix_recipe_yaml_config: Optional[str] = field(
        default="", metadata={"help": "the default yaml file path for data mixed ratio recipe config"}
    )
    data_mix_or_unify: Optional[str] = field(
        default="unify", metadata={"help": "whether to mix data (i.e., add general data) or unify data (i.e., only use agent data). Options: ['mix', 'unify']"}
    )
    is_data_pre_verification: Optional[bool] = field(
        default=False, metadata={"help": "whether to conduct data pre-verification before training"}
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


def prepare_accelerator(script_args, num_total_data):
    """
    Given the script args, prepare the accelerator engine with deepspeed plugin.
    
    Args:
        script_args: training script args
        num_total_data: total number of data samples
    
    Returns:
        accelerator: the accelerator engine
    """
    # Load the base DeepSpeed config
    with open(script_args.ds_config_path, 'r') as f:
        ds_config = json.load(f)
    
    # Update optimizer configuration if needed
    # Configure AdamW optimizer for DeepSpeed with standard settings
    optimizer_config = {
        "type": script_args.optimizer_type,
        "params": {
            "lr": script_args.learning_rate,
            "betas": [0.9, 0.999],  # Standard AdamW settings
            "eps": 1e-8,
            "weight_decay": script_args.weight_decay,
            "adam_w_mode": True,  # Ensure AdamW mode (decoupled weight decay)
            "torch_adam": True,   # Use PyTorch's AdamW instead of DeepSpeed's fused version
        }
    }
    ds_config["optimizer"] = optimizer_config
    # Only print once (will be shown when DeepSpeed prints full config anyway)
    
    # set max_steps (if not provided)
    if script_args.max_steps is None:
        script_args.max_steps = num_total_data // (script_args.per_device_train_batch_size * script_args.num_nodes * script_args.num_gpus_per_node)
    
    # by default, save the model every half of the max_steps
    if script_args.save_steps is None:
        script_args.save_steps = script_args.max_steps // 2
    
    # Calculate optimizer steps (max_steps is already per-process)
    num_optimizer_steps = script_args.max_steps // script_args.gradient_accumulation_steps

    # Update scheduler configuration if needed
    # Always configure scheduler to ensure it matches the training parameters
    # Map lr_scheduler_type to DeepSpeed scheduler type
    scheduler_type_mapping = {
        "cosine": "WarmupCosineLR",
        "linear": "WarmupLR", 
        "constant": "WarmupLR",
        "constant_with_warmup": "WarmupLR",
        "polynomial": "WarmupDecayLR",
        "cosine_with_restarts": "WarmupCosineLR",  # DeepSpeed doesn't have restarts, use regular cosine
    }
    
    ds_scheduler_type = scheduler_type_mapping.get(script_args.lr_scheduler_type, "WarmupCosineLR")
    
    # Default scheduler config
    scheduler_config = {
        "type": ds_scheduler_type,
        "params": {}
    }
    
    # Configure parameters based on scheduler type
    if ds_scheduler_type == "WarmupCosineLR":
        # WarmupCosineLR uses ratios, not absolute learning rates
        # warmup_min_rato = max(base_lr / 10, min_lr) / base_lr
        warmup_min_ratio = max(script_args.learning_rate / 10, script_args.min_learning_rate) / script_args.learning_rate
        scheduler_config["params"] = {
            "warmup_min_ratio": warmup_min_ratio,
            "warmup_num_steps": script_args.num_warmup_steps,
            "cos_min_ratio": script_args.min_learning_rate / script_args.learning_rate,  # End learning rate as ratio
            "total_num_steps": num_optimizer_steps,
        }
    elif ds_scheduler_type == "WarmupDecayLR":
        scheduler_config["params"] = {
            "warmup_min_lr": max(script_args.learning_rate / 10, script_args.min_learning_rate),
            "warmup_max_lr": script_args.learning_rate,
            "warmup_num_steps": script_args.num_warmup_steps,
            "total_num_steps": num_optimizer_steps,
        }
    else:  # WarmupLR
        scheduler_config["params"] = {
            "warmup_min_lr": max(script_args.learning_rate / 10, script_args.min_learning_rate),
            "warmup_max_lr": script_args.learning_rate,
            "warmup_num_steps": script_args.num_warmup_steps,
            "warmup_type": "linear",
        }
    
    # Update with custom parameters if provided
    if script_args.scheduler_params:
        # Parse scheduler params (expected format: "warmup_min_lr=1e-5,warmup_type=linear")
        for param in script_args.scheduler_params.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                # Convert numeric strings to appropriate types
                if key in ['warmup_min_lr', 'warmup_max_lr', 'warmup_min_ratio', 'cos_min_ratio']:
                    value = float(value)
                elif key in ['warmup_num_steps', 'total_num_steps']:
                    value = int(value)
                scheduler_config["params"][key] = value
    
    # Update the config
    ds_config["scheduler"] = scheduler_config
    # Scheduler config will be shown when DeepSpeed prints full config
    
    # Update train_micro_batch_size_per_gpu to match script_args
    ds_config["train_micro_batch_size_per_gpu"] = script_args.per_device_train_batch_size
    
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config,  # Pass the dict directly instead of file path
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

    return accelerator, ds_config

def prepare_data(accelerator, script_args, ds_config, seed=9120, sample_probs=None, yaml_data=None, num_total_data=None):
    """
    Given the script args and optionally the random seed, prepare the tokenizer, train, and eval dataset and the data collator for training.
    
    Args:
        script_args: training script args
        ds_config: DeepSpeed configuration from prepare_accelerator
        seed: random seed, default is 9120
        sample_probs: sample probabilities for data mixing
        yaml_data: data mix recipe yaml data
        num_total_data: total number of data samples

    Returns:
        tokenizer: tokenizer object
        train_dataset: the train dataset object
        eval_dataset: the eval dataset object
        collator: data collator
    """
    # tokenizer
    if accelerator.is_main_process: 
        print("Loading tokenizer from:\n ", script_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, add_bos_token=False)      # note: it is important to not add bos here, since apply_chat_template is already doing that
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.truncation_side = "left" # We want to keep the label side

    if "llama" in script_args.model_name.lower():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    elif "pad_token" in tokenizer.special_tokens_map: # all qwen2.5 & qwen3 models has same pad token (<|endoftext|>) and pad token id (151643)
        tokenizer.pad_token = tokenizer.special_tokens_map["pad_token"]
        tokenizer.pad_token_id = tokenizer.pad_token_id
    else:
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0

    # only want to check the `calculated_steps` when doing real training, not during data verifying
    if not script_args.is_data_pre_verification:     
        # Calculate the max training steps. Note the num_optimization_updates = max_steps / gradient_accumulation_steps
        # accelerator.num_processes = script_args.num_nodes * script_args.num_gpus_per_node
        calculated_steps = num_total_data // (script_args.per_device_train_batch_size * accelerator.num_processes)
        
        # If max_steps is provided, validate it matches calculated value
        if accelerator.is_main_process and not script_args.debug_mode and script_args.max_steps != calculated_steps:
            raise ValueError(
                f"❤️ Provided max_steps ({script_args.max_steps}) doesn't match calculated steps ({calculated_steps}) ❤️. "
                f"Please check your data configuration and batch settings!"
            )
        
        if accelerator.is_main_process:
            # Print training configuration details
            num_optimizer_steps = script_args.max_steps // script_args.gradient_accumulation_steps
            print(f"\n[Training Configuration]")
            print(f"  Total data samples: {num_total_data}")
            print(f"  Data mix or unify: {script_args.data_mix_or_unify}")
            print(f"  Per-device batch size: {script_args.per_device_train_batch_size}")
            print(f"  Number of processes (num_nodes * num_gpus_per_node): {accelerator.num_processes}")
            print(f"  Gradient accumulation steps: {script_args.gradient_accumulation_steps}")
            print(f"  Total batch size: {script_args.per_device_train_batch_size * accelerator.num_processes * script_args.gradient_accumulation_steps}")
            print(f"  Total training steps: {script_args.max_steps}")
            print(f"  Total optimizer steps: {num_optimizer_steps}")
            print(f"  Use lora: {script_args.use_lora}")
            print(f"  Use fc_mode: {script_args.fc_mode}")
            print(f"  Enable thinking: {script_args.enable_thinking}")
            print(f"  Mask prompt loss: {script_args.mask_prompt_loss}")
            print(f"  Learning rate: {script_args.learning_rate}")
            print(f"  Min learning rate: {script_args.min_learning_rate}")
            print(f"  Lr scheduler type: {script_args.lr_scheduler_type}")
            print(f"  Num warmup steps: {script_args.num_warmup_steps}")
            print(f"  Scheduler params: {script_args.scheduler_params}")
            print(f"  Weight decay: {script_args.weight_decay}")
            print(f"  Optimizer type: {script_args.optimizer_type}")
            print(f"  Weight precision: {script_args.weight_precision}")
            print(f"  DS stage: {script_args.ds_stage}")
            print(f"  DS config path: {script_args.ds_config_path}")
            print(f"  Gradient checkpointing: {script_args.gradient_checkpointing}")
            print(f"  Debug mode: {script_args.debug_mode}")
            print(f"  Data mix recipe yaml config: {script_args.data_mix_recipe_yaml_config}")
            print(f"  Data save dir: {script_args.data_save_dir}")
            print(f"  Is data pre verification: {script_args.is_data_pre_verification}")
            print(f"\n  → Full configuration saved to: model_config_files/{script_args.model_save_id}.json")
            print()
                     
            # Save all configurations to JSON file
            config_to_save = {
                "training_configuration": {
                    "total_data_samples": num_total_data,
                    "data_mix_or_unify": script_args.data_mix_or_unify,
                    "per_device_train_batch_size": script_args.per_device_train_batch_size,
                    "per_device_eval_batch_size": script_args.per_device_eval_batch_size,
                    "num_processes (num_nodes * num_gpus_per_node)": accelerator.num_processes,
                    "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
                    "total_train_batch_size": script_args.per_device_train_batch_size * accelerator.num_processes * script_args.gradient_accumulation_steps,
                    "total_training_steps": script_args.max_steps,
                    "total_optimizer_steps": num_optimizer_steps,
                    "save_steps": script_args.save_steps,
                    "logging_steps": script_args.logging_steps,
                },
                "model_configuration": {
                    "model_name": script_args.model_name,
                    "seq_length": script_args.seq_length,
                    "use_lora": script_args.use_lora,
                    "fc_mode": script_args.fc_mode,
                    "enable_thinking": script_args.enable_thinking,
                    "mask_prompt_loss": script_args.mask_prompt_loss,
                    "gradient_checkpointing": script_args.gradient_checkpointing,
                },
                "optimizer_configuration": {
                    "optimizer_type": script_args.optimizer_type,
                    "learning_rate": script_args.learning_rate,
                    "min_learning_rate": script_args.min_learning_rate,
                    "weight_decay": script_args.weight_decay,
                },
                "scheduler_configuration": {
                    "lr_scheduler_type": script_args.lr_scheduler_type,
                    "num_warmup_steps": script_args.num_warmup_steps,
                    "scheduler_params": script_args.scheduler_params,
                },
                "deepspeed_configuration": {
                    "ds_stage": script_args.ds_stage,
                    "ds_config_path": script_args.ds_config_path,
                    "weight_precision": script_args.weight_precision,
                },
                "data_configuration": {
                    "data_mix_recipe_yaml_config": script_args.data_mix_recipe_yaml_config,
                    "data_save_dir": script_args.data_save_dir,
                    "is_data_pre_verification": script_args.is_data_pre_verification,
                    "streaming": script_args.streaming,
                    "shuffle_buffer_size": script_args.shuffle_buffer_size,
                    "num_workers": script_args.num_workers,
                },
                "logging_configuration": {
                    "use_log": script_args.use_log,
                    "log_with": script_args.log_with,
                    "project_name": script_args.project_name,
                    "run_name": script_args.run_name,
                    "model_save_id": script_args.model_save_id,
                    "model_output_dir": script_args.output_dir,
                    "log_freq": script_args.log_freq,
                },
                "other_configuration": {
                    "seed": script_args.seed,
                    "debug_mode": script_args.debug_mode,
                },
            }
            
            # Add LoRA configuration if enabled
            if script_args.use_lora:
                config_to_save["lora_configuration"] = {
                    "lora_alpha": script_args.lora_alpha,
                    "lora_dropout": script_args.lora_dropout,
                    "lora_r": script_args.lora_r,
                    "lora_target_modules": script_args.lora_target_modules,
                }
            
            # Load and add the actual DeepSpeed configuration
            # ds_config is already passed from prepare_accelerator with all configurations set
            config_to_save["deepspeed_actual_full_config"] = ds_config
            
            # Also save the DeepSpeed scheduler config separately for easy access
            config_to_save["deepspeed_full_scheduler_config"] = ds_config["scheduler"]
            config_to_save["deepspeed_full_scheduler_config"]["params"]["begin_lr"] = script_args.learning_rate * ds_config["scheduler"]["params"]["warmup_min_ratio"]
            config_to_save["deepspeed_full_scheduler_config"]["params"]["end_lr"] = script_args.learning_rate * ds_config["scheduler"]["params"]["cos_min_ratio"]

            
            # Save to JSON file
            os.makedirs("model_config_files", exist_ok=True)
            save_json(os.path.join("model_config_files", f"{script_args.model_save_id}.json"), config_to_save)

            print(f"Model Configuration saved to: model_config_files/{script_args.model_save_id}.json\n")
            print()
            
            new_path = script_args.data_mix_recipe_yaml_config.replace(".yaml", "") + "--processed.yaml"
            save_yaml_file(new_path, yaml_data)

    accelerator.wait_for_everyone()
     
    data = []
    for selected_dataset_name in yaml_data:
        data.append(AnyDatasetLoader(selected_dataset_name, tokenizer, script_args))

    # load collator
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
            script_args=script_args,
            seq_length=script_args.seq_length,
            fc_mode=script_args.fc_mode,
            enable_thinking=script_args.enable_thinking,
            mask_prompt_loss=script_args.mask_prompt_loss,
            seed=seed)

    return tokenizer, train_dataset, eval_dataset, collator

def main(script_args):
    # load data mix config
    yaml_data = load_yaml_file(script_args.data_mix_recipe_yaml_config)
    sample_probs, yaml_data, num_total_data = create_sampled_ratio(yaml_data)
      
    # spin up accelerator
    accelerator, ds_config = prepare_accelerator(script_args, num_total_data)
        
    # login HF (remove if not use any data from HF)
    if accelerator.is_main_process:
        with open(script_args.hf_credential_json_config, 'r') as f:
            hf_token = json.load(f)["token"]
            login(token=hf_token)
    accelerator.wait_for_everyone()
    
    # random seed controller for each device
    seed = init_device_seed(seed=script_args.seed, accelerator=accelerator)
    
    # setup data
    tokenizer, train_dataset, eval_dataset, collator = prepare_data(accelerator, script_args, ds_config, seed, sample_probs, yaml_data, num_total_data)

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