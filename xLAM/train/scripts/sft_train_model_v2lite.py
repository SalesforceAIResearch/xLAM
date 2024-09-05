# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from agentstudio.agentstudio_utils import load_yaml_file

from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from huggingface_hub import login
import json
from xLAM.train.fm_datasets \
    import (webshop_multi_turn_v2,
            hotpotqa_multi_turn_v2,
            toolalpaca_multi_turn_v2,
            toolbench_multi_turn_v2,
            apibank_multi_turn_v2,)

from xLAM.train.fm_datasets.base import SFTFoundationModelDataBase
from xLAM.train.fm_utils.interleave_datasets import interleave_data
from xLAM.train.fm_utils.derived_data_collator import DataCollatorForPromptAnswer
from xLAM.train.fm_utils.common import bookkeep_dataset_args, bookkeep_script_args
from xLAM.train.fm_utils.seed_random import init_device_seed
from xLAM.train.fm_trainers.sft_foundation_trainer_lite import SFTFoundationTrainerLite, prepare_accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    use_log: Optional[bool] = field(default=True, metadata={"help": "whether to use log such as wandb"})
    run_name: Optional[str] = field(default="test_run", metadata={"help": "experiment run name for 'wandb'"})
    collator: Optional[str] = field(default="PromptAnswerDataCollator", metadata={"help": "How to collate data for training"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer_size: Optional[int] = field(default=500000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    ds_config_path: Optional[str] = field(default="./config_trainerlite_ds_02.json", metadata={"help": "deepspeed config path"})
    ds_stage: Optional[int] = field(default=2, metadata={"help": "deepspeed stage. Must be either 2 or 3"})
    weight_precision: Optional[str] = field(default="bf16", metadata={"help": "weight precision. Supporting bf16 and nf4"})
    max_steps: Optional[int] = field(default=8000, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=3, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    use_lora: Optional[bool] = field(default=False,
                                     metadata={"help": "if True, use LoRA training, otherwise, do full training"})
    lora_alpha: Optional[float] = field(default=64, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=32, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,w1,w2,w3,gate",
                                               metadata={"help": "target modules for LoRA, separated by commas"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    data_save_dir: Optional[str] = field(
        default="/.", # aws
        metadata={"help": "the default dataset dir"}
    )
    data_mix_recipe_yaml_config: Optional[str] = field(
        default="",
        metadata={"help": "the default yaml file for data mixed ratio recipe config"}
    )
    hf_credential_json_config: Optional[str] = field(
        default="",
        metadata={"help": "the json file for HuggingFace credential config"}
    )
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# hf credential
with open(script_args.hf_credential_json_config) as f:
    login(token=json.load(f)['token'])

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
tokenizer.truncation_side = "left" # We want to keep the label side

assert tokenizer.truncation_side == "left"

sft_webshop_multi_turn = webshop_multi_turn_v2.SFTWebShopMultiTurnV2(tokenizer, script_args)
sft_hotpotqa_multi_turn = hotpotqa_multi_turn_v2.SFTHotpotQAMultiTurnV2(tokenizer, script_args)
sft_toolalpaca_multi_turn = toolalpaca_multi_turn_v2.SFTToolAlpacaMultiTurnV2(tokenizer, script_args)
sft_toolbench_multi_turn = toolbench_multi_turn_v2.SFTToolBenchMultiTurnV2(tokenizer, script_args)
sft_apibank_multi_turn = apibank_multi_turn_v2.SFTAPIBankMultiTurnV2(tokenizer, script_args)

data = [
    sft_webshop_multi_turn,  # 14,082
    sft_hotpotqa_multi_turn,  # 1919
    sft_toolalpaca_multi_turn,  # 8,599
    sft_toolbench_multi_turn,  # 57,843
    sft_apibank_multi_turn,  # 4,902
]

sample_probs = [0.15, 0.02, 0.08, 0.7, 0.05]

# spin up accelerator
accelerator = prepare_accelerator(script_args)

# random seed controller for each device
seed = init_device_seed(seed=42, accelerator=accelerator)

# data collator, datasets and trainer initialization
if script_args.collator == "PromptAnswerDataCollator":
    # train on the generated prompts only
    dataset_base = SFTFoundationModelDataBase(tokenizer=tokenizer, args=None)
    instruction_template = dataset_base.instruction_template
    response_template = dataset_base.response_template

    instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    collator = DataCollatorForPromptAnswer(instruction_template=instruction_template_ids,
                                           response_template=response_template_ids,
                                           tokenizer=tokenizer,
                                           mlm=False)

    train_dataset, eval_dataset = \
        interleave_data(
            data_objects=data,
            sample_probs=sample_probs,
            return_type="prompt_answer",
            seq_length=script_args.seq_length,
            seed=seed)

    trainer = SFTFoundationTrainerLite(
        args=script_args,
        accelerator=accelerator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )
else:
    raise Exception("We only support `PromptAnswerDataCollator` collator")

# train and save
trainer.train()
output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.save_model(output_dir)

# bookkeep script args
bookkeep_script_args(script_args, script_args.output_dir)
bookkeep_dataset_args(data, sample_probs, script_args.output_dir)
