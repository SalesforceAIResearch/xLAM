# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

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
from xLAM.train.fm_trainers.sft_foundation_trainer import SFTFoundationTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="agent_foundation_Mixtral-8x7B-Instruct", metadata={"help": "experiment run name for 'wandb'"})
    collator: Optional[str] = field(default="PromptAnswerDataCollator", metadata={"help": "How to collate data for training"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=8000, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=3, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    ds_config_path: Optional[str] = field(default="./ds_zero2.json", metadata={"help": "deepspeed config path"})
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=32, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    data_save_dir: Optional[str] = field(
        default = "",
        metadata={"help": "the default dataset dir"}
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "fc_in",
                    "fc_out",
                    "wte"
                    ],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
tokenizer.truncation_side = "left" # We want to keep the label side

assert tokenizer.truncation_side == "left"

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    tf32=True,
    deepspeed=script_args.ds_config_path,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    run_name=script_args.run_name,
)

sft_webshop_multi_turn = webshop_multi_turn_v2.SFTWebShopMultiTurnV2(tokenizer, script_args)
sft_hotpotqa_multi_turn = hotpotqa_multi_turn_v2.SFTHotpotQAMultiTurnV2(tokenizer, script_args)
sft_toolalpaca_multi_turn = toolalpaca_multi_turn_v2.SFTToolAlpacaMultiTurnV2(tokenizer, script_args)
sft_toolbench_multi_turn = toolbench_multi_turn_v2.SFTToolBenchMultiTurnV2(tokenizer, script_args)
sft_apibank_multi_turn = apibank_multi_turn_v2.SFTAPIBankMultiTurnV2(tokenizer, script_args)

data = [
    sft_webshop_multi_turn,  # 14,082
    sft_hotpotqa_multi_turn,  # 1919
    sft_toolalpaca_multi_turn,  # 8,599
    sft_toolbench_multi_turn,  # 57843
    sft_apibank_multi_turn,  # 4,902
]

sample_probs = [0.15, 0.02, 0.08, 0.7, 0.05]

seed = init_device_seed(seed=42) # we have device specific seed control

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

    trainer = SFTFoundationTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=False,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )
else:
    raise Exception("This script is designed to use `DataCollatorForPromptAnswer`")

trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

if isinstance(collator, DataCollatorForPromptAnswer):
    print("***** Final sample usage summary *****")
    print(f"samples_total: {collator.samples_total}")
    print(f"samples skip due to no response: {collator.samples_skip_no_response}")
    print(f"samples skip due to no instruct: {collator.samples_skip_no_instruct}")

bookkeep_script_args(script_args, script_args.output_dir)
bookkeep_dataset_args(data, sample_probs, script_args.output_dir)

