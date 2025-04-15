import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_scheduler,
    HfArgumentParser,
)
import os
from pathlib import Path

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)

from safetensors import safe_open

from huggingface_hub import login
import json

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str) # This is the raw base model path
parser.add_argument("--load_from_state_dict_path", type=str, required=False, default=None)
parser.add_argument("--custom_path", type=str, default="")
parser.add_argument("--output_root", type=str)
parser.add_argument("--save_output", type=bool, default=True)
parser.add_argument("--use_base", type=bool, default=False)
parser.add_argument("--use_16bit", type=bool, default=True)

def load_model(args):
    use_16bit = bool(args.use_16bit)
    model_name_or_path = args.model_name_or_path
    custom_path = args.custom_path
    load_from_state_dict_path = args.load_from_state_dict_path
    
    if use_16bit:
        type_16bit = torch.bfloat16
        print("Load 16bit model")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=type_16bit,
            trust_remote_code=True,
            use_auth_token=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            device_map='auto',
        )

        if load_from_state_dict_path is not None:
            state_dict = dict()
            with safe_open(load_from_state_dict_path, framework="pt", device="cpu") as f:
                for key in f.keys(): state_dict[key] = f.get_tensor(key)
            
            model.load_state_dict(state_dict, strict=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("Load model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    return model, tokenizer

def main(args) -> None:
    model_name_or_path = args.model_name_or_path
    custom_path = args.custom_path
    load_from_state_dict_path = args.load_from_state_dict_path
    
    use_base = bool(args.use_base)
    if load_from_state_dict_path is not None: use_base = True
    
    output_path = os.path.join(args.output_root, "final_merged_checkpoint_torch")

    model, tokenizer = load_model(args)
    if not use_base:
        print("Loading custom model ...")
        custom_model = PeftModel.from_pretrained(model, custom_path).eval()
        merged_model = custom_model.merge_and_unload(progressbar=True)
        print()
    else:
        print("Using base model ...")
        custom_model = model.eval()
        merged_model = custom_model
        print()

    if not os.path.exists(output_path):
        print(f"{output_path} does not exist. Create one now ...")
        os.makedirs(output_path)

    print(f"Saving to {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("❤️Done❤️")
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    