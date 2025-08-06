import json
import os
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--trained_model_root", type=str, help="the path to the trained model")
parser.add_argument("--base_model_root", type=str, help="the path to the base model")
parser.add_argument("--use_16bit", type=bool, help="whether to use 16bit model", required=False, default=True)
parser.add_argument("--adapter_config_template", type=str, help="the path to adapter config template", required=False, default="")

def main(script_args) -> None:
    base_model_root = script_args.base_model_root
    trained_model_root = script_args.trained_model_root
    
    if script_args.adapter_config_template != "":
        with open(script_args.adapter_config_template) as f: adapter_content = json.load(f)
        print(json.dumps(adapter_content, indent=4))

    print("*" * 50)
    print(f"Post-processing for {trained_model_root}")
    
    if "lora" in trained_model_root.lower():
        raw_model_dir = base_model_root
        output_dir = f"{trained_model_root}"
        custom_path = f"{output_dir}/final_checkpoint"
        
        if not os.path.exists(custom_path):
            raise Exception(f"Path not exist:", custom_path)

        if 'adapter_config.json' not in os.listdir(custom_path):
            with open(f"{custom_path}/adapter_config.json", "w") as f: json.dump(adapter_content, f, indent=4)

        print("     Using LoRA")
        print("         raw_model_dir", raw_model_dir)
        print("         output_dir", output_dir)
        print("         custom_path", custom_path)
        subprocess.call([
            'python',
            'sft_torch_export.py',
            f"--model_name_or_path", raw_model_dir,
            f"--custom_path", custom_path,
            f"--output_root", output_dir,
            f"--use_16bit", str(script_args.use_16bit),
        ])
    else:
        raw_model_dir = base_model_root
        state_dict_path = f"{trained_model_root}/final_checkpoint/full_model.safetensors"
        output_dir = f"{trained_model_root}"

        if not os.path.exists(state_dict_path):
            raise Exception(f"Path not exist:", state_dict_path)

        print("     Using Full training")
        print("         raw_model_dir", raw_model_dir)
        print("         output_dir", output_dir)
        print("         state_dict_path", state_dict_path)
        subprocess.call([
            'python',
            'sft_torch_export.py',
            f"--model_name_or_path", raw_model_dir,
            f"--load_from_state_dict_path", state_dict_path,
            f"--output_root", output_dir,
        ])


if __name__ == "__main__":
    script_args = parser.parse_args()
    main(script_args)

