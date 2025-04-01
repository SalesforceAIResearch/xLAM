from pathlib import Path
from dataclasses import asdict, is_dataclass
import json
import yaml
import jsonlines

def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent


def bookkeep_script_args(script_args, output_path):
    if is_dataclass(script_args):
        args_dict = asdict(script_args)
        with open(f"{output_path}/script_args.json", "w") as outfile:
            json.dump(args_dict, outfile)
    else:
        raise Exception("bookkeep_script_args() expects a dataclass object")


def bookkeep_dataset_args(datasets, sample_probs, output_path):
    assert len(datasets) == len(sample_probs)
    dataset_names = []
    for d in datasets:
        dataset_names.append(d.name)

    dataset_dict = {"datasets": [(d, p) for d, p in zip(dataset_names, sample_probs)]}

    with open(f"{output_path}/dataset_args.json", "w") as outfile:
        json.dump(dataset_dict, outfile)

def open_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def open_jsonl(filepath):
    reader = open(filepath, "r", encoding="utf-8")
    data = jsonlines.Reader(reader)
    return list(data)


def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error in loading YAML file:", exc)

def create_sampled_ratio(yaml_data):
    total_data = 0
    for dataset_name in yaml_data.keys():
        total_data += int(yaml_data[dataset_name]["size"] * yaml_data[dataset_name]["epochs"])
        
    all_sampled_ratios = 0.
    sample_probs = []
    for dataset_name in yaml_data.keys():
        sampled_training_examples = int(yaml_data[dataset_name]["size"] * yaml_data[dataset_name]["epochs"])
        yaml_data[dataset_name]["sampled_training_examples"] = sampled_training_examples
        yaml_data[dataset_name]["sampled_training_ratio"] = sampled_training_examples * 1. / total_data
        all_sampled_ratios += yaml_data[dataset_name]["sampled_training_ratio"]
    
        sample_probs.append(yaml_data[dataset_name]["sampled_training_ratio"])
        
    return sample_probs

def dict_to_json_print_format(d):
    return json.dumps(d, indent=4)