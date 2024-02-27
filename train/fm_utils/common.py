from pathlib import Path
from dataclasses import asdict, is_dataclass
import json


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



