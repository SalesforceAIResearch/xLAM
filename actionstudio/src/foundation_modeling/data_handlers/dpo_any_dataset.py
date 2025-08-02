import os
from datasets import load_dataset
from actionstudio.src.foundation_modeling.data_handlers.base import SFTFoundationModelDataBaseV2
from actionstudio.src.foundation_modeling import data_handlers


class DPOAnyDatasetLoader(SFTFoundationModelDataBaseV2):
    def __init__(self, dataset_name, tokenizer, args):
        super().__init__(tokenizer, args, args.fc_mode)
        """
        ❤️TODO❤️: This dataset loader is not YET TESTED.
        """
        self.name = "DPOAnyDatasetLoader"
        self.dataset_name = dataset_name
        self.loader_dir = os.path.dirname(data_handlers.__file__)
        

    def create_datasets(self, return_type="basic", seed=None):
        data_file = os.path.join(os.path.join(self.data_save_dir, self.dataset_name), "train.json")
        train_data = load_dataset(
            "json",
            split="train",
            data_files=data_file
        )

        train_data = train_data.rename_column("input", "prompt")
        train_data = train_data.rename_column("output", "chosen")

        original_columns = train_data.column_names

        train_dataset = train_data.map(
            self.prepare_sample_text,
            batched=False,  # batch seems not stable, could throw error?
            num_proc=self.args.num_workers,
            remove_columns=original_columns,
        )

        train_dataset = train_dataset.shuffle(seed=seed)

        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= self.args.seq_length * self.chars_per_token
                      and len(x["prompt"]) + len(x["rejected"]) <= self.args.seq_length * self.chars_per_token
        )

        return train_dataset, None
