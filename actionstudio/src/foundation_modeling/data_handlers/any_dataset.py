import os
import argparse
from datasets import load_dataset, IterableDataset
from actionstudio.src.foundation_modeling.data_handlers.base import SFTFoundationModelDataBaseV2
from actionstudio.src.foundation_modeling import data_handlers


class AnyDatasetLoader(SFTFoundationModelDataBaseV2):
    def __init__(self, dataset_name, tokenizer, args):
        super().__init__(tokenizer, args, args.fc_mode)
        self.name = "AnyDatasetLoader"
        self.data_save_dir = args.data_save_dir
        self.dataset_name = dataset_name
        self.loader_dir = os.path.dirname(data_handlers.__file__)

    def _load_dataset(self, dataset_name, split="train", seed=None):
        """
        Load dataset from the specified directory or Hugging Face dataset.
        
        # trust_remote_code=True to suppress the warning. 
        # The datasets library treats any custom dataset script (even local ones) 
        # as "remote code" that needs explicit trust.
        """
        if os.path.exists(os.path.join(self.data_save_dir, dataset_name)):
            data = load_dataset(
                os.path.join(self.loader_dir, "unified_dataset_simple_v2.py"),
                split=split,
                data_dir=os.path.join(self.data_save_dir, dataset_name),
                max_context_length=self.args.seq_length,
                cache_dir=os.path.join(self.data_save_dir, "monica_cached", dataset_name),
                num_proc = self.args.num_workers if not self.args.streaming else None,
                streaming = self.args.streaming,
                trust_remote_code=True,
            )           
        else:
            try:
                print(f"Loading dataset {dataset_name} with {split} split from Hugging Face...")
                data = load_dataset(
                    dataset_name,
                    split=split,
                    # use_auth_token=True,
                    cache_dir=os.path.join(self.data_save_dir, "monica_cached", dataset_name),
                    num_proc=self.args.num_workers if not self.args.streaming else None,
                    streaming=self.args.streaming,
            )
            except ValueError as e:
                if split == "validation":
                    return None
                else:
                    print(f"Error loading dataset {dataset_name}: {e}")
                    raise ValueError(f"{dataset_name} should appear under the path {self.data_save_dir} or be a valid Hugging Face dataset")
        
        # data = IterableDataset.from_generator(self._high_score_filter_generator, gen_kwargs={"data": data})
        
        ## we randomly shuffle the data to avoid the un-even sampling after data interleave section
        data = data.shuffle(seed=seed, buffer_size=self.args.shuffle_buffer_size)
        return data
        
    def create_datasets(self, return_type="basic", seed=None):
        train_data = self._load_dataset(self.dataset_name, split="train", seed=seed)
        valid_data = self._load_dataset(self.dataset_name, split="validation", seed=seed)
    
        return train_data, valid_data


if __name__ == "__main__":
    """
    unitest for the dataset loader
    """
    parser = argparse.ArgumentParser(description="LLM-based evaluation.")
    parser.add_argument(
        "--data-save-dir",
        type=str,
        default="",
        help="the default dataset dir"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="the name of the dataset to be loaded."
    )
    parser.add_argument(
        "--fc-mode",
        type=bool,
        default=True,
        help="whether to use the fc mode."
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=16324,
        help="the sequence length to load the dataset."
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=500000,
        help="the buffer size to shuffle the dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed to shuffle the dataset."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4, 
        help="the number of workers to load the dataset."
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="whether to stream the dataset."
    )

   
    args = parser.parse_args()
    
    args.data_save_dir = ""
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")
    
    for dataset_name in ["toolace"]:
        sanity_check = AnyDatasetLoader(dataset_name, tokenizer, args)

        train_data, valid_data = sanity_check.create_datasets(args.seed)
        
        print("*" * 50)
        print(f"Dataset: {dataset_name}")
        print("train data info:\n", train_data)
        print("validation data info:\n", valid_data)

