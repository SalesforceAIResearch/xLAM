from datasets import load_dataset, IterableDataset
from train.fm_datasets.base import SFTFoundationModelDataBase
from agentstudio import agents
import os


class SFTHotpotQAMultiTurnV2(SFTFoundationModelDataBase):

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.name = "SFTHotpotQAMultiTurn_v2"
        self.data_save_dir = args.data_save_dir
        self.loader_dir = os.path.dirname(agents.__file__)

    @staticmethod
    def _high_score_filter_generator(data, score=None):
        for d in data:
            yield {"prompt": d["input"], "chosen": d["output"]}

    def create_datasets(self, return_type="constant_length", seed=None):
        # # in total 1919 rows of records
        train_data = load_dataset(
            os.path.join(self.loader_dir, "unified_dataset.py"),
            split="train",
            data_dir=os.path.join(self.data_save_dir, "hotpotqa"),
            max_context_length=self.args.seq_length,
            cache_dir=os.path.join(self.data_save_dir, "monica_cached", "hotpotqa"),
            num_proc = self.args.num_workers if not self.args.streaming else None,
            streaming = self.args.streaming,
        )
        train_data = IterableDataset.from_generator(self._high_score_filter_generator, gen_kwargs={"data": train_data})
        # we randomly shuffle the data to avoid the un-even sampling after data interleave section
        train_data = train_data.shuffle(seed=seed, buffer_size=1950)
        valid_data = load_dataset(
            os.path.join(self.loader_dir, "unified_dataset.py"),
            split="validation",
            data_dir=os.path.join(self.data_save_dir, "hotpotqa"),
            max_context_length=self.args.seq_length,
            cache_dir=os.path.join(self.data_save_dir, "monica_cached", "hotpotqa"),
            num_proc=self.args.num_workers if not self.args.streaming else None,
            streaming=self.args.streaming,
        )
        valid_data = IterableDataset.from_generator(self._high_score_filter_generator, gen_kwargs={"data": valid_data})

        # self.chars_per_token = self.chars_token_ratio(train_data)
        # print(f"The character to token ratio of the dataset is: {self.chars_per_token:.2f}")

        return train_data, valid_data
