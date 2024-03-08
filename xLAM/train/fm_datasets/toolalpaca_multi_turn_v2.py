from datasets import load_dataset, IterableDataset
from xLAM.train.fm_datasets.base import SFTFoundationModelDataBase
from agentstudio import agents
import os


class SFTToolAlpacaMultiTurnV2(SFTFoundationModelDataBase):

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.name = "SFTToolAlpacaMultiTurn_v2"
        self.data_save_dir = args.data_save_dir
        self.loader_dir = os.path.dirname(agents.__file__)

    def create_datasets(self, return_type="constant_length", seed=None):
        # in total 8599 rows of records
        train_data = load_dataset(
            os.path.join(self.loader_dir, "unified_dataset.py"),
            split="train",
            data_dir=os.path.join(self.data_save_dir, "toolalpaca"),
            max_context_length=self.args.seq_length,
            cache_dir=os.path.join(self.data_save_dir, "monica_cached", "toolalpaca"),
            num_proc = self.args.num_workers if not self.args.streaming else None,
            streaming = self.args.streaming,
        )
        # we randomly shuffle the data to avoid the un-even sampling after data interleave section
        train_data = train_data.shuffle(seed=seed, buffer_size=9000)

        train_data = train_data.rename_column("input", "prompt")
        train_data = train_data.rename_column("output", "chosen")
        # For now, we do not have validation data

        valid_data = None

        return train_data, valid_data
