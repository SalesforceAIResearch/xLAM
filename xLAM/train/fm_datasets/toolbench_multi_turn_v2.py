from datasets import load_dataset, IterableDataset
from xLAM.train.fm_datasets.base import SFTFoundationModelDataBase
from agentstudio import agents
import os


class SFTToolBenchMultiTurnV2(SFTFoundationModelDataBase):

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.name = "SFTToolBenchMultiTurn_v2"
        self.data_save_dir = args.data_save_dir
        self.loader_dir = os.path.dirname(agents.__file__)

    def create_datasets(self, return_type="constant_length", seed=None):
        # in total 57843 rows of records
        train_data = load_dataset(
            os.path.join(self.loader_dir, "unified_dataset.py"),
            split="train",
            data_dir=os.path.join(self.data_save_dir, "toolbench_fixed"),
            max_context_length=self.args.seq_length,
            is_dpo_first_turn=False,
            cache_dir=os.path.join(self.data_save_dir, "monica_cached", "toolbench_fixed"),
            num_proc = self.args.num_workers if not self.args.streaming else None,
            streaming = self.args.streaming,
        )
        # we randomly shuffle the data to avoid the un-even sampling after data interleave section
        train_data = train_data.shuffle(seed=seed, buffer_size=60000)

        train_data = train_data.rename_column("input", "prompt")
        train_data = train_data.rename_column("output", "chosen")
        # For now, we do not have validation data

        valid_data = None

        return train_data, valid_data
