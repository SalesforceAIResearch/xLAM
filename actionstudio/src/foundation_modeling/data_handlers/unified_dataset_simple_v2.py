import json
import os
import random
import datasets

from actionstudio.src.foundation_modeling.utils.common import open_json

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """
This is a base dataset file for the unified dataset.
"""

_URL = """@article{zhang2025actionstudio,
  title={ActionStudio: A Lightweight Framework for Data and Training of Action Models},
  author={Zhang, Jianguo and Hoang, Thai and Zhu, Ming and Liu, Zuxin and Wang, Shiyu and Awalgaonkar, Tulika and Prabhakar, Akshara and Chen, Haolin and Yao, Weiran and Liu, Zhiwei and others},
  journal={arXiv preprint arXiv:2503.22673},
  year={2025}
}"""


class DataConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_context_length=None, is_dpo_first_turn=False, rated_traj_score_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_context_length = max_context_length
        self.is_dpo_first_turn = is_dpo_first_turn
        self.rated_traj_score_threshold = rated_traj_score_threshold
       

class UnifiedData(datasets.GeneratorBasedBuilder):
    """Base File."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = DataConfig
    BUILDER_CONFIGS = [
        DataConfig(name="default", description="Default config")
    ]
    DEFAULT_CONFIG_NAME = "default"
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "messages": datasets.Value("string"),
                    "prompt": datasets.Value("string"),
                    "chosen": datasets.Value("string"),
                    "tools": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        split_dir = self.config.data_dir
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train/train.json"), 
                    "max_context_length": self.config.max_context_length,
                    "is_dpo_first_turn": self.config.is_dpo_first_turn,
                    "rated_traj_score_threshold": self.config.rated_traj_score_threshold,
                    "subset": "train"
                }),
             datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train/train.json"),
                    "max_context_length": self.config.max_context_length,
                    "is_dpo_first_turn": self.config.is_dpo_first_turn,
                    "rated_traj_score_threshold": self.config.rated_traj_score_threshold,
                    "subset": "validation"
                })
        ]
        
    def _generate_examples(self, 
                           path=None, 
                           max_context_length=None, 
                           subset=None,
                           is_dpo_first_turn=False,
                           rated_traj_score_threshold=None
                           ):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
    
        data = open_json(os.path.join(path)) 
        count_idx = 0
        for example in data:        
            count_idx += 1
            yield f"{count_idx}", example
