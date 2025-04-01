import warnings
from typing import Any, Dict, List, Union, Optional

import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling


class DataCollatorForPromptAnswer(DataCollatorForLanguageModeling):
    """
    Data collator used for (prompt, answer) tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        instruction_template (`Optional[str]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        disable_warning: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.ignore_index = ignore_index
        self.warning_buffer = 20  # we do not want to dump too many info during training
        self.samples_total = 0
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Given a list of examples in the current batch, prepare the training input ready
        """
        # Note: since we have processed the labels masking logic inside derived_dataSet_v2 already, we only need to directly
        # parse the training format and return here
        
        self.samples_total += len(examples)
        
        batch = {
            "input_ids": torch.stack([e["input_ids"] for e in examples], dim=0),
            "attention_mask": torch.stack([e["attention_mask"] for e in examples], dim=0),
            "labels": torch.stack([e["labels"] for e in examples], dim=0),
        }

        return batch