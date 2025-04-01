from torch.utils.data import IterableDataset
from datasets import interleave_datasets
import warnings
import random
import torch
import copy
from transformers import LlamaTokenizerFast

from actionstudio.src.foundation_modeling.data_handlers.base import SFTFoundationModelDataBaseV2


class PromptAnswerDataset(IterableDataset):
    """"
        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset` or `List[dataset.Dataset]`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        sample_probs,
        seed,
        dataset_text_field=None,
        formatting_func=None,
        mask_prompt_loss=True,
        ignore_index: int = -100,
        infinite=False,
        seq_length=1024,
        num_of_sequences=256,
        eos_token_id=0,
        shuffle=True,
        fc_mode=True,
    ):
        self.tokenizer = tokenizer

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            self.need_response_space = True
        else:
            self.need_response_space = False

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        if isinstance(dataset, list):
            self.dataset = interleave_datasets(dataset, probabilities=sample_probs, seed=seed, stopping_strategy="all_exhausted")
        else:
            self.dataset = dataset
        self.seq_length = seq_length
        self.num_of_sequences = num_of_sequences
        self.infinite = infinite
        self.shuffle = shuffle
        self.current_size = 0

        if formatting_func is None:
            print("Formatting function is None")
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            print("Formatting function is NOT None")
            self.formatting_func = formatting_func
            
        self.mask_prompt_loss = mask_prompt_loss        
        self.ignore_index = ignore_index
        
        print(f"mask_prompt_loss = {self.mask_prompt_loss} -- ignore_index = {self.ignore_index}")

        dataset_base = SFTFoundationModelDataBaseV2(tokenizer=self.tokenizer, args=None, fc_mode=fc_mode)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """
        Iterate through the dataset and prepare the training data, including:
        1) prepare the data in the text-ready format from raw format with `formatting_func`
        2) mask the input data if requested -- i.e., the training target is only output section
        3) postprocess (join, pad, create attention mask, ...)
        """
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.num_of_sequences:
                    break
                try:
                    content = self.formatting_func(next(iterator))
                    if content["input"] == "" and content["chosen"] == "":
                        warnings.warn("Invalid data. Skip!")
                    else:
                        buffer.append(content)
                        buffer_len += 1
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            if self.shuffle: random.shuffle(buffer)
            
            all_parsed_curr_data_instances = []
            for curr_data_instance in buffer:
                context_history_text, output_text = curr_data_instance["input"], curr_data_instance["chosen"]
                
                # tokenize data context history
                tokenized_context_history = self.tokenizer(context_history_text, truncation=True, max_length=self.seq_length)
                
                # tokenize data output response
                tokenized_response = self.tokenizer(output_text, truncation=True, max_length=self.seq_length)
                
                # mask if need
                if self.mask_prompt_loss: tokenized_context_history["labels"] = [self.ignore_index] * len(tokenized_context_history["input_ids"])
                else: tokenized_context_history["labels"] = copy.deepcopy(tokenized_context_history["input_ids"])
                
                if len(tokenized_context_history["input_ids"]) + len(tokenized_response["input_ids"]) >= self.seq_length:
                    warnings.warn("Too long data point. Skip!")
                    continue
                
                # now, join the context history with the response
                parsed_curr_data_instance = {
                    "input_ids": torch.cat([
                        torch.LongTensor(tokenized_context_history["input_ids"]),
                        torch.LongTensor(tokenized_response["input_ids"]),
                        torch.LongTensor([self.tokenizer.pad_token_id] * (self.seq_length - len(tokenized_context_history["input_ids"]) - len(tokenized_response["input_ids"])))
                    ]),
                    "labels": torch.cat([
                        torch.LongTensor(tokenized_context_history["labels"]),
                        torch.LongTensor(tokenized_response["input_ids"]),
                        torch.LongTensor([self.ignore_index] * (self.seq_length - len(tokenized_context_history["labels"]) - len(tokenized_response["input_ids"])))
                    ]),
                    "attention_mask": torch.LongTensor([1] * (len(tokenized_context_history["input_ids"]) + len(tokenized_response["input_ids"])) + [0] * (self.seq_length - (len(tokenized_context_history["input_ids"]) + len(tokenized_response["input_ids"])))),
                }
                
                all_parsed_curr_data_instances.append(parsed_curr_data_instance)
                
            for example in all_parsed_curr_data_instances:
                self.current_size += 1
                yield {
                    "input_ids": example["input_ids"],
                    "labels": example["labels"],
                    "attention_mask":  example["attention_mask"],
                }
