from tqdm import tqdm
from transformers import LlamaTokenizerFast
import ast
import warnings


class SFTFoundationModelDataBaseV2:

    def __init__(self, tokenizer, args, fc_mode=True):
        self.tokenizer = tokenizer
        self.args = args
        self.fc_mode = fc_mode                    # inside main code we need fc_mode as an arg
        
        self.name = "SFTFoundationModelDataBaseV2"

        self.token_try_run = False

        print("tokenizer.name_or_path =", tokenizer.name_or_path)
        self.chars_per_token = 3.0 # a default chars per token

    @property
    def instruction_template(self):
        # return self._instruction_template
        return ""           # TODO: temporary for now, later we can remove this function

    @property
    def response_template(self):
        # return self._response_template
        return ""           # TODO: temporary for now, later we can remove this function

    def _prepare_single_sample_text(self, example, has_reject=False):
        """
        Given a single example, prepare the training-ready text and return output in the format of
        {
            "input": "<input_text>",
            "chosen": "<output_text>",
            "reject" "<optional_field_if_has_reject>"
        }
        General instruction data
        {
            "prompt": "<input_text>",
            "chosen: "<output_text>",
        }
        """
        # print("example =", example)
        if "messages" not in example.keys() or example["messages"] is None:
            if "prompt" not in example.keys() or example["prompt"] is None:
                warnings.warn(
                "Cannot find key \"messages\" AND \"prompt\"! Invalid data!"
                )

                return {
                    "input": "",
                    "chosen": "",
                }

            example["messages"] = str([{
                "role": "user",
                "content": example["prompt"]
            }])
            
            example["tools"] = str([])

        try:
            messages = ast.literal_eval(example["messages"])
            chosen = example["chosen"]
            tools = ast.literal_eval(example["tools"])
            
            if len(chosen.strip()) == 0:
                warnings.warn(
                "Empty \"chosen\"!"
                )

                return {
                    "input": "",
                    "chosen": "",
                }

            # First, we process input
            if self.fc_mode and len(tools) > 0:
                # print("     FC Mode")
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                )
            else:
                # print("     Non FC Mode")
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False
                )
            
            # Second, we process output (we required it to be model's response)
            model_response = example["chosen"]
            if not model_response.endswith(self.tokenizer.eos_token): model_response += self.tokenizer.eos_token
            
            if not has_reject:
                return {
                    "input": model_input,
                    "chosen": model_response,
                }
            
            # has reject
            model_reject = example["reject"]
            if not model_reject.endswith(self.tokenizer.eos_token): model_reject += self.tokenizer.eos_token

            return {
                    "input": model_input,
                    "chosen": model_response,
                    "reject": model_reject,
                }
        except:
            warnings.warn(
                "Failed to parse the data example. Maybe check your fields!"
            )

            # we do not raise exception, instead, we created the warning then skip this instance so that we dont break training
            return {
                "input": "",
                "chosen": "",
            }
            
    def prepare_sample_text(self, example):
        """
        Given a list of examples OR ONE example, for each example, we prepare the training-ready text. Finally, we return all training-ready text
        """
        if isinstance(example, list):
            output_texts = []
            
            for data_index in range(len(example)): output_texts.append(self._prepare_single_sample_text(example[data_index]))

            return output_texts
    
        else: return self._prepare_single_sample_text(example)

    def chars_token_ratio(self, data, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(zip(range(nb_examples), iter(data)), total=nb_examples):
            text = self.prepare_sample_text(example)
            total_characters += len(text["input"]) + len(text["chosen"])
            if self.tokenizer.is_fast:
                total_tokens += len(self.tokenizer(text["input"] + text["chosen"]).tokens())
            else:
                total_tokens += len(self.tokenizer.tokenize(text["input"] + text["chosen"]))

        return total_characters / total_tokens

    def create_datasets(self, return_type="basic", seed=None):

        raise NotImplementedError


class DPOFoundationModelDataBaseV2(SFTFoundationModelDataBaseV2):
    
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.tokenizer = tokenizer
        self.args = args
        self.name = "DPOFoundationModelDataBaseV2"

    def create_datasets(self, return_type="basic", seed=None):

        raise NotImplementedError

    def prepare_sample_text(self, example):
        if isinstance(example, list):
            output_texts = []
            for i in range(len(example)): output_texts.append(self._prepare_single_sample_text(example[i], has_reject=True))
            return output_texts
        else: return self._prepare_single_sample_text(example[i], has_reject=True)
