from tqdm import tqdm
from transformers import LlamaTokenizerFast
from train.fm_utils.xgen_tokenizer import Xgen15BTokenizer


class SFTFoundationModelDataBase:

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.name = "SFTFoundationModelDataBase"

        self.token_try_run = False

        if isinstance(tokenizer, LlamaTokenizerFast):
            # without space but with \n, it will tokenize the word after \n in a different way
            self.response_space = " "
        else:
            self.response_space = ""

        if isinstance(tokenizer, Xgen15BTokenizer):
            self.BOT = ""
            self.EOP = "<|endofprompt|>"
            self.EOT = "<|endofprompt|>"
            self._response_template = f"<|assistant|>\n"
            self._instruction_template = f"<|system|>\n{self.EOT}\n<|user|>\n"
        elif isinstance(tokenizer, LlamaTokenizerFast):
            if "zephyr" in tokenizer.name_or_path:
                self.BOT = ""
                self.EOP = "<|endofprompt|>"
                self.EOT = "<|endofprompt|>"
                self._response_template = f"<|assistant|>\n"
                self._instruction_template = "<|user|>\n"
            elif "Mixtral" in tokenizer.name_or_path:
                self.BOT = tokenizer.bos_token
                self.EOP = "[/INST]"
                self.EOT = tokenizer.eos_token
                # self._response_template = f"<|assistant|>\n"
                self._response_template = self.EOP
                self._instruction_template = "[INST]"

                self.token_try_run = True
            else:
                raise Exception("Unknown tokenizer belonging to LlamaTokenizerFast")
        else:
            raise Exception("Unknown tokenizer")

        self.chars_per_token = 3.0 # a default chars per token

    @property
    def instruction_template(self):
        return self._instruction_template

    @property
    def response_template(self):
        return self._response_template

    def prepare_sample_text(self, example):
        """

        An example sample text would look like this:


        <|system|>
        <|endofprompt|>
        <|user|>
         This is the basic agent object. You can use APIs {'Salesforce_Contact_Create': 'Create a new Salesforce Contact Record. The required fields of the record includes: FirstName, LastName, Title, Email, Phone.', 'Salesforce_Account_Create': 'Create a new Salesforce Account Record. The required fields of the record includes: Name, BillingStreet, BillingCity, BillingState, BillingPostalCode, BillingCountry, Sic, AnnualRevenue, Description (concise, do not use special characters).'}. And you can use self-actions {'Think': 'Conduct thinking and reasoning process for solving task.', 'Finish': 'Complete the task with a response.'}

                    [API_CALL Format]Action:Salesforce_Contact_Create[{'LastName': 'Yao', 'FirstName': 'Weiran', 'Title': 'Research Scientist', 'Email': 'weiran.yao@salesforce.com'}]
        Action:Salesforce_Account_Create[{'Name': 'Example Account', 'BillingStreet': '123 Main Street', 'BillingCity': 'Anytown', 'BillingState': 'CA', 'BillingPostalCode': '12345', 'BillingCountry': 'United States', 'Sic': '1234567890', 'AnnualRevenue': '1000000', 'Description': 'Example Description'}]

        [End of API_CALL Format]

                    [Example]Task: Create a Contact for Caiming Xiong, with phone 6033063311, title VP of Salesforce AI Research, email cxiong@salesforce.com and save it.
        Action:Think[{'INNER': 'I should first format the text description into a parameters dictionary.'}]
        Observation:
        Action:Think[{'INNER': 'I should call Salesforce_Contact_Create[{act_params}].'}]
        Observation:
        Action:Salesforce_Contact_Create[{'LastName': 'Xiong', 'FirstName': 'Caiming', 'Title': 'VP of Salesforce AI Research', 'Phone': '6033063311', 'Email': 'cxiong@salesforce.com'}]
        Observation:Contact Created.
        Action:Finish[{'INNER': 'I just created a contact for Caiming Xiong on Salesforce.'}]
        Observation:Task Completed.

        [End of Example]

        Task: Create contacts for the top three executives at Netflix.
        - Name: Olivier Poitrey, Position Title: Director of Engineering, LinkedIn profile: https://www.linkedin.com/in/olivierpoitrey
        - Name: Stephanie Anderson, Position Title: Workplace Coordinator, LinkedIn profile: https://www.linkedin.com/in/stephaniemichelleanderson
        - Name: David Hyman, Position Title: Chief Legal Officer, LinkedIn profile: https://www.linkedin.com/in/dhymansf

        Action:<|endofprompt|>

        <|assistant|>
         Think[{'INNER': 'I should first format the text description into a parameters dictionary.'}]<|endofprompt|>

        """
        if isinstance(example, list):
            output_texts = []
            for i in range(len(example)):
                # we have text format: '### Prompt: .... \n\n ### Answer: ...'
                # so technically there should have no context-based encoding
                # https://huggingface.co/docs/trl/main/en/sft_trainer :: Using token_ids directly for response_template
                if self.token_try_run:
                    # we do experiments on Mixtral, this is a temporary format, under further unification
                    """
                    According to Mixtral instruction, the default template is like this
                    
                    <s> [INST] Instruction [/INST] Model answer</s>
                    
                    """
                    text = f"{self.BOT} {self.instruction_template} {example[i]['prompt']} {self.EOP}" \
                           f"{self.response_space}{example[i]['chosen']}{self.EOT}"
                else:
                    text = f"{self.BOT} {self.instruction_template} {example[i]['prompt']}{self.EOP}" \
                           f"\n\n{self.response_space}{self.response_template} {example[i]['chosen']}{self.EOT}"
                output_texts.append(text)
            return output_texts

        else:
            # Prepare the text from a sample of the dataset
            if self.token_try_run:
                text = f"{self.BOT} {self.instruction_template} {example['prompt']} {self.EOP}" \
                       f"{self.response_space}{example['chosen']}{self.EOT}"
            else:
                text = f"{self.BOT} {self.instruction_template} {example['prompt']}{self.EOP}" \
                       f"\n\n{self.response_space}{self.response_template} {example['chosen']}{self.EOT}"
            return text

    def chars_token_ratio(self, data, nb_examples=400):
        """
        Estimate the average number of characters per token in the dataset.
        """
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(zip(range(nb_examples), iter(data)), total=nb_examples):
            text = self.prepare_sample_text(example)
            total_characters += len(text)
            if self.tokenizer.is_fast:
                total_tokens += len(self.tokenizer(text).tokens())
            else:
                total_tokens += len(self.tokenizer.tokenize(text))

        return total_characters / total_tokens

    def create_datasets(self, return_type="basic", seed=None):

        raise NotImplementedError


class DPOFoundationModelDataBase(SFTFoundationModelDataBase):
    
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.tokenizer = tokenizer
        self.args = args
        self.name = "DPOFoundationModelDataBase"

    def create_datasets(self, return_type="basic", seed=None):

        raise NotImplementedError

    def prepare_sample_text(self, example):
        return {
                "prompt": [f"{self.instruction_template} {question}"
                           for question in example["prompt"]],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
