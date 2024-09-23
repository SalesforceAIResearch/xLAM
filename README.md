<div align="center">
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main"><img width="400px" height="auto" src="./images/xLAM@2400px.png"></a>
</div>

<br/>

<div align="center">

  <!-- [![Release Notes](https://img.shields.io/github/release/SalesforceAIResearch/xLAM)](https://github.com/SalesforceAIResearch/xLAM/releases) -->
  ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
  [![License](https://img.shields.io/badge/License-Apache-green.svg)]()
   [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4)
 [![GitHub star chart](https://img.shields.io/github/stars/SalesforceAIResearch/xLAM?style=social)](https://star-history.com/#SalesforceAIResearch/xLAM)

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2402.15506">Paper</a> |
  <!-- <a href="https://github.com/SalesforceAIResearch/AgentLite/tree/main?tab=readme-ov-file#-key-features">Key Features</a> | -->
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#model-instruction">Model Instruction</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#framework">Framework</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#installation">Installation</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#train">Train</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#benchmarks">Benchmarks</a> |
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main?tab=readme-ov-file#acknowledgement">Acknowledgement</a>
</p>

---

## 🎉 News
- **[09.2024]**: Join our [Discord Community](https://discord.gg/tysWwgZyQ2) if you have any feedbacks!
- **[09.2024]**: Check our xLAM [Technical Report Paper](https://arxiv.org/abs/2409.03215). 
- **[08.2024]**: We are excited to announce the release of full xLAM family, our suite of Large Action Models! From the "tiny giant" to industrial powerhouses.  These models have achieved impressive rankings, placing #1 and #6 on the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard).
Check our [Hugging Face collection](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4).
- **[07.2024]**: We are excited to announce the release of our two function-calling models: [xLAM-1b-fc-r](https://huggingface.co/Salesforce/xLAM-1b-fc-r) and [xLAM-7b-fc-r](https://huggingface.co/Salesforce/xLAM-7b-fc-r). These models have achieved impressive rankings, placing #3 and #25 on the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard), outperforming many significantly larger models. Stay tuned for more powerful models coming soon.
- **[06.2024]** Check our latest work [APIGen](https://apigen-pipeline.github.io/), the best open-sourced models for function calling. Our dataset [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) is currently among the Top-3 trending datasets on HuggingFace, standing out in a field of 173,670 datasets as of July 4, 2024. See also the [Twitter by Salesforce CEO](https://x.com/Benioff/status/1808365628551844186), [VentureBeat](https://venturebeat.com/ai/salesforce-proves-less-is-more-xlam-1b-tiny-giant-beats-bigger-ai-models/) and [新智元](https://mp.weixin.qq.com/s/B3gyaGwzlQaUXyI8n7Rguw).
- **[03.2024]** [xLAM model](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4) is released! Try it together with [AgentLite benchmark](https://github.com/SalesforceAIResearch/AgentLite/tree/main/benchmark) or other benchmarks, which is comparable to GPT-4!
- **[02.2024]** Initial Release of AgentOhana and xLAM [paper](https://arxiv.org/abs/2402.15506)!


This repo is for research purposes only.


Autonomous agents powered by large language models (LLMs) have garnered significant research attention. However, fully harnessing the potential of LLMs for
agent-based tasks presents inherent challenges due to the heterogeneous nature of
diverse data sources featuring multi-turn trajectories. 

This repo introduces xLAM that aggregates agent trajectories from distinct environments, spanning a wide
array of scenarios. It standardizes and unifies these trajectories into
a consistent format, streamlining the creation of a generic data loader optimized
for agent training. Leveraging the data unification, our training pipeline maintains
equilibrium across different data sources and preserves independent randomness
across devices during dataset partitioning and model training. 


<p align="center">
    <br>
    <img src="./images/framework.png" width="700"/>
    <img src="./images/roadmap.png" width="700"/>
    <br>
<p>

# Model Instruction

If you already know [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [xLAM-v0.1](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4) is a significant upgrade and better at many things. 
For the same number of parameters, the model have been fine-tuned across a wide range of agent tasks and scenarios, all while preserving the capabilities of the original model.

xLAM-v0.1-r represents the version 0.1 of the Large Action Model series, with the "-r" indicating it's tagged for research. 
This model is compatible with VLLM and FastChat platforms. 




```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Salesforce/xLAM-v0.1-r")
model = AutoModelForCausalLM.from_pretrained("Salesforce/xLAM-v0.1-r", device_map="auto")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Note:** You may need to tune the Temperature setting  for different applications. Typically, a lower Temperature is helpful for tasks that require deterministic outcomes. 
Additionally, for tasks demanding adherence to specific formats or function calls, explicitly including formatting instructions is advisable and important. 

# Deploying and Interacting with xLAM Models

There are two main options for serving the xLAM model as an OpenAI-compatible chat completion API (here we use `Salesforce/xLAM-8x7b-r` and 4xA100 (40GB) setup as an example):

## Option 1: Using vLLM (Recommended)

vLLM offers efficient serving with lower latency. To serve the model with vLLM:

```bash
vllm serve Salesforce/xLAM-8x7b-r --host 0.0.0.0 --port 8000 --tensor-parallel-size 4
```

## Option 2: Using FastChat

FastChat provides a more feature-rich serving setup. To serve with FastChat:

1. Start the controller:
```bash
python3 -m fastchat.serve.controller --host 0.0.0.0
```

2. Start the OpenAI-compatible API server:
```bash
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000
```

3. Launch the model worker:
```bash
python3 -m fastchat.serve.vllm_worker \
       --model-names "Salesforce/xLAM-8x7b-r" \
       --model-path Salesforce/xLAM-8x7b-r \
       --host 0.0.0.0 \
       --port 31005 \
       --worker-address http://localhost:31001 \
       --num-gpus 4 \
       --limit-worker-concurrency 64
```

## Using the Chat Completion API

Once the model is served, you can use the following xLAM client to interact with it for function calling or other applications:

```python
from xLAM.client import xLAMChatCompletion, xLAMConfig

# Configure the client
config = xLAMConfig(base_url="http://localhost:8000/v1/", model="Salesforce/xLAM-8x7b-r")
llm = xLAMChatCompletion.from_config(config)

# Example conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in New York?"},
    {"role": "assistant", "content": "To get the weather information for New York, I'll need to use the get_weather function.", "tool_calls": {"name": "get_weather", "arguments": '{"location": "New York", "unit": "fahrenheit"}'}},
    {"role": "tool", "name": "get_weather", "content": '{"temperature": 72, "description": "Partly cloudy"}'},
    {"role": "user", "content": "Now, search for the weather in San Francisco."}
]

# Example function definition (optional)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, New York"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search",
        "description": "Search for information on the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query, e.g. 'latest news on AI'"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "respond",
        "description": "When you are ready to respond, use this function. This function allows the assistant to formulate and deliver appropriate replies based on the input message and the context of the conversation. Generate a concise response for simple questions, and a more detailed response for complex questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The content of the message to respond to."}
            },
            "required": ["message"]
        }
    }
]

response = llm.completion(messages, tools=tools)
print(response)
```

# Framework

## A unified data formatting and streaming loader. 

```python
from fm_datasets import webshop_multi_turn_v2
from fm_utils.seed_random import init_device_seed
from fm_utils.interleave_datasets import interleave_data


sft_webshop_multi_turn = webshop_multi_turn_v2.SFTWebShopMultiTurnV2(tokenizer, script_args)

seed = init_device_seed(seed=42)

train_dataset, eval_dataset = \
    interleave_data(
        data_objects=[sft_webshop_multi_turn],
        sample_probs=[1.0],
        return_type="prompt_answer",
        seq_length=4096,
        seed=seed)
```


## Supervised fine tuning and DPO fine tuning. 
We have SFT trainer v1 and v2lite, 
where v1 is more based on `trl` module optimized for LoRA while v2lite is starting from scratch with Accelerator optimized for fully-finetuning.
They share almost the same interface.

```python
from xLAM.fm_utils.derived_data_collator import DataCollatorForPromptAnswer
from xLAM.fm_trainers.sft_foundation_trainer import SFTFoundationTrainer
from xLAM.train.fm_trainers.sft_foundation_trainer_lite import SFTFoundationTrainerLite, prepare_accelerator

script_args = parser.parse_args_into_dataclasses()[0]

collator = DataCollatorForPromptAnswer(
    instruction_template=instruction_template_ids,
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False)

# v2 trainer

accelerator = prepare_accelerator(script_args)
trainer = SFTFoundationTrainerLite(
        args=script_args,
        accelerator=accelerator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )

trainer.train()
```

# Installation
You can use our configured docker environment `gcr.io/salesforce-research-internal/xlam-2024-02-14`, and one example yaml file is shown at `envs_config`.
Then, you can `pip install -e . --no-dependencies`

Or, you can directly `pip install -e .`. There is a chance that your configured environment might have some error.

# Train

You can refer to the complete example [scripts](https://github.com/SalesforceAIResearch/xLAM/tree/main/xLAM/train/scripts) to learn more details

Or you can simply run the bash scripts to have a quick start for our example

for v1: 
```bash
nohup accelerate launch --config_file xLAM/train/scripts/multi_gpu.yaml xLAM/train/scripts/sft_train_model_v1.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --seq_length 4096 --run_name sft_mixtral8X7B_v2_02072024 --output_dir {path} > sft_mixtral8X7B_v2_02072024.nohup 2>&1 &
```

for v2:
```bash
source xLAM/train/scripts/model_run_v2lite_full.sh
```


# Benchmarks

## [BOLAA](https://github.com/salesforce/BOLAA)

### Webshop


<div class="datagrid" style="width:700px;">
<table>
<!-- <thead><tr><th></th><th colspan="6"></th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>ZS</th><th>ZST</th><th>ReaAct</th><th>PlanAct</th><th>PlanReAct</th><th>BOLAA</th></tr></thead>
<tbody>
<tr><td>Llama-2-70B-chat </td><td>0.0089 </td><td>0.0102</td><td>0.4273</td><td>0.2809</td><td>0.3966</td><td>0.4986</td></tr>
<tr><td>Vicuna-33B </td><td>0.1527 </td><td>0.2122</td><td>0.1971</td><td>0.3766</td><td>0.4032</td><td>0.5618</td></tr>
<tr><td>Mixtral-8x7B-Instruct-v0.1 </td><td>0.4634 </td><td>0.4592</td><td><u>0.5638</u></td><td>0.4738</td><td>0.3339</td><td>0.5342</td></tr>
<tr><td>GPT-3.5-Turbo </td><td>0.4851 </td><td><u>0.5058</u></td><td>0.5047</td><td>0.4930</td><td><u>0.5436</u></td><td><u>0.6354</u></td></tr>
<tr><td>GPT-3.5-Turbo-Instruct </td><td>0.3785 </td><td>0.4195</td><td>0.4377</td><td>0.3604</td><td>0.4851</td><td>0.5811</td></tr>
<tr><td>GPT-4-0613</td><td><u>0.5002</u></td><td>0.4783 </td><td>0.4616</td><td><strong>0.7950</strong></td><td>0.4635</td><td>0.6129</td></tr>
<tr><td>xLAM-v0.1-r</td><td><strong>0.5201</strong></td><td><strong>0.5268</strong></td><td><strong>0.6486</strong></td><td><u>0.6573</u></td><td><strong>0.6611</strong></td><td><strong>0.6556</strong></td></tr>
</tbody>
</table>

### HotpotQA

<div class="datagrid" style="width:700px;">
<table>
<!-- <thead><tr><th></th><th colspan="6"></th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>ZS</th><th>ZST</th><th>ReaAct</th><th>PlanAct</th><th>PlanReAct</th></thead>
<tbody>
<tr><td>Mixtral-8x7B-Instruct-v0.1 </td><td>0.3912 </td><td>0.3971</td><td>0.3714</td><td>0.3195</td><td>0.3039</td></tr>
<tr><td>GPT-3.5-Turbo </td><td>0.4196 </td><td>0.3937</td><td>0.3868</td><td>0.4182</td><td>0.3960</td></tr>
<tr><td>GPT-4-0613</td><td><strong>0.5801</strong></td><td><strong>0.5709 </strong></td><td><strong>0.6129</strong></td><td><strong>0.5778</strong></td><td><strong>0.5716</strong></td></tr>
<tr><td>xLAM-v0.1-r</td><td><u>0.5492</u></td><td><u>0.4776</u></td><td><u>0.5020</u></td><td><u>0.5583</u></td><td><u>0.5030</u></td></tr>
</tbody>
</table>

## [AgentLite](https://github.com/SalesforceAIResearch/AgentLite/tree/main)

**Please note:** All prompts provided by AgentLite are considered "unseen prompts" for xLAM-v0.1-r, meaning the model has not been trained with data related to these prompts.

#### Webshop 

<div class="datagrid" style="width:780px;">
<table>
<!-- <thead><tr><th></th><th colspan="2">Easy</th><th colspan="2">Medium</th><th colspan="2">Hard</th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>Act</th><th>ReAct</th><th>BOLAA</th></tr></thead>
<tbody>
<tr><td>GPT-3.5-Turbo-16k </td><td>0.6158 </td><td>0.6005</td><td>0.6652</td></tr>
<tr><td>GPT-4-0613</td><td><strong>0.6989 </strong></td><td><strong>0.6732</strong></td><td><strong>0.7154</strong></td></tr>
<tr><td>xLAM-v0.1-r</td><td><u>0.6563</u></td><td><u>0.6640</u></td><td><u>0.6854</u></td></tr>
</tbody>
</table>

#### HotpotQA

<div class="datagrid" style="width:700px;">
<table>
<thead><tr><th></th><th colspan="2">Easy</th><th colspan="2">Medium</th><th colspan="2">Hard</th></tr></thead>
<thead><tr><th>LLM Name</th><th>F1 Score</th><th>Accuracy</th><th>F1 Score</th><th>Accuracy</th><th>F1 Score</th><th>Accuracy</th></tr></thead>
<tbody>
<tr><td>GPT-3.5-Turbo-16k-0613 </td><td>0.410 </td><td>0.350</td><td>0.330</td><td>0.25</td><td>0.283</td><td>0.20</td></tr>
<tr><td>GPT-4-0613</td><td><strong>0.611</strong></td><td><strong>0.47</strong> </td><td><strong>0.610</strong></td><td><strong>0.480</strong></td><td><strong>0.527</strong></td><td><strong>0.38</strong></td></tr>
<tr><td>xLAM-v0.1-r</td><td><u>0.532</u></td><td><u>0.45</u></td><td><u>0.547</u></td><td><u>0.46</u></td><td><u>0.455</u></td><td><u>0.36</u></td></tr>
</tbody>
</table>



## [ToolBench](https://github.com/OpenBMB/ToolBench)

<div class="datagrid" style="width:780px;">
<table>
<!-- <thead><tr><th></th><th colspan="2">Easy</th><th colspan="2">Medium</th><th colspan="2">Hard</th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>Unseen Insts & Same Set</th><th>Unseen Tools & Seen Cat</th><th>Unseen Tools & Unseen Cat</th></tr></thead>
<tbody>
<tr><td>TooLlama V2 </td><td>0.4385 </td><td>0.4300</td><td>0.4350</td></tr>
<tr><td>GPT-3.5-Turbo-0125 </td><td>0.5000 </td><td>0.5150</td><td>0.4900</td></tr>
<tr><td>GPT-4-0125-preview</td><td><strong>0.5462</strong></td><td><u>0.5450</u></td><td><u>0.5050</u></td></tr>
<tr><td>xLAM-v0.1-r</td><td><u>0.5077</u></td><td><strong>0.5650</strong></td><td><strong>0.5200</strong></td></tr>
</tbody>
</table>

## [MINT-BENCH](https://github.com/xingyaoww/mint-bench)


<div class="datagrid" style="width:780px;">
<table>
<!-- <thead><tr><th></th><th colspan="2">Easy</th><th colspan="2">Medium</th><th colspan="2">Hard</th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>1-step</th><th>2-step</th><th>3-step</th><th>4-step</th><th>5-step</th></tr></thead>
<tbody>
<tr><td>GPT-4-0613</td><td>-</td><td>-</td><td>-</td><td>-</td><td>69.45</td></tr>
<tr><td>Claude-Instant-1</td><td>12.12</td><td>32.25</td><td>39.25</td><td>44.37</td><td>45.90</td></tr>
<tr><td>xLAM-v0.1-r</td><td>4.10</td><td>28.50</td><td>36.01</td><td>42.66</td><td>43.96</td></tr>
<tr><td>Claude-2 </td><td>26.45 </td><td>35.49</td><td>36.01</td><td>39.76</td><td>39.93</td></tr>
<tr><td>Lemur-70b-Chat-v1 </td><td>3.75 </td><td>26.96</td><td>35.67</td><td>37.54</td><td>37.03</td></tr>
<tr><td>GPT-3.5-Turbo-0613 </td><td>2.73</td><td>16.89</td><td>24.06</td><td>31.74</td><td>36.18</td></tr>
<tr><td>AgentLM-70b </td><td>6.48</td><td>17.75</td><td>24.91</td><td>28.16</td><td>28.67</td></tr>
<tr><td>CodeLlama-34b </td><td>0.17</td><td>16.21</td><td>23.04</td><td>25.94</td><td>28.16</td></tr>
<tr><td>Llama-2-70b-chat </td><td>4.27</td><td>14.33</td><td>15.70</td><td>16.55</td><td>17.92</td></tr>
</tbody>
</table>


## [Tool-Query](https://github.com/hkust-nlp/AgentBoard)

<div class="datagrid" style="width:780px;">
<table>
<!-- <thead><tr><th></th><th colspan="2">Easy</th><th colspan="2">Medium</th><th colspan="2">Hard</th></tr></thead> -->
<thead><tr><th>LLM Name</th><th>Success Rate</th><th>Progress Rate</th></tr></thead>
<tbody>
<tr><td>xLAM-v0.1-r</td><td><strong>0.533</strong></td><td><strong>0.766</strong></td></tr>
<tr><td>DeepSeek-67B </td><td>0.400 </td><td>0.714</td></tr>
<tr><td>GPT-3.5-Turbo-0613 </td><td>0.367 </td><td>0.627</td></tr>
<tr><td>GPT-3.5-Turbo-16k </td><td>0.317</td><td>0.591</td></tr>
<tr><td>Lemur-70B </td><td>0.283</td><td>0.720</td></tr>
<tr><td>CodeLlama-13B </td><td>0.250</td><td>0.525</td></tr>
<tr><td>CodeLlama-34B </td><td>0.133</td><td>0.600</td></tr>
<tr><td>Mistral-7B </td><td>0.033</td><td>0.510</td></tr>
<tr><td>Vicuna-13B-16K </td><td>0.033</td><td>0.343</td></tr>
<tr><td>Llama-2-70B </td><td>0.000</td><td>0.483</td></tr>
</tbody>
</table>

# Licenses

This code is licensed under Apache 2.0.  For models based on the [deepseek model](https://huggingface.co/collections/deepseek-ai/deepseek-coder-65f295d7d8a0a29fe39b4ec4), which require you to follow the use based restrictions in the [linked deepseek license](https://github.com/deepseek-ai/deepseek-coder/blob/main/LICENSE-MODEL). This is a research only project.

# Acknowledgement

We want to acknowledge the work which have made contributions to our paper and the agent research community! If you find our work useful, please consider to cite

```bibtex
@article{zhang2024agentohana,
  title={AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning},
  author={Zhang, Jianguo and Lan, Tian and Murthy, Rithesh and Liu, Zhiwei and Yao, Weiran and Tan, Juntao and Hoang, Thai and Yang, Liangwei and Feng, Yihao and Liu, Zuxin and others},
  journal={arXiv preprint arXiv:2402.15506},
  year={2024}
}
```

```bibtex
@article{liu2024apigen,
  title={APIGen: Automated PIpeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author={Liu, Zuxin and Hoang, Thai and Zhang, Jianguo and Zhu, Ming and Lan, Tian and Kokane, Shirley and Tan, Juntao and Yao, Weiran and Liu, Zhiwei and Feng, Yihao and others},
  journal={arXiv preprint arXiv:2406.18518},
  year={2024}
}
```

```bibtex
@article{zhang2024xlamfamilylargeaction,
  title={xLAM: A Family of Large Action Models to Empower AI Agent Systems}, 
  author={Zhang, Jianguo  and Lan, Tian  and Zhu, Ming  and Liu, Zuxin and Hoang, Thai and Kokane, Shirley and Yao, Weiran and Tan, Juntao and Prabhakar, Akshara and Chen, Haolin and Liu, Zhiwei and Feng, Yihao and Awalgaonkar, Tulika and Murthy, Rithesh and Hu, Eric and Chen, Zeyuan and Xu, Ran and Niebles, Juan Carlos and Heinecke, Shelby and Wang, Huan and Savarese, Silvio and Xiong, Caiming},
  journal={arXiv preprint arXiv:2409.03215}
  year={2024}
}
```
