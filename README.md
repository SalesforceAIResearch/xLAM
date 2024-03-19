<div align="center">
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main"><img width="350px" height="auto" src="./images/xlam-no-background.png"></a>
</div>

<br/>




<div align="center">

  <!-- [![Release Notes](https://img.shields.io/github/release/SalesforceAIResearch/xLAM)](https://github.com/SalesforceAIResearch/xLAM/releases) -->
  ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
  [![License](https://img.shields.io/badge/License-Apache-green.svg)]()
   [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4)
 [![GitHub star chart](https://img.shields.io/github/stars/SalesforceAIResearch/xLAM?style=social)](https://star-history.com/#SalesforceAIResearch/xLAM)

</div>

## 🎉 News
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
    <br>
<p>

# Framework

## A unified data formatting and streaming loader. 

```
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

```
from fm_utils.derived_data_collator import DataCollatorForPromptAnswer
from fm_trainers.sft_foundation_trainer import SFTFoundationTrainer


collator = DataCollatorForPromptAnswer(
    instruction_template=instruction_template_ids,
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False)

trainer = SFTFoundationTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=False,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
)

trainer.train()
```

# Installation
You can use our configured docker environment `gcr.io/salesforce-research-internal/xlam-2024-02-14`, and one example yaml file is shown at `envs_config`.
Then, you can `pip install -e . --no-dependencies`

Or, you can directly `pip install -e .`. There is a chance that your configured environment might have some error.

# Train

You can refer to the complete example [scripts](https://github.com/SalesforceAIResearch/xLAM/tree/main/xLAM/train/scripts) to learn more details

Or you can simply run this bash script to have a quick start for our example
```bash
nohup accelerate launch --config_file xLAM/train/scripts/multi_gpu.yaml xLAM/train/scripts/sft_mixtral8X7B_accelerator.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --seq_length 4096 --run_name sft_mixtral8X7B_v2_02072024 --output_dir {path} > sft_mixtral8X7B_v2_02072024.nohup 2>&1 &
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



## ToolBench

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
<tr><td>xLAM-v0.1-r</td><td>0.433</td><td>0.677</td></tr>
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

