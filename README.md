<div align="center">
  <a href="https://github.com/SalesforceAIResearch/xLAM/tree/main"><img width="300px" height="auto" src="./images/xlam-no-background.png"></a>
</div>

<br/>


<div align="center">

  <!-- [![Release Notes](https://img.shields.io/github/release/SalesforceAIResearch/xLAM)](https://github.com/SalesforceAIResearch/xLAM/releases) -->
  ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
  [![License](https://img.shields.io/badge/License-Apache-green.svg)]()
   [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/Salesforce/xlam-models-65f00e2a0a63bbcd1c2dade4)
 [![GitHub star chart](https://img.shields.io/github/stars/SalesforceAIResearch/xLAM?style=social)](https://star-history.com/#SalesforceAIResearch/xLAM)

</div>

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
