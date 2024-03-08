# xLAM
This repo is for research purpose only.


Autonomous agents powered by large language models (LLMs) have garnered significant research attention. However, fully harnessing the potential of LLMs for
agent-based tasks presents inherent challenges due to the heterogeneous nature of
diverse data sources featuring multi-turn trajectories. This repo introduces xLAM
that aggregates agent trajectories from distinct environments, spanning a wide
array of scenarios. It standardizes and unifies these trajectories into
a consistent format, streamlining the creation of a generic data loader optimized
for agent training. Leveraging the data unification, our training pipeline maintains
equilibrium across different data sources and preserves independent randomness
across devices during dataset partitioning and model training. 

There are several key components 

## fm_datasets
A unified data formatting and streaming loader. 

## fm_trainers
Supervised fine tuning and DPO fine tuning. 

# Installation
You can use our configured docker environment `gcr.io/salesforce-research-internal/xlam-2024-02-14`, and one example yaml file is shown at `envs_config`.
Then, you can `pip install -e . --no-dependencies`

Or, you can directly `pip install -e .`. There is a chance that your configured environment might have some error.

