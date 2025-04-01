# ActionStudio: A Lightweight Framework for Agentic Data and Training of Large Action Models

**Note**: The data and code provided here are intended **solely for research purposes** and should **not be used commercially**.

---

## Installation

### Dependencies

Install dependencies with:

```bash
conda create --name actionstudio python=3.10

bash requirements.sh
```

### Installing ActionStudio

**Development Version** (Latest):

To use the latest code under active development, install ActionStudio in editable mode from the parent actionstudio directory:

```bash
pip install -e .
```

## Structure

```text
actionstudio/
├── datasets/                             # Open-source unified trajectory datasets
├── examples/                             # Usage examples and configurations
│   ├── data_configs/                     # YAML configs for data mixtures
│   ├── deepspeed_configs/                # DeepSpeed training configuration files
│   └── trainings/                        # Bash scripts for various training methods (**`README.md`**)
├── src/                                  # Source code
│   ├── data_conversion/                  # Converting trajectories into training data (**`README.md`**)
│   └── criticLAM/                        # Critic Large Action Model implementation (**`README.md`**)
└── foundation_modeling/                  # Core modeling components
    ├── data_handlers/
    ├── train/
    ├── trainers/
    └── utils/
```

Most top-level folders include a **README.md** with detailed instructions and explanations.

### Data Conversion

See the [Data Conversion Guide](./actionstudio/src/data_pipeline/README.md). 

### Model Training

See the [Model Training Guide](./actionstudio/src/foundation_modeling/README.md).

## Licenses

The code is licensed under Apache 2.0, and the datasets are under the CC-BY-NC-4.0 License. The code and data provided are intended for research purposes only.