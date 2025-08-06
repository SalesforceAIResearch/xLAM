# 🧠 [ActionStudio](https://arxiv.org/pdf/2503.22673): A Lightweight Framework for Agentic Data and Training of Large Action Models

---
> **Note**: Any data related to xLAM is **partially released due to internal regulations** to support the advancement of the agent research community.

---

## 📦  Installation

### 🔧 Dependencies

Install dependencies from the root `xLAM` directory (where `setup.py` is located) with:

```bash
conda create --name actionstudio python=3.10

bash requirements.sh
```

### 🚀 Installing ActionStudio

**Development Version** (Latest):

To use the latest code under active development, install ActionStudio in **editable mode** from the root `xLAM` directory (where `setup.py` is located):

```bash
pip install -e .
```

## 🗂️ Structure

```text
actionstudio/
├── datasets/                             # Open-source **`unified trajectory datasets`**
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

🔍 Most top-level folders include a **README.md** with detailed instructions and explanations.

❤️ Please follow [Example Training Bash Scripts and Instructions](https://github.com/SalesforceAIResearch/xLAM/blob/main/actionstudio/examples/trainings/README.md) for comprehensive instructions. 

## ⚡ Efficiency
<img width="1705" alt="image" src="https://github.com/user-attachments/assets/7885ba5f-2155-431b-941f-0cbfc6641432" />


## 📜 Licenses

The code is licensed under Apache 2.0, and the datasets are under the CC-BY-NC-4.0 License. The data provided are intended for research purposes only.

## 🛠️ Code Updates History

#### 💫 **Aug 05, 2025**
- **Unified config tracking**
Every run now writes its full training configuration to a single JSON file—keyed by a unique model ID—in [model_config_files](./examples/trainings/model_config_files/) for easy reference and reproducibility.

- **HF ⇄ DeepSpeed parity**
Resolved inconsistencies between Hugging Face and DeepSpeed hyper-parameter settings to ensure they stay perfectly in sync.

- **Learning-rate scheduler tuning**
Refined default scheduler parameters for smoother warm-up and steadier convergence.

- **General code cleanup**
Streamlined modules, removed dead paths, and added inline docs for easier maintenance.


#### **May 09, 2025**
- Fixed argument error in data_verifier. Ref to [#24](https://github.com/SalesforceAIResearch/xLAM/issues/24).

#### **April 14, 2025**
- Updated dependency versions to support the latest models and techniques
- Added auto calculation and assignment of training steps
- Enabled automatic checkpoint merging at the end of training. 
    - 📄 See [actionstudio/examples/trainings/README.md](actionstudio/examples/trainings/README.md) for training examples and usage
- Improved documentation and inline code comments

