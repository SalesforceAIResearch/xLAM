### Configuring HuggingFace and Weights & Biases

Make sure you have access to both HuggingFace and Weights & Biases (wandb).

**Login to HuggingFace**

Run the following command and paste your HuggingFace API token when prompted:

```bash
huggingface-cli login
```

**Login to Weights & Biases (wandb)**

Run the following command and paste your wandb API token when prompted:


```bash
wandb login --relogin
```

### Example Training Bash Scripts

Please refer `xLAM/actionstudio/examples/trainings/README.md` for a list of bash training scripts. 

### To run SFT Training

🚀 Please follow [Example Training Bash Scripts and Instructions](../../examples/trainings/README.md) for comprehensive instructions. 

1. Prepare training-ready data format, following instruction under `actionstudio/src/data_pipeline`

2. Activate our conda environment
```
conda activate actionstudio
```

3. Verify data with data verifier (example under `actionstudio/examples/trainings/sft_data_verifier.sh`)

4. Select corresponding training configs: Refer to training examples under `actionstudio/examples/trainings`

5. After training done, export the model using postprocessing code (example under `actionstudio/examples/trainings/postprocessing.sh`)

### To run DPO Training

1. Prepare training-ready data format, following instruction under `actionstudio/src/data_pipeline` with additional field of `reject` for each data instance.

2. Activate our conda environment
```
conda activate actionstudio
```

3. Use `actionstudio/src/foundation_modeling/train/train_dpo.py` to launch the training


