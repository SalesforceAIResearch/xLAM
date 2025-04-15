# Example Training Bash Scripts

This directory contains bash scripts for data verification, model training, and model merging.

## Quick Start

### Data Verification (Optional but Highly Recommended)
Verify training data integrity:

```bash
bash sft_data_verifier.sh
```


#### Single-Node Full Fine-Tuning
Train the model via full fine-tuning with bfloat16 precision on a single node:


```bash
bash sft_bf16_single_pods.sh
```


#### Multi-Node Full Fine-Tuning

Distributed full fine-tuning with bfloat16 precision across two nodes:

**Node 0**


```bash
bash sft_bf16_multi_pods_rank_0.sh
```

**Node 1**

```bash
bash sft_bf16_multi_pods_rank_1.sh
```


#### LORA Fine-Tuning

Fine-tune the model with Low-Rank Adaptation (LoRA), using bfloat16 precision on a single node:


```bash
bash sft_lora_bf16_single_pods.sh
```

#### Quantization with LoRA Fine-Tuning
Fine-tuning combined with 4-bit NormalFloat (NF4) quantization and LoRA on a single node:


```bash
bash sft_lora_nf4_single_pods.sh
```

#### Checkpoints Merge

To merge `final_checkpoint` into a consolidated folder named `final_merged_checkpoint_torch`, run:

```bash
bash postprocessing.sh
```

**Note:** This merging step is automatically included at the end of each training script. You only need to run it manually if you wish to save a merged checkpoint at a specific epoch or step, or if training was skipped or interrupted before completion.
