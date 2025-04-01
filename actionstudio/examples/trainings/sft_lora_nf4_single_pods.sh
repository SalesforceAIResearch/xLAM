#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1. Setup Multi-nodes
NNODES=1                    # Update number of nodes here
GPUS_PER_NODE=8             # Update number of GPUs per node
RANK=0                      # Update node rank (e.g., 0, 1, 2, 3, ...)

export MASTER_ADDR=1        # Update your master address here (e.g., 192.168.1.1)
export MASTER_PORT=9120

if [ "$NNODES" -eq "1" ]
then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --master_port $MASTER_PORT"
elif [ "$NNODES" -gt "1" ]
then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    echo "Invalid number of nodes: $NNODES"
    exit 1
fi
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 2. Setup training
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      Infra & access
export NCCL_P2P_LEVEL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SRC_ROOT="<YOUR_PATH_HERE>/xLAM/actionstudio"                   # Update your source root here
export PYTHONPATH=./

HF_CREDENTIAL_CONFIG="<YOUR_PATH_HERE>"                   # Huggingface credential config json file
WANDB_CREDENTIAL_CONFIG="<YOUR_PATH_HERE>"
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      model configs
BASE_MODEL_NAME="<YOUR_PATH_HERE>"
PRECISION="nf4"
SEQ_LEN=4096
FC_MODE=True                                                                        # Function call mode

USE_LORA=True                                                                      # By default, full training rather than LORA
if [ "$USE_LORA" = True ]; then
    TRAINING_TECHNIQUE="lora_training"
    LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
else
    TRAINING_TECHNIQUE="full_training"
fi

DS_STAGE=2                                                                                    # Deepspeed stage (2 or 3)
DS_CONFIG_PATH="${SRC_ROOT}/examples/deepspeed_configs/config_ds_0${DS_STAGE}.json"           # Deepspeed config path
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      Data configs
DATA_SAVE_DIR="<YOUR_PATH_HERE>/xLAM/actionstudio/datasets/train_data"
DATA_MIX_RECIPE="<YOUR_PATH_HERE>/xLAM/actionstudio/examples/data_configs/data_mixture_config.yaml"
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      Export, logging, and monitoring
WANDB_PROJECT="actionstudio"
WANDB_RUN_NAME="sft_test_run_1_lora_nf4"

OUTPUT_MODEL_ROOT="<YOUR_PATH_HERE>/${WANDB_RUN_NAME}"
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      training configs
NUM_TRAINING_STEPS=40
MODEL_SAVE_EVERY_STEPS=$((NUM_TRAINING_STEPS))                                      # Optional: Save model every x steps

BATCH_SIZE=8
GRADIENT_ACCUM_STEPS=1
LEARNING_RATE=1e-6
SHUFFLE_BUFFER_SIZE=500000
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 3. Execute training code
if [ "$USE_LORA" = True ]; then
    torchrun ${DISTRIBUTED_ARGS} "${SRC_ROOT}/src/foundation_modeling/train/train_sft.py" \
        --model_name "${BASE_MODEL_NAME}" \
        --project_name $WANDB_PROJECT \
        --run_name $WANDB_RUN_NAME \
        --fc_mode $FC_MODE \
        --ds_stage $DS_STAGE \
        --ds_config_path "${DS_CONFIG_PATH}" \
        --data_save_dir $DATA_SAVE_DIR \
        --data_mix_recipe_yaml_config $DATA_MIX_RECIPE \
        --hf_credential_json_config "${HF_CREDENTIAL_CONFIG}" \
        --wandb_credential_json_config "${WANDB_CREDENTIAL_CONFIG}" \
        --weight_precision $PRECISION \
        --use_lora $USE_LORA \
        --lora_target_modules $LORA_TARGET_MODULES \
        --seq_length $SEQ_LEN \
        --learning_rate $LEARNING_RATE \
        --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
        --per_device_train_batch_size $BATCH_SIZE \
        --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE \
        --max_steps $NUM_TRAINING_STEPS \
        --save_steps $MODEL_SAVE_EVERY_STEPS \
        --output_dir "${OUTPUT_MODEL_ROOT}"
else
    torchrun ${DISTRIBUTED_ARGS} "${SRC_ROOT}/src/foundation_modeling/train/train_sft.py" \
        --model_name "${BASE_MODEL_NAME}" \
        --project_name $WANDB_PROJECT \
        --run_name $WANDB_RUN_NAME \
        --fc_mode $FC_MODE \
        --ds_stage $DS_STAGE \
        --ds_config_path "${DS_CONFIG_PATH}" \
        --data_save_dir $DATA_SAVE_DIR \
        --data_mix_recipe_yaml_config $DATA_MIX_RECIPE \
        --hf_credential_json_config "${HF_CREDENTIAL_CONFIG}" \
        --wandb_credential_json_config "${WANDB_CREDENTIAL_CONFIG}" \
        --weight_precision $PRECISION \
        --seq_length $SEQ_LEN \
        --learning_rate $LEARNING_RATE \
        --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
        --per_device_train_batch_size $BATCH_SIZE \
        --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE \
        --max_steps $NUM_TRAINING_STEPS \
        --save_steps $MODEL_SAVE_EVERY_STEPS \
        --output_dir "${OUTPUT_MODEL_ROOT}"
fi
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
