# export NCCL_DEBUG=INFO

export NCCL_P2P_LEVEL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./

PRECISION="bf16"

LORA_R=32
LORA_ALPHA=16
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj,w1,w2,w3,gate"

SEQ_LEN=8192

BATCH_SIZE=2
GRADIENT_ACCUM_STEPS=3
MAX_STEPS=2000
SAVE_STEPS=1000
LEARNING_RATE=1e-5

DS_STAGE=3
DS_CONFIG="config_trainerlite_ds_03.json"

#DS_STAGE=2
#DS_CONFIG="config_trainerlite_ds_02.json"

SRC_ROOT="xLAM/train/scripts"
MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
DATA_VERSION=""
DATA_SAVE_DIR=""
OUTPUT_ROOT="xLAM/train/scripts"

RUN_NAME="test_run_lora"

torchrun --nproc_per_node=8 "${SRC_ROOT}/sft_train_model_v2lite.py" \
    --model_name $MODEL_NAME\
    --run_name $RUN_NAME\
    --ds_stage $DS_STAGE \
    --ds_config_path "${SRC_ROOT}/${DS_CONFIG}" \
    --data_save_dir $DATA_SAVE_DIR \
    --weight_precision $PRECISION \
    --use_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --seq_length $SEQ_LEN \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --output_dir "${OUTPUT_ROOT}"


