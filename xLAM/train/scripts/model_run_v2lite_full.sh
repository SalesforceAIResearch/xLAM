# export NCCL_DEBUG=INFO

export NCCL_P2P_LEVEL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./

PRECISION="bf16"
SEQ_LEN=8192

BATCH_SIZE=2
GRADIENT_ACCUM_STEPS=3
MAX_STEPS=2000
SAVE_STEPS=1000
LEARNING_RATE=1e-5

SRC_ROOT="xLAM/train/scripts"
MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
DATA_VERSION=""
DATA_SAVE_DIR=""
OUTPUT_ROOT="xLAM/train/scripts"

RUN_NAME="test_run_full"

torchrun --nproc_per_node=8 "${SRC_ROOT}/sft_train_model_v2lite.py" \
    --model_name $MODEL_NAME\
    --run_name $RUN_NAME\
    --ds_stage 3 \
    --ds_config_path "${SRC_ROOT}/config_trainerlite_ds_03_offload.json" \
    --data_save_dir $DATA_SAVE_DIR \
    --weight_precision $PRECISION \
    --seq_length $SEQ_LEN \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --output_dir "${OUTPUT_ROOT}"


