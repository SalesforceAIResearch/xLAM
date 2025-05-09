#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      Infra & access
SRC_ROOT="<YOUR_PATH_HERE>/xLAM/actionstudio"                   # Update your source root here
export PYTHONPATH=./

WANDB_CREDENTIAL_CONFIG="<YOUR_PATH_HERE>"
HF_CREDENTIAL_CONFIG="<YOUR_PATH_HERE>"                   # Huggingface credential config json file
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      model configs
BASE_MODEL_NAME="<YOUR_PATH_HERE>"
FC_MODE=True                                                                        # Function call mode

DS_STAGE=3                                                                                    # Deepspeed stage (2 or 3)
DS_CONFIG_PATH="${SRC_ROOT}/examples/deepspeed_configs/config_ds_0${DS_STAGE}.json"           # Deepspeed config path
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##      Data configs
DATA_SAVE_DIR="<YOUR_PATH_HERE>/xLAM/actionstudio/datasets/train_data"
DATA_MIX_RECIPE="<YOUR_PATH_HERE>/xLAM/actionstudio/examples/data_configs/data_mixture_config.yaml"
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
torchrun --nproc_per_node 1 "${SRC_ROOT}/src/foundation_modeling/train/data_verifier.py" \
    --model_name $BASE_MODEL_NAME \
    --fc_mode $FC_MODE \
    --ds_stage $DS_STAGE \
    --ds_config_path "${DS_CONFIG_PATH}" \
    --data_save_dir $DATA_SAVE_DIR \
    --data_mix_recipe_yaml_config $DATA_MIX_RECIPE \
    --is_data_verfication True \
    --hf_credential_json_config "${HF_CREDENTIAL_CONFIG}" \
    --wandb_credential_json_config "${WANDB_CREDENTIAL_CONFIG}"
