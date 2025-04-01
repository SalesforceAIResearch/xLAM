export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SRC_ROOT="<YOUR_PATH_HERE>/xLAM/actionstudio"
TRAINED_MODEL_ROOT="<YOUR_ROOT_TO_TRAINED_MODEL>"
BASE_MODEL_ROOT="<YOUR_ROOT_TO_BASE_MODEL>"

cd "${SRC_ROOT}/src/foundation_modeling/train"

python postprocessing.py \
    --trained_model_root $TRAINED_MODEL_ROOT \
    --base_model_root   $BASE_MODEL_ROOT
