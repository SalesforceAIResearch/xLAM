export OPENAI_API_KEY="<YOUR OpenAI API KEY>"  # Replace with your actual OpenAI API key. You can also use any other service API key.

# Default target data directory
TARGET_DATA_DIR="<YOUR PATH>/xLAM/actionstudio/datasets/unified_data"  # Replace with your actual target data directory

# Environment names
ENV_NAMES=('invocation_greeting' 'iqa' 'toolace' 'toolalpaca' 'xlam-fc-60k')  # Replace with your actual environment names

LLM_MODEL="gpt-4o-2024-11-20"
# LLM_MODEL="vLLM Hosted model name"


NUM_TARGET_PROCESSED_TRAJS_PER_ENV="ALL" # Number of target processed trajectories per environment, it can be "ALL" to process all trajs or an integer number
BATCH_SIZE=50 # Batch requests to the vLLM hosted model name and proprietary API name

# Create output directory if it doesn't exist
mkdir -p "nohup_files/${LLM_MODEL}" # Track the progress

# Iterate over each environment and start a background process
for env_name in "${ENV_NAMES[@]}"; do
    nohup python trajectory_critic.py \
        --llm-model "${LLM_MODEL}" \
        --target-data-dir "${TARGET_DATA_DIR}" \
        --target-env-name "${env_name}" \
        --num-target-processed-trajs "${NUM_TARGET_PROCESSED_TRAJS_PER_ENV}" \
        --batch-size "${BATCH_SIZE}" \
        > "nohup_files/${LLM_MODEL}/${env_name}.nohup" 2>&1 &
done
echo "All processes have been started."
