#!/bin/bash

# Set the model path relative to the script location
MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="this picture is "
IMAGE_FILE_PATH="MultimodalLanguageModel/image_0.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

# Convert Windows path to Python format if needed
PYTHON_SCRIPT="${SCRIPT_DIR}/inference.py"

python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU

