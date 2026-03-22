export CUDA_VISIBLE_DEVICES=2
source /home/lishichao/miniconda3/etc/profile.d/conda.sh
conda activate merge-qwen
cd /data/lishichao/project/era-2026/MLLMerging/LLaMA-Factory

BASE_MODEL="/data/lishichao/data/model/Qwen2-VL-7B"
MERGE_MODELS=(
  "/data/lishichao/data/model/Qwen2-VL-7B-Geometry"
  "/data/lishichao/data/model/Qwen2-VL-7B-Chart"
  "/data/lishichao/data/model/Qwen2-VL-7B-OCR"
  "/data/lishichao/data/model/Qwen2-VL-7B-Grounding"
  "/data/lishichao/data/model/Qwen2-VL-7B-VQA"
)
OUTPUT_PATH="/data/lishichao/data/model/Qwen_merged/merged_all_0321"
MERGE_METHOD="wudi2"
SCALING_COEFFICIENT="1.0"

python model_merging_train.py \
  --base-model "$BASE_MODEL" \
  --merge-models "${MERGE_MODELS[@]}" \
  --output-path "$OUTPUT_PATH" \
  --merge-method "$MERGE_METHOD" \
  --scaling-coefficient "$SCALING_COEFFICIENT"
