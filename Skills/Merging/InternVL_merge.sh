export CUDA_VISIBLE_DEVICES=0
source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate merge
cd /data/shichao/era-2026/MLLMerging/InternVL/internvl_chat

# BASE_MODEL="OpenGVLab/InternVL2_5-1B"
BASE_MODEL="/data/shichao/data/InternVL_merged/merged_exclude_vqa"

MERGE_MODELS=(
    # 'yongxianwei/InternVL2_5-1B_OCR'
    'yongxianwei/InternVL2_5-1B_VQA'
    # 'yongxianwei/InternVL2_5-1B_Geometry'
    # 'yongxianwei/InternVL2_5-1B_Chart'
    # 'yongxianwei/InternVL2_5-1B_Grounding'
    # '/data/shichao/data/InternVL_merged/merged_exclude_vqa'
)
OUTPUT_PATH="/data/shichao/data/InternVL_merged/internvl_wudi2_exclude_add_vqa"
MERGE_METHOD="wudi2"
SCALING_COEFFICIENT="1.0"
TORCH_DTYPE="float16"

python model_merging_train.py \
  --base-model "$BASE_MODEL" \
  --merge-models "${MERGE_MODELS[@]}" \
  --output-path "$OUTPUT_PATH" \
  --merge-method "$MERGE_METHOD" \
  --scaling-coefficient "$SCALING_COEFFICIENT" \
  --torch-dtype "$TORCH_DTYPE"
