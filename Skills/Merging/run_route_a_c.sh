#!/usr/bin/env bash
set -euo pipefail

# Run Route A (MC-wudi2) and Route C (Tucker-wudi2) merging experiments
# GPU 1: Route A (mc_wudi2)
# GPU 2: Route C (tucker_wudi2)

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/InternVL/internvl_chat

BASE_MODEL="OpenGVLab/InternVL2_5-1B"
EXPERTS=(
    "yongxianwei/InternVL2_5-1B_OCR"
    "yongxianwei/InternVL2_5-1B_VQA"
    "yongxianwei/InternVL2_5-1B_Geometry"
    "yongxianwei/InternVL2_5-1B_Chart"
    "yongxianwei/InternVL2_5-1B_Grounding"
)
OUTPUT_DIR="/data/shichao/data/InternVL_merged"

run_merge() {
    local METHOD=$1
    local NAME=$2
    local GPU=$3
    shift 3  # remaining args are extra flags

    echo "=== [$(date)] Starting merge: ${NAME} on GPU ${GPU} ==="

    CUDA_VISIBLE_DEVICES=${GPU} python model_merging_train.py \
        --base-model "${BASE_MODEL}" \
        --merge-models "${EXPERTS[@]}" \
        --output-path "${OUTPUT_DIR}/${NAME}" \
        --merge-method "${METHOD}" \
        --scaling-coefficient 0.1 \
        --torch-dtype float16 \
        "$@" \
        2>&1 | tee "${OUTPUT_DIR}/${NAME}_merge.log"

    echo "=== [$(date)] Finished merge: ${NAME} ==="
}

# Route A: MC-wudi2 with different beta values
run_merge mc_wudi2 mc_wudi2_b05 1 --mc-beta 0.5 &
PID_A=$!

# Route C: Tucker-wudi2 with k=40
run_merge tucker_wudi2 tucker_wudi2_k40 2 --tucker-top-k 40 &
PID_C=$!

echo "Merging started. PIDs: Route A=$PID_A, Route C=$PID_C"
echo "Monitor with: tail -f ${OUTPUT_DIR}/mc_wudi2_b05_merge.log"
echo "              tail -f ${OUTPUT_DIR}/tucker_wudi2_k40_merge.log"
wait
echo "=== All merging complete ==="
