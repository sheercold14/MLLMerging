#!/usr/bin/env bash
set -euo pipefail

# Quick TextVQA-only evaluation for scaling sweep
# Usage: CUDA_VISIBLE_DEVICES=1 bash eval_sweep_textvqa.sh

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

MODELS=(
    ta_s01_all5
    ta_s02_all5
    ta_s03_all5
    ta_s05_all5
    etvd_s01_all5
    etvd_s02_all5
    etvd_s03_all5
    etvd_s05_all5
)

for MODEL_NAME in "${MODELS[@]}"; do
    # Skip if already evaluated
    RESULT_DIR="outputs/${MODEL_NAME}"
    if find "${RESULT_DIR}" -name "*TextVQA*acc*" 2>/dev/null | grep -q .; then
        echo "Skipping ${MODEL_NAME} (TextVQA already done)"
        continue
    fi

    echo "=========================================="
    echo "Evaluating: ${MODEL_NAME} on TextVQA"
    echo "=========================================="

    MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --master-port "${MASTER_PORT}" --nproc-per-node=1 run.py \
      --data TextVQA_VAL \
      --model "${MODEL_NAME}" \
      --verbose \
      --reuse || echo "WARNING: ${MODEL_NAME} TextVQA evaluation failed"
done

echo "All TextVQA evaluations complete!"
