#!/usr/bin/env bash
set -euo pipefail

# Evaluate scaling coefficient sweep models
# Usage: CUDA_VISIBLE_DEVICES=1 bash eval_sweep.sh <model_name1> <model_name2> ...

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Usage: $0 model1 model2 ..."
    exit 1
fi

for MODEL_NAME in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating: ${MODEL_NAME}"
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
      --data TextVQA_VAL OCRVQA_TESTCORE ChartQA_TEST GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz \
      --model "${MODEL_NAME}" \
      --verbose \
      --reuse \
      --judge gpt-4o-mini || echo "WARNING: ${MODEL_NAME} evaluation failed"
done

echo "All evaluations complete!"
