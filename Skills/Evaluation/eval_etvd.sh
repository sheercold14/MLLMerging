#!/usr/bin/env bash
set -euo pipefail

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export CUDA_VISIBLE_DEVICES=2
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

MODEL_NAME="etvd_merge_all5"
MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"

echo "Evaluating ${MODEL_NAME}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Master port ${MASTER_PORT}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master-port "${MASTER_PORT}" --nproc-per-node=1 run.py \
  --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \
  --model "${MODEL_NAME}" \
  --verbose \
  --reuse \
  --judge gpt-4o-mini
