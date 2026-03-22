#!/usr/bin/env bash
set -euo pipefail

export LMUData=/data/lishichao/project/era-2026/MLLMerging/LMUData
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY="${OPENAI_API_KEY}"


# MathVista / MathVision scoring requires a working OpenAI-compatible judge.
# export OPENAI_API_KEY=your_key_here

source /home/lishichao/miniconda3/etc/profile.d/conda.sh
conda activate /data/lishichao/env/eval-kit

cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit
MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
# IMPORTANT:
# --model must be a model name registered in vlmeval/config.py, not a filesystem path.
# If you want to evaluate a local model path, first add a corresponding entry in vlmeval/config.py.

#   --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \
# -- reuse
MODEL_NAME="merge_exclude_ocr"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master-port "${MASTER_PORT}" --nproc-per-node=2 run.py \
  --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \
  --model "${MODEL_NAME}" \
  --reuse \
  --verbose \
  --judge gpt-4o-mini
