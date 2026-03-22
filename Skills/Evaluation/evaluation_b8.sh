#!/usr/bin/env bash
set -euo pipefail

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="${OPENAI_API_KEY}"


# MathVista / MathVision scoring requires a working OpenAI-compatible judge.
# export OPENAI_API_KEY=your_key_here

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# python - <<'PY'
# import os
# import sys
# import torch

# visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# requested = len([x for x in visible.split(",") if x.strip()]) if visible else 0
# available = torch.cuda.is_available()
# count = torch.cuda.device_count()

# print(f"torch.cuda.is_available={available}")
# print(f"torch.cuda.device_count={count}")

# if not available or count == 0:
#     raise SystemExit(
#         "CUDA is unavailable in the current environment. "
#         "Please verify the host GPU driver/container runtime before launching evaluation."
#     )

# if requested and count < requested:
#     raise SystemExit(
#         f"Visible CUDA device count mismatch: requested {requested} via CUDA_VISIBLE_DEVICES={visible}, "
#         f"but torch only detected {count} device(s)."
#     )
# PY

# IMPORTANT:
# --model must be a model name registered in vlmeval/config.py, not a filesystem path.
# If you want to evaluate a local model path, first add a corresponding entry in vlmeval/config.py.

#   --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \

MODEL_NAME="merge_wudi2_exclude_vqa"
MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Launching torchrun on master port ${MASTER_PORT}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master-port "${MASTER_PORT}" --nproc-per-node=1 run.py \
  --data TextVQA_VAL \
  --model "${MODEL_NAME}" \
  --verbose \
  --reuse \
  --judge gpt-4o-mini
