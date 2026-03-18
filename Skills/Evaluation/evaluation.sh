#!/usr/bin/env bash
set -euo pipefail

export LMUData=/data/lishichao/project/era-2026/MLLMerging/LMUData
export CUDA_VISIBLE_DEVICES=0,1,2
export OPENAI_API_KEY="${OPENAI_API_KEY}"


# MathVista / MathVision scoring requires a working OpenAI-compatible judge.
# export OPENAI_API_KEY=your_key_here

source /home/lishichao/miniconda3/etc/profile.d/conda.sh
conda activate /data/lishichao/env/eval-kit

cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit

# IMPORTANT:
# --model must be a model name registered in vlmeval/config.py, not a filesystem path.
# If you want to evaluate a local model path, first add a corresponding entry in vlmeval/config.py.

#   --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \

MODEL_NAME="Qwen2-VL-7B-Instruct"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc-per-node=3 run.py \
  --data MathVista_MINI MathVision_MINI \
  --model "${MODEL_NAME}" \
  --verbose \
  --reuse \
  --judge gpt-4-turbo
