#!/bin/bash
# Train Qwen2-VL-7B OCR LoRA adapter
# Usage:
#   bash scripts/train_ocr.sh                    # full rank=8, all available GPUs
#   bash scripts/train_ocr.sh small               # small rank=4 for testing
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_ocr.sh small  # single GPU test

set -e

cd "$(dirname "$0")/.."

# Select config
CONFIG_NAME="${1:-full}"
if [ "$CONFIG_NAME" = "small" ]; then
    CONFIG="configs/qwen2vl_ocr_lora_sft_small.yaml"
    echo ">>> Using small config (rank=4) for testing"
else
    CONFIG="configs/qwen2vl_ocr_lora_sft.yaml"
    echo ">>> Using full config (rank=8)"
fi

# Step 1: Prepare data if not already done
if [ ! -f "data/llavar_ocr.json" ]; then
    echo ">>> Preparing OCR training data..."
    python scripts/prepare_ocr_data.py
fi

# Step 2: Detect GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo ">>> Detected $NUM_GPUS GPUs"

# Step 3: Train
if [ "$NUM_GPUS" -gt 1 ]; then
    echo ">>> Launching multi-GPU training with $NUM_GPUS GPUs..."
    FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
else
    echo ">>> Launching single-GPU training..."
    llamafactory-cli train "$CONFIG"
fi

echo ">>> Training complete!"
echo ">>> To merge LoRA adapter into base model:"
echo ">>>   llamafactory-cli export configs/qwen2vl_ocr_merge_lora.yaml"
