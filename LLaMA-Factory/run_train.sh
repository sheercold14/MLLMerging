#!/bin/bash
# Fix CUDA cupti library path
export LD_LIBRARY_PATH=/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cublas/lib:/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890

CONFIG="${1:-configs/qwen2vl_ocr_lora_sft.yaml}"
echo "Using config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

exec /data/lishichao/env/llamaF/bin/llamafactory-cli train "$CONFIG"
