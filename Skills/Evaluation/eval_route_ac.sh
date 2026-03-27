#!/usr/bin/env bash
set -euo pipefail

# Evaluate Route A (MC-wudi2) and Route C (Tucker-wudi2) merged models
# Use GPU 1 and 2 (GPU 0 has cali_x_mask.py)

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# Key benchmarks: TextVQA, ChartQA, OCRVQA (same as previous experiments)
BENCHMARKS="TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE"

run_eval() {
    local MODEL=$1
    local GPU=$2
    local PORT
    PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
    echo "[$(date +%H:%M:%S)] Starting eval: ${MODEL} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --master-port "${PORT}" --nproc-per-node=1 run.py \
      --data ${BENCHMARKS} \
      --model "${MODEL}" \
      --verbose \
      --reuse 2>&1 | tee "outputs/${MODEL}_eval.log"
    echo "[$(date +%H:%M:%S)] Finished eval: ${MODEL}"
}

# Run both models in parallel on separate GPUs
run_eval mc_wudi2_b05 1 &
run_eval tucker_wudi2_k40 2 &
wait
echo "=== All Route A/C evaluations complete ==="
