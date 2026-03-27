#!/usr/bin/env bash
set -euo pipefail

# Evaluate new merged models on GPU 3 alongside existing evals
export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export CUDA_VISIBLE_DEVICES=3
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# Focus on key benchmarks first: TextVQA + ChartQA + OCRVQA
BENCHMARKS="TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE"

run_eval() {
    local MODEL=$1
    local PORT
    PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
    echo "[$(date +%H:%M:%S)] Starting eval: ${MODEL}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --master-port "${PORT}" --nproc-per-node=1 run.py \
      --data ${BENCHMARKS} \
      --model "${MODEL}" \
      --verbose \
      --reuse 2>&1 | tee "outputs/${MODEL}_eval.log"
    echo "[$(date +%H:%M:%S)] Finished eval: ${MODEL}"
}

# Run 2 key models in parallel
run_eval ta_no_geometry_s01 &
run_eval etvd_v2_s01_all5 &
wait
echo "=== Phase 1 complete ==="

# Then run remaining models
run_eval ta_weighted_v1 &
run_eval ta_no_geometry_s015 &
wait
echo "=== Phase 2 complete ==="

echo "All new model evaluations complete!"
