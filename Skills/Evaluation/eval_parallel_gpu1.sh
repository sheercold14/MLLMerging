#!/usr/bin/env bash
set -euo pipefail

# Run 4 evaluations in parallel on GPU 1
# Each uses ~3GB, 4 instances ≈ 12GB (out of 24GB available)

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export CUDA_VISIBLE_DEVICES=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# Benchmarks: TextVQA (5000), OCRVQA (3072), ChartQA (2.5k), GQA (12k)
# Start with TextVQA + ChartQA for each model (diverse, reasonable size)
BENCHMARKS="TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz"

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
      --reuse \
      --judge gpt-4o-mini 2>&1 | tee "outputs/${MODEL}_eval.log"
    echo "[$(date +%H:%M:%S)] Finished eval: ${MODEL}"
}

# Run 4 TA scaling variants in parallel
echo "=== Batch 1: TA scaling sweep ==="
run_eval ta_s01_all5 &
run_eval ta_s02_all5 &
run_eval ta_s03_all5 &
run_eval ta_s05_all5 &
wait
echo "=== Batch 1 complete ==="

# Run 4 ETVD scaling variants in parallel
echo "=== Batch 2: ETVD scaling sweep ==="
run_eval etvd_s01_all5 &
run_eval etvd_s02_all5 &
run_eval etvd_s03_all5 &
run_eval etvd_s05_all5 &
wait
echo "=== Batch 2 complete ==="

echo "All evaluations complete!"
