#!/usr/bin/env bash
set -euo pipefail

# Run evaluations on GPU 3 (free)
# Batch 1: etvd_s03 + etvd_s05 (2 models in parallel)
# Batch 2: ta_s02 OCRVQA + ta_s03 ChartQA (failed benchmarks, 2 models in parallel)

export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
export CUDA_VISIBLE_DEVICES=3
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit

cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

BENCHMARKS="TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz"

run_eval() {
    local MODEL=$1
    local DATA=$2
    local PORT
    PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
    echo "[$(date +%H:%M:%S)] Starting eval: ${MODEL} on ${DATA}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --master-port "${PORT}" --nproc-per-node=1 run.py \
      --data ${DATA} \
      --model "${MODEL}" \
      --verbose \
      --reuse \
      --judge gpt-4o-mini 2>&1 | tee "outputs/${MODEL}_gpu3_eval.log"
    echo "[$(date +%H:%M:%S)] Finished eval: ${MODEL}"
}

# Batch 1: ETVD s=0.3 and s=0.5 full evaluation (2 in parallel)
echo "=== Batch 1: ETVD s=0.3 and s=0.5 ==="
run_eval etvd_s03_all5 "${BENCHMARKS}" &
run_eval etvd_s05_all5 "${BENCHMARKS}" &
wait
echo "=== Batch 1 complete ==="

# Batch 2: Re-run failed TA benchmarks
echo "=== Batch 2: Failed TA benchmarks ==="
run_eval ta_s02_all5 "OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz" &
run_eval ta_s03_all5 "ChartQA_TEST GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz" &
wait
echo "=== Batch 2 complete ==="

echo "All GPU 3 evaluations complete!"
