# Context Transfer: Micro-Capability Model Merging Experiments

> This document captures all necessary context for continuing experiments in a new conversation.
> Date: 2026-03-23

---

## 1. Project Overview

**Goal**: Merge 5 fine-tuned InternVL2.5-1B experts into one model that surpasses the wudi2 baseline. Targeting NeurIPS 2026.

**Base model**: `OpenGVLab/InternVL2_5-1B`
**Experts** (indexed 0-4):
- 0: `yongxianwei/InternVL2_5-1B_OCR`
- 1: `yongxianwei/InternVL2_5-1B_VQA`
- 2: `yongxianwei/InternVL2_5-1B_Geometry`
- 3: `yongxianwei/InternVL2_5-1B_Chart`
- 4: `yongxianwei/InternVL2_5-1B_Grounding`

---

## 2. File Map

### Core Code
| File | Purpose |
|------|---------|
| `InternVL/internvl_chat/model_merging.py` | All merging methods (TaskVector, wudi2, mc_wudi2, anova_wudi2, tucker_wudi2, etc.) |
| `InternVL/internvl_chat/model_merging_train.py` | CLI entry point for running merges |
| `VLMEvalKit/vlmeval/config.py` | Model registry (must register new models here before eval) |
| `Skills/Merging/InternVL_merge.sh` | Shell template for launching merges |
| `Skills/Evaluation/evaluation_b8.sh` | Shell template for launching evaluation |
| `Skills/Evaluation/summary_b8.sh` | Summarize eval results (calls VLMEvalKit/results.py) |
| `Skills/theory/mc_analysis.py` | Micro-capability interference analysis script |

### Results & Logs
| File | Purpose |
|------|---------|
| `Skills/Results/micro_wudi_results.md` | Full methods + results summary |
| `Skills/claude-experiments/2026-03-23_route_ac_experiment.md` | Detailed experiment log |
| `Skills/theory/results/mc_analysis_sample20.json` | Interference analysis data |
| `VLMEvalKit/outputs/` | Raw evaluation outputs |

### Model Outputs
All merged models saved to `/data/shichao/data/InternVL_merged/`:
- `merged_model_all` (wudi2, a.k.a. merge_ours_internvl)
- `mc_wudi2_b05` (Route A)
- `anova_wudi2` (Route B)
- `tucker_wudi2_k40` (Route C)

---

## 3. How to Run a New Experiment (Step-by-Step)

### Step 1: Implement method in model_merging.py
Add a new function (e.g., `mc_tucker_wudi2_merging()`) to `InternVL/internvl_chat/model_merging.py`.

### Step 2: Register in model_merging_train.py
1. Import the new function
2. Add to `MERGE_METHOD_CHOICES` list
3. Add the elif branch in `merge_models()`
4. Add any new CLI args to `build_parser()`

### Step 3: Run the merge
```bash
export CUDA_VISIBLE_DEVICES=<gpu_id>
source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate merge
cd /data/shichao/era-2026/MLLMerging/InternVL/internvl_chat

python model_merging_train.py \
  --base-model "OpenGVLab/InternVL2_5-1B" \
  --merge-models \
    'yongxianwei/InternVL2_5-1B_OCR' \
    'yongxianwei/InternVL2_5-1B_VQA' \
    'yongxianwei/InternVL2_5-1B_Geometry' \
    'yongxianwei/InternVL2_5-1B_Chart' \
    'yongxianwei/InternVL2_5-1B_Grounding' \
  --output-path "/data/shichao/data/InternVL_merged/<model_name>" \
  --merge-method "<method_name>" \
  --scaling-coefficient 0.1 \
  --torch-dtype float16 \
  [--tucker-top-k 40] [--mc-beta 0.5] [--mc-energy-ratio 0.95]
```

### Step 4: Register for evaluation
Add entry in `VLMEvalKit/vlmeval/config.py` under the InternVL section (~line 275):
```python
'<model_name>': partial(InternVLChat, model_path='/data/shichao/data/InternVL_merged/<model_name>', version='V2.0'),
```

### Step 5: Run evaluation
```bash
export CUDA_VISIBLE_DEVICES=<gpu_id>
export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit
cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# Quick eval (3 benchmarks, ~20min):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master-port <random_port> --nproc-per-node=1 run.py \
  --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE \
  --model "<model_name>" --verbose --reuse --judge gpt-4o-mini

# Full eval (7 benchmarks):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master-port <random_port> --nproc-per-node=1 run.py \
  --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \
  --model "<model_name>" --verbose --reuse --judge gpt-4o-mini
```

### Step 6: Summarize results
```bash
cd /data/shichao/era-2026/MLLMerging/VLMEvalKit
python results.py /data/shichao/era-2026/MLLMerging/VLMEvalKit/outputs/<model_name>/<run_folder>
```

---

## 4. Architecture Details

### Parameter Exclusion
These params are NOT merged (kept from base model):
```python
DEFAULT_EXCLUDE_PATTERNS = [
    "vision_model.*",    # vision encoder frozen
    ".*lm_head.*",       # output head frozen
    ".*norm.*",          # layer norms frozen
    ".*embed_tokens.*",  # embeddings frozen
    ".*bias.*",          # biases frozen
]
```

### Layer Shapes in InternVL2.5-1B LLM (Qwen2-0.5B)
The LLM has 24 transformer layers. Per layer, 2D parameter shapes:
- `q_proj`, `o_proj`: 896 x 896
- `k_proj`, `v_proj`: 128 x 896
- `gate_proj`, `up_proj`: 4864 x 896
- `down_proj`: 896 x 4864

Total: 170 2D layers processed by optimization-based methods.

### TaskVector Class (model_merging.py:103-161)
- Computes `finetuned - pretrained` for each included param
- `combine_with_pretrained_model()`: returns `pretrained + scaling_coefficient * task_vector`
- All methods use `scaling_coefficient=0.1` (s=1.0 is catastrophic)

---

## 5. Implemented Methods

### wudi2 (baseline) -- model_merging.py:641-760
Per-2D-layer optimization:
1. For each expert i: SVD of task vector -> keep top-(m/N) singular values for "unique" part
2. Cleaned target: `tau_tilde_i = u2_k @ diag(s2_k) @ v2_k + average_vector`
3. Projection subspace: `L_i = S_masked @ v` (top-(m/N) SVD directions of full task vector)
4. Loss: `sum_i ||(m - tau_tilde_i) @ L_i^T||^2 / ||tau_i||^2`
5. Optimize m in full (m x n) space via Adam, lr=1e-5, 300 steps
6. Init: `m = sum(vectors)`

### Route A: MC-wudi2 -- model_merging.py:761-958
Pre-processing step before wudi2:
1. Individual SVD per expert (95% energy cutoff) -> K micro-capabilities per expert
2. Build K_total x K_total interference matrix: `I_ab = (u_a . u_b)(v_a . v_b) * sigma_a * sigma_b`
3. For negative (conflict) pairs: `suppression = max(0, 1 - beta * conflict/own_energy)`
4. Reconstruct cleaned task vectors with suppressed sigmas
5. Feed to standard wudi2 optimization

Hyperparams: `--mc-beta 0.5 --mc-energy-ratio 0.95`

### Route B: ANOVA-wudi2 -- model_merging.py:960-1173
Two-phase approach:
1. **Global pre-processing**: Compute pairwise affinity across all 170 layers via joint SVD occupancy -> hierarchical clustering -> discover expert groups
2. **Per-layer**: ANOVA decomposition `tau_i = mu + (mu_g(i) - mu) + residual_i` -> merge shared/group deterministically (alpha_0=1.0, alpha_g=0.8) -> only optimize residuals via wudi2

Discovered groups: `[[OCR, VQA], [Geometry, Chart, Grounding]]`

### Route C: Tucker-wudi2 -- model_merging.py:1175-1319
Constrained optimization:
1. Pool top-k left/right SVD vectors from each expert -> QR orthogonalize -> joint bases B (m x r2), C (n x r3)
2. Optimize X (r2 x r3) with same wudi2 loss, but `m = B @ X @ C^T`
3. lr=1e-3 (higher than wudi2's 1e-5, smaller search space)
4. Compression: 109x for MLP layers (4864x896 -> 200x200), 20x for 896x896, 4x for 128x896

Hyperparams: `--tucker-top-k 40`

---

## 6. Results

### Primary Comparison (3 benchmarks)

| Model | TextVQA | ChartQA | OCRVQA | Avg |
|-------|---------|---------|--------|-----|
| InternVL2_5-1B (base) | 74.23 | 69.72 | 41.76 | 61.90 |
| wudi2 (merge_ours_internvl) | 76.01 | 68.44 | 46.29 | 63.58 |
| MC-wudi2 beta=0.5 (Route A) | 75.97 | 69.00 | 46.19 | 63.72 |
| ANOVA-wudi2 (Route B) | 75.54 | 70.12 | 44.73 | 63.46 |
| **Tucker-wudi2 k=40 (Route C)** | 75.53 | **70.68** | 45.21 | **63.81** |

### Other Methods (from earlier experiments)
| Model | TextVQA | ChartQA | OCRVQA |
|-------|---------|---------|--------|
| TA s=0.1 | 76.13 | 62.40 | 44.82 |
| ETVD s=0.1 | 74.06 | 64.36 | - |
| TA-norm | 70.08 | - | - |

### wudi2 Exclude-One (ChartQA analysis)
| Config | ChartQA |
|--------|---------|
| All 5 experts | 68.44 |
| -Chart | 62.96 |
| -Geometry | 70.32 |
| -OCR | 70.60 |
| -VQA | 70.68 |
| -Grounding | 69.12 |

Key finding: Removing OCR or VQA improves ChartQA most -- these experts destructively interfere with Chart.

---

## 7. Key Findings & Insights

### Why Tucker Works
- wudi2 optimizes in full m x n space (up to 4.4M free params for MLP layers) with only 5 constraint equations -> massive underdetermined system
- The optimizer can find solutions with off-subspace components that destructively interfere
- Tucker constrains to 200x200 = 40K params, all within expert task vector span -> prevents off-subspace artifacts
- Particularly benefits ChartQA where Chart-Geometry interference is confined to their shared subspace

### Micro-Capability Interference
- 48.7% of cross-expert micro-cap pairs are conflicts (negative interference)
- OCR-VQA has highest conflict magnitude (6.15), nearly 2x any other pair
- VQA is most conflict-prone expert overall
- MLP layers dominate conflict (top 8 most conflicting layers are all MLP)
- Geometry is most isolated expert

### Expert Grouping (from ANOVA affinity analysis)
```
       OCR    VQA    Geo    Chart  Ground
OCR    0.000  0.029  0.006  0.006  0.018
VQA    0.029  0.000  0.000  0.000  0.000
Geo    0.006  0.000  0.000  0.141  0.018
Chart  0.006  0.000  0.141  0.000  0.024
Ground 0.018  0.000  0.018  0.024  0.000
```
Groups: [[OCR,VQA], [Geometry,Chart,Grounding]]
- OCR-VQA: text understanding
- Geometry-Chart: visual/spatial reasoning (highest affinity 0.141)

### Tucker Residual Loss Pattern
- K/V proj layers: converge to near-zero loss (< 1e-6) -- Tucker basis captures everything
- MLP layers: non-trivial residual loss (~0.04-0.07) -- Tucker basis misses some structure at k=40
- This means increasing k should help, especially for MLP layers

---

## 8. Planned Next Experiments (Priority Order)

### P1: Tucker k=60 and k=80
**Rationale**: MLP layers show residual loss 0.04-0.07 at k=40 (r=200), meaning the Tucker basis misses structure. Higher rank should recover OCRVQA without losing ChartQA gains.

**Implementation**: Just change `--tucker-top-k 60` (or 80). No code changes needed.

```bash
# k=60 -> r=300
python model_merging_train.py \
  --base-model "OpenGVLab/InternVL2_5-1B" \
  --merge-models ... \
  --output-path "/data/shichao/data/InternVL_merged/tucker_wudi2_k60" \
  --merge-method tucker_wudi2 --scaling-coefficient 0.1 --tucker-top-k 60
```

### P2: MC-Tucker-wudi2 (Route A+C combination)
**Rationale**: MC cleans micro-capability conflicts in input; Tucker constrains search space. Two-level attack on interference.

**Implementation**: New function `mc_tucker_wudi2_merging()` that:
1. First applies MC conflict suppression (from mc_wudi2)
2. Then builds Tucker basis on the cleaned task vectors
3. Then optimizes in Tucker subspace

### P3: Adaptive Tucker Rank
**Rationale**: K/V layers are saturated at k=40, MLP layers want more. Per-layer adaptive rank based on cumulative energy retention (e.g., 99%).

**Implementation**: Modify `get_tucker_merging_vector()` to accept per-layer k, or compute k dynamically from SVD energy.

### P4: ANOVA-Tucker (Route B+C)
**Rationale**: ANOVA separates shared/group/residual. Only residuals need optimization. Tucker subspace on residuals only.

**Implementation**: New function that does ANOVA decomposition first, then Tucker-constrained wudi2 on residual components.

### P5: Full 7-benchmark evaluation
Run all winning methods on: TextVQA_VAL, OCRVQA_TESTCORE, VizWiz, GQA_TestDev_Balanced, ChartQA_TEST, MathVision_MINI, MathVista_MINI.

### P6: Per-expert scaling in Tucker space
Add learnable weights w_i to the loss function, or use micro-cap conflict scores for initial weights.

---

## 9. GPU Status

- **GPU 0**: Running `cali_x_mask.py` -- DO NOT use
- **GPU 1, 2, 3**: Available for experiments

---

## 10. Environment Notes

### Conda Environments
- `merge`: for running merges (`model_merging_train.py`)
- `eval-kit`: for running evaluations (`VLMEvalKit`)

### Working Directory
`/data/shichao/era-2026/MLLMerging/InternVL/internvl_chat`

### Model Registry Location
`VLMEvalKit/vlmeval/config.py` -- existing entries around line 275:
```python
'merge_ours_internvl': partial(InternVLChat, model_path='/data/shichao/data/InternVL_merged/merged_model_all', version='V2.0'),
'mc_wudi2_b05': partial(InternVLChat, model_path='/data/shichao/data/InternVL_merged/mc_wudi2_b05', version='V2.0'),
'tucker_wudi2_k40': partial(InternVLChat, model_path='/data/shichao/data/InternVL_merged/tucker_wudi2_k40', version='V2.0'),
'anova_wudi2': partial(InternVLChat, model_path='/data/shichao/data/InternVL_merged/anova_wudi2', version='V2.0'),
```

### Key Constants
- `scaling_coefficient = 0.1` (ALWAYS, s=1.0 is catastrophic)
- wudi2 optimization: Adam lr=1e-5, 300 steps, init=sum(vectors)
- Tucker optimization: Adam lr=1e-3, 300 steps, init=mean of projected targets
- MC suppression: beta=0.5, energy_ratio=0.95

---

## 11. Code Structure Quick Reference

### model_merging.py function map
```
Line 89:   get_param_names_to_merge()
Line 103:  class TaskVector
Line 162:  ties_merging()
Line 277:  copy_params_to_model()
Line 288:  mask_input_with_mask_rate()
Line 318:  mask_model_weights()
Line 355:  task_arithmetic()
Line 379:  svd_merging()
Line 485:  iso_merging()
Line 546:  wudi_merging()
Line 641:  wudi_merging2()          <- baseline
Line 761:  mc_wudi2_merging()       <- Route A
Line 960:  anova_wudi2_merging()    <- Route B
Line 1175: tucker_wudi2_merging()   <- Route C
Line 1321: merge_models()           <- legacy entry point (not used by CLI)
```

### model_merging_train.py structure
```
Line 22:  DEFAULT_EXCLUDE_PATTERNS
Line 30:  MERGE_METHOD_CHOICES list
Line 56:  load_model()
Line 65:  merge_models() -- main dispatch
Line 164: build_parser()
Line 195: main()
```

### Common pattern for all optimization methods
All methods follow the same structure:
1. Extract TaskVectors for each expert
2. For each 2D param (170 total): stack vectors -> run method -> store result
3. For non-2D params: simple average
4. Create TaskVector from merged dict -> combine_with_pretrained_model(scaling=0.1)
