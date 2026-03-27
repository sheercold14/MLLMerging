# LLM Merging: Tucker Subspace Constraint (2026-03-24)

## 1. Core Research Thesis

### The Underdetermined Problem in Optimization-Based Merging

All optimization-based merging methods (wudi2/OptMerge, AdaMerging, etc.) suffer from a fundamental but overlooked problem:

- Each 2D linear layer has **millions of free parameters** (e.g., 4096x14336 = 58.7M for Llama-8B MLP)
- But only **N expert constraints** (typically N=5~8)
- This is a **massively underdetermined** system: millions of unknowns, single-digit equations
- The optimizer finds **one of infinitely many solutions**, determined by initialization/trajectory, not by the problem structure
- Result: **off-subspace artifacts** -- merged vector contains components outside any expert's task-vector subspace, causing unpredictable degradation on specific benchmarks

### Evidence from MLLM experiments (InternVL2.5-1B, 5 experts)

| Model | TextVQA | ChartQA | OCRVQA | GQA | VizWiz | 5-Avg |
|-------|---------|---------|--------|-----|--------|-------|
| Base | 74.23 | 69.72 | 41.76 | 79.93 | 29.01 | 58.93 |
| wudi2 (unconstrained) | 76.01 | **68.44** | 46.29 | 79.35 | 30.88 | 60.19 |
| Tucker-CF k=60 (constrained) | 75.76 | **71.04** | 45.12 | 79.80 | 30.46 | 60.44 |

- wudi2 **degrades** ChartQA by -1.28 vs base (off-subspace artifacts)
- Tucker **improves** ChartQA by +1.32 vs base (subspace constraint prevents artifacts)
- All simple methods (TA, DARE+TA) also degrade ChartQA (~62.5, far below base)

### Our Solution: Tucker Subspace Constraint + Closed-Form

1. **Cross-expert joint basis**: Pool top-k singular vectors from ALL experts -> QR orthogonalize -> shared bases B (m x r2), C (n x r3)
2. **Constrained search**: tau_m = B @ X @ C^T, where X is r2 x r3 (e.g., 200x200 = 40K params vs 4.4M original)
3. **Closed-form solution**: wudi2 loss is quadratic in X -> exact solution X* = G @ H^{-1}
   - H = sum_i Z_i^T Z_i / n_i (r3 x r3)
   - G = sum_i (B^T T_i)(L_i^T Z_i) / n_i (r2 x r3)
   - Z_i = L_i @ C (calibration subspace projection)
   - Tikhonov regularization for rank-deficient H: H_reg = H + (1e-6 * trace(H) / r3) * I

### Key Differentiation from OptMerge (ICLR 2026)

| | OptMerge | Tucker (Ours) |
|---|---|---|
| SVD usage | Per-expert independent SVD -> denoise targets | Cross-expert pooled SVD -> joint basis |
| Search space | tau_m free in full m x n space | tau_m = B @ X @ C^T (constrained) |
| Free params | m x n (~4.4M for 1B MLP, ~58.7M for 8B MLP) | r2 x r3 (~40K), ~100-1000x compression |
| Optimization | Adam/SGD, 300 steps | Closed-form, no iteration |
| Core idea | "Denoise targets, search freely" | "Constrain search space to joint subspace" |
| Cross-expert analysis | None (each expert processed independently) | Yes (pooled basis reflects expert structure) |

OptMerge denoises the **right side** (targets) but leaves the **left side** (tau_m) unconstrained.
We constrain the **left side** -- the search space itself.

## 2. Paper Framing (Target: NeurIPS 2026 or equivalent CCF-A)

### Title candidates
- "Subspace-Constrained Model Merging: Solving the Underdetermined Problem"
- "From Underdetermined to Well-Conditioned: Closed-Form Model Merging via Joint Subspace Constraint"

### Contribution structure
1. **Problem identification**: First to formalize the underdetermined problem in optimization-based merging
2. **Theoretical analysis**: Underdetermined -> poor generalization; subspace constraint -> well-conditioned -> closed-form
3. **Method**: Tucker subspace constraint as a principled instance; architecture-agnostic, data-free
4. **Comprehensive experiments**: LLM (MergeBench, main) + MLLM (OptMerge benchmark, supplementary)

### Ablation story
- Unconstrained (wudi2) vs Constrained (Tucker) at same loss function -> validates constraint
- Closed-form vs Adam at same constraint -> validates closed-form is optimal
- Rank sweep (k=40/60/80) -> optimal subspace dimensionality
- Scaling with #experts and model size -> when does constraint matter most
- Per-layer analysis: which layers are most underdetermined, where does Tucker help most

## 3. Hypothesis to Validate

**Core hypothesis**: Tucker's advantage over wudi2 should **grow** when:
- Model is larger (more params per layer -> more underdetermined)
- More expert domains (more complex interference patterns)
- Tasks are more diverse (coding vs math vs multilingual -> larger off-subspace artifacts)

If this hypothesis holds on MergeBench (8B, 5 domains), the paper is strong.
If Tucker's advantage stays at +0.25 on 8B too, we need deeper theoretical contribution.

## 4. MergeBench Reproduction Task

### Overview
- Paper: [MergeBench (NeurIPS 2025 D&B)](https://arxiv.org/abs/2505.10833)
- Code: https://github.com/uiuctml/MergeBench
- Models: https://huggingface.co/MergeBench
- 8 base models (Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B, Gemma-2-9B, pretrained + instruct)
- 5 domains: instruction following, math, multilingual, coding, safety
- 8 merging baselines: TA, TIES, DARE, Fisher, RegMean, etc.

### Phase 1: Environment Setup & Download (~0.5 day)

```bash
# Clone MergeBench
cd /data/shichao/era-2026/
git clone https://github.com/uiuctml/MergeBench.git
cd MergeBench
pip install -e .  # or follow their setup instructions

# Download smallest model set first for smoke test
# Gemma-2-2B base + 5 domain experts (~6 models x 4GB = ~24GB disk)
# Check HuggingFace: https://huggingface.co/MergeBench
```

### Phase 2: Reproduce Baselines on Gemma-2-2B (~1 day)

Goal: Verify we can reproduce MergeBench's reported numbers for TA, TIES, DARE on Gemma-2-2B.

```bash
# Follow MergeBench eval pipeline
# Run Task Arithmetic, TIES, DARE on Gemma-2-2B
# Compare with Table results in the paper
```

### Phase 3: Implement Tucker-CF for Generic CausalLM (~1 day)

Adapt our Tucker-CF from InternVL-specific to architecture-agnostic:

Key changes needed:
1. **Model loading**: Use `AutoModelForCausalLM` instead of InternVL's `AutoModel`
2. **Layer-by-layer processing**: Cannot load 6 x 8B models to GPU simultaneously
   - Load all state_dicts to CPU memory
   - For each layer: extract task vectors -> move to GPU -> Tucker-CF -> move result back
   - Peak GPU: one layer's task vectors (~560MB for 8B MLP layer x 5 experts)
3. **Parameter naming**: Adapt exclude patterns for Llama/Gemma naming conventions
   - Llama: `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
   - Exclude: `model.embed_tokens`, `model.norm`, `lm_head`, `*bias*`
4. **wudi2 baseline**: Also adapt unconstrained wudi2 for comparison

Core Tucker-CF logic (from model_merging.py) is architecture-agnostic -- operates on (m, n) weight matrices:
```python
def get_tucker_cf_vector(task_vectors, low_ranks, top_k):
    """
    task_vectors: list of N tensors, each (m, n)
    low_ranks: list of N tensors, each (m, n) -- tau_i^T used as data proxy
    top_k: per-expert SVD rank for basis construction
    Returns: merged task vector (m, n)
    """
    # 1. Pool top-k singular vectors from all experts
    # 2. QR orthogonalize -> B (m, r2), C (n, r3)
    # 3. Closed-form: X* = G @ H_reg^{-1}
    # 4. Return B @ X* @ C^T
```

### Phase 4: Run Tucker-CF on Gemma-2-2B (~0.5 day)

Compare Tucker-CF vs wudi2 vs MergeBench baselines on Gemma-2-2B, 5 domains.

### Phase 5: Scale to Llama-3.1-8B (~1-2 days)

If Phase 4 shows promise, run on 8B:
- Download Llama-3.1-8B base + 5 domain experts (~96GB disk)
- Merging: layer-by-layer on 24GB GPU (feasible)
- Inference: 8B float16 = 16GB, fits on 24GB 4090
- Evaluation: follow MergeBench protocol

### Hardware Requirements
- GPU: 1x 24GB 4090 (sufficient for both merging and inference)
- CPU RAM: ~96GB for 8B (holding 6 state_dicts); ~24GB for 2B
- Disk: ~120GB for 8B checkpoints; ~30GB for 2B checkpoints
- Merging time estimate: ~10-30 min for Tucker-CF (layer-by-layer, closed-form)
- Inference time: depends on MergeBench eval suite (likely hours for full 5-domain eval)

### Success Criteria
- Tucker-CF vs wudi2 gap on 8B **> 1.0 avg**: Strong signal, full paper viable
- Tucker-CF vs wudi2 gap on 8B **0.5~1.0 avg**: Moderate, need strong theory/analysis
- Tucker-CF vs wudi2 gap on 8B **< 0.5 avg**: Weak, reconsider direction

## 5. Key References

- OptMerge (ICLR 2026): /data/shichao/Agent/merging/optmerge.pdf -- our group's work, the baseline we improve upon
- MergeBench (NeurIPS 2025): https://arxiv.org/abs/2505.10833 -- standardized LLM merging benchmark
- FusionBench: https://github.com/tanganke/fusion_bench -- toolkit with 19 fusion methods
- DARE (ICML 2024): random pruning + rescaling
- TIES (NeurIPS 2023): trim + sign resolution + merge
- TSV (CVPR 2025): SVD-based interference reduction
- Iso-C (ICML 2025): isotropic merging framework

## 6. Existing Code & Models (MLLM track)

All in /data/shichao/era-2026/MLLMerging/:
- Merging code: `InternVL/internvl_chat/model_merging.py` (all methods including Tucker-CF)
- CLI entry: `InternVL/internvl_chat/model_merging_train.py`
- Evaluation: `VLMEvalKit/` + `Skills/Evaluation/`
- Previous experiment logs: `Skills/claude-experiments/2026-03-23_tucker_cf_experiments.md`

Saved MLLM merged models in /data/shichao/data/InternVL_merged/:
- tucker_cf_k40, tucker_cf_k60, tucker_cf_k80
- tucker_adam_k60, mc_tucker_cf_k60
- dare_ta_s01

## 7. Implementation Notes for New Session (Sonnet)

### Critical: Tikhonov Regularization
K/V projection layers (and GQA heads in Llama) have small m but large n. H matrix can be rank-deficient.
MUST add: `H_reg = H + (1e-6 * trace(H) / r3) * I` before inversion.
Without this, these layers produce catastrophic losses (>100). See 2026-03-23 experiment log.

### Tucker-CF Algorithm Summary
```
For each linear layer:
  1. Compute task vectors: tau_i = theta_i - theta_0  (i = 1..N)
  2. For each expert i:
     - SVD(tau_i) -> U_i, S_i, V_i
     - Keep top-k: U_i[:, :k], V_i[:, :k]
  3. Pool all U columns -> QR -> B (m x r2)
     Pool all V columns -> QR -> C (n x r3)
  4. For each expert i:
     - target_i = tau_i (the task vector)
     - low_rank_i = tau_i^T (used as data proxy, per wudi2 formulation)
     - n_i = ||tau_i||_F^2 (normalization)
  5. Accumulate:
     H = sum_i (Z_i^T @ Z_i) / n_i    where Z_i = low_rank_i @ C
     G = sum_i (B^T @ target_i) @ (low_rank_i^T @ Z_i) / n_i
  6. Regularize: H_reg = H + (1e-6 * trace(H) / r3) * I
  7. Solve: X* = G @ torch.linalg.inv(H_reg)
  8. Result: merged_tau = B @ X* @ C^T
  9. Final: theta_merged = theta_0 + scaling_coefficient * merged_tau
```

### Exclude patterns for Llama/Gemma
```python
LLAMA_EXCLUDE_PATTERNS = [
    ".*embed_tokens.*",
    ".*lm_head.*",
    ".*norm.*",       # RMSNorm layers
    ".*bias.*",
    ".*rotary_emb.*",
]
```

### Layer-by-layer merging skeleton (for 8B on 24GB GPU)
```python
import torch
from safetensors.torch import load_file

def load_state_dict_cpu(model_path):
    """Load model weights to CPU memory only."""
    # Handle both .bin and .safetensors
    ...

def merge_layer_by_layer(base_path, expert_paths, output_path, method="tucker_cf", top_k=60):
    base_sd = load_state_dict_cpu(base_path)
    expert_sds = [load_state_dict_cpu(p) for p in expert_paths]

    merged_sd = {}
    for key in base_sd:
        if should_merge(key):  # 2D linear layer weights only
            task_vectors = [(expert_sds[i][key] - base_sd[key]).float().cuda()
                           for i in range(len(expert_sds))]
            if method == "tucker_cf":
                merged_tv = tucker_cf_single_layer(task_vectors, top_k=top_k)
            elif method == "wudi2":
                merged_tv = wudi2_single_layer(task_vectors)
            merged_sd[key] = (base_sd[key] + merged_tv.cpu().to(base_sd[key].dtype)).clone()
            del task_vectors, merged_tv
            torch.cuda.empty_cache()
        else:
            merged_sd[key] = base_sd[key].clone()  # copy as-is (norms, embeddings, etc.)

    save_model(merged_sd, output_path, base_path)
```

### Reference implementation
The complete working Tucker-CF code is in:
`/data/shichao/era-2026/MLLMerging/InternVL/internvl_chat/model_merging.py`

Look for `tucker_wudi2_cf_merging()` function and inner `get_tucker_cf_vector()`.
The core logic (~60 lines) can be directly extracted and made architecture-agnostic.
