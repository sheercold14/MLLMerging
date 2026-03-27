# Micro-Capability Enhanced Model Merging: Methods & Experimental Results

## 1. Problem Statement

Given 5 fine-tuned InternVL2.5-1B experts (OCR, VQA, Geometry, Chart, Grounding), we aim to merge them into a single model that preserves or improves upon each expert's capability. The baseline method **wudi2** achieves the best known average but suffers from **ChartQA degradation** (-1.28 vs base). Our goal: surpass wudi2's average while fixing ChartQA.

## 2. Baseline: wudi2

Per-2D-layer optimization:
1. For each expert i: separate shared (avg across experts) from unique via SVD
2. Construct cleaned target τ̃_i and projection subspace L_i
3. Optimize merged vector m via Adam (300 steps, lr=1e-5) in full m×n space
4. Loss = Σ_i ||(m - τ̃_i) @ L_i^T||² / ||τ_i||²

Key limitation: optimizes in **full parameter space** (up to 4864×896 = 4.4M dims per layer), which allows the optimizer to find solutions that destructively interfere across experts.

## 3. Improvement Routes

### 3.1 Route A: MC-wudi2 (Micro-Capability Conflict Pre-Resolution)

**Idea**: Before wudi2 optimization, identify and suppress conflicting micro-capabilities across experts.

**Algorithm**:
1. For each 2D layer, extract micro-capabilities via individual SVD per expert (95% energy cutoff)
   - K ranges from 107 to 3426 per layer (avg ~1800)
   - MLP gate/up_proj have highest K (~3000+), attention K/V have lowest (~400-500)
2. Build K_total × K_total interference matrix: I_ab = (u_a · u_b)(v_a · v_b) · σ_a · σ_b
3. For each micro-cap, compute conflict_score = sum of negative cross-expert interference
4. Suppression factor = max(0, 1 - β · relative_conflict)
5. Reconstruct cleaned task vectors with suppressed sigmas
6. Feed cleaned task vectors to standard wudi2 optimization

**Hyperparameters**: β=0.5, energy_ratio=0.95, scaling_coefficient=0.1

**Observations**:
- 100% of micro-caps receive some suppression (all have cross-expert conflict)
- Average suppression factor: 0.62-0.81 (attention layers suppressed more, MLP less)
- No micro-caps completely zeroed out (except 1-3 in layers 22-23)

### 3.2 Route B: ANOVA-wudi2 (Hierarchical Expert Decomposition)

**Idea**: Discover expert groupings via ETVD affinity analysis, then decompose task vectors into shared/group/residual components. Only residuals need optimization.

**Algorithm**:
1. Compute affinity matrix across all 170 2D layers via joint SVD occupancy analysis
2. Hierarchical clustering → discover expert groups
3. ANOVA decomposition per layer: τ_i = μ (grand mean) + (μ_g(i) - μ) (group effect) + residual_i
4. Shared (μ) merged at α_0=1.0, group components at α_g=0.8
5. Only residuals go through wudi2 optimization

**Discovered groups**: [[OCR, VQA], [Geometry, Chart, Grounding]]
- OCR-VQA: share text understanding capabilities
- Geometry-Chart: highest pairwise affinity (0.141)
- VQA is most isolated from non-OCR experts (affinity=0.0 with Geometry, Chart, Grounding)

**Affinity Matrix** (OCR=0, VQA=1, Geo=2, Chart=3, Ground=4):
```
       OCR    VQA    Geo    Chart  Ground
OCR    0.000  0.029  0.006  0.006  0.018
VQA    0.029  0.000  0.000  0.000  0.000
Geo    0.006  0.000  0.000  0.141  0.018
Chart  0.006  0.000  0.141  0.000  0.024
Ground 0.018  0.000  0.018  0.024  0.000
```

**Energy decomposition** (sample layer): μ_energy=0.50, group_energy=0.47, residual_energy=0.32

### 3.3 Route C: Tucker-wudi2 (Subspace-Constrained Optimization)

**Idea**: Constrain wudi2's search space to a low-rank Tucker subspace built from expert SVD bases.

**Algorithm**:
1. For each 2D layer, compute individual SVD of each expert's task vector
2. Pool top-k left/right singular vectors from all experts → QR orthogonalize → joint bases B (m×r₂), C (n×r₃)
3. Optimize X (r₂×r₃) instead of m (m×n), with same wudi2 loss
4. Reconstruct: m = B @ X* @ C^T

**Hyperparameters**: top_k=40, lr=1e-3 (higher than wudi2's 1e-5, appropriate for smaller search space)

**Observations**:
- r₂=r₃=200 for layers ≥896×896 (5×40=200 pooled vectors, all linearly independent)
- r₂=128 for 128×896 layers (K/V proj, output dim is bottleneck)
- Compression ratio: **109x** for MLP layers (4864×896 → 200×200), 20x for 896×896, 4x for 128×896
- K/V proj layers: near-zero loss (loss < 1e-6) — Tucker basis captures everything
- MLP layers: non-trivial residual loss (~0.04-0.07) — Tucker basis may miss some structure

## 4. Micro-Capability Interference Analysis

Analysis of 20 sampled layers (36,548 total micro-capabilities):

| Metric | Value |
|--------|-------|
| Total micro-caps analyzed | 36,548 |
| Cross-expert conflict rate | 48.7% of pairs |
| Highest conflict pair | OCR-VQA (magnitude 6.15) |
| Most conflict-prone expert | VQA |
| Most isolated expert | Geometry |

**Conflict distribution by layer type**:
- MLP layers dominate conflict (top 8 most conflicting layers are all MLP)
- Layer 3 up_proj has the highest conflict magnitude
- Attention Q/O layers: moderate conflict
- Attention K/V layers: lowest conflict

## 5. Results

| Model | TextVQA | ChartQA | OCRVQA | Avg |
|-------|---------|---------|--------|-----|
| InternVL2_5-1B (base) | 74.23 | 69.72 | 41.76 | 61.90 |
| wudi2 | 76.01 | 68.44 | 46.29 | 63.58 |
| MC-wudi2 β=0.5 (Route A) | 75.97 | 69.00 | 46.19 | 63.72 |
| ANOVA-wudi2 (Route B) | 75.54 | **70.12** | 44.73 | 63.46 |
| Tucker-wudi2 k=40 (Route C) | 75.53 | **70.68** | 45.21 | **63.81** |

### Key Findings

1. **Tucker-wudi2 achieves best average (63.81)**, beating wudi2 (63.58) by +0.23
2. **ChartQA is the discriminating benchmark**:
   - wudi2 **loses** 1.28 on ChartQA vs base (68.44 vs 69.72)
   - Tucker-wudi2 **gains** 0.96 on ChartQA vs base (70.68 vs 69.72)
   - ANOVA-wudi2 **gains** 0.40 on ChartQA vs base (70.12 vs 69.72)
   - MC-wudi2 nearly recovers (69.00 vs 69.72, only -0.72)
3. **TextVQA-OCRVQA tradeoff**: Methods that improve ChartQA tend to lose on TextVQA and OCRVQA
   - wudi2: best TextVQA (76.01) and OCRVQA (46.29)
   - Tucker-wudi2: lower TextVQA (75.53, -0.48) and OCRVQA (45.21, -1.08)
4. **Subspace constraint is protective**: Tucker's low-rank constraint prevents the optimizer from finding destructive interference solutions
5. **ANOVA grouping is meaningful**: The discovered groups [[OCR,VQA], [Geo,Chart,Ground]] align with task semantics and interference patterns

## 6. Analysis: Why Tucker Fixes ChartQA

wudi2 optimizes in full m×n space. For MLP layers (4864×896), this is 4.4M free parameters with only 5 constraint equations (one per expert). The optimizer can exploit high-dimensional degrees of freedom to minimize the loss while creating off-subspace components that destructively interfere — particularly affecting Chart expert which has highest cross-expert similarity with Geometry.

Tucker constrains search to 200×200 = 40K parameters (109x fewer), all within the span of expert task vectors. This:
- Prevents off-subspace artifacts
- Forces the optimizer to find solutions that genuinely balance expert contributions
- Particularly benefits ChartQA because the Chart-Geometry interference is confined to their shared subspace

## 7. Insights for Next Experiments

### 7.1 Most Promising Direction: Tucker with Higher Rank

Tucker-wudi2 with k=40 already beats wudi2 despite MLP layers showing non-trivial residual loss (0.04-0.07). This means the Tucker basis at r=200 is **missing some structure**. Increasing top_k to 60 or 80 (r=300 or 400) could capture more expert-specific directions while still maintaining the protective subspace constraint.

**Prediction**: Tucker k=60 should improve OCRVQA (currently -1.08 vs wudi2) without losing ChartQA gains.

### 7.2 Combine Route A + C: MC-Tucker-wudi2

MC-wudi2 achieves best MC-related improvement on ChartQA among pre-processing methods (69.00 vs wudi2's 68.44). Tucker achieves best overall ChartQA (70.68). Combining them:
1. First apply MC conflict suppression (clean up micro-capability interference)
2. Then optimize in Tucker subspace (prevent new interference from forming)

This attacks the problem at two levels: input cleaning + search space constraint.

### 7.3 Adaptive Tucker Rank

Current Tucker uses fixed k=40 for all layers. But:
- K/V layers already converge to near-zero loss at k=40 (r=128) — they don't need more
- MLP layers have residual loss 0.04-0.07 — they could benefit from higher rank
- Attention Q/O layers are intermediate

**Proposal**: Set per-layer Tucker rank adaptively based on expert task vector complexity:
- Use cumulative energy from pooled singular values to determine rank
- Target 99% energy retention per layer
- This could give k~20 for K/V layers and k~80 for MLP layers

### 7.4 ANOVA + Tucker Hybrid

ANOVA decomposes task vectors into μ + group + residual. The shared (μ) and group components are deterministic — they don't need optimization. Only residuals need optimization, and these residuals live in a **smaller subspace** than full task vectors.

**Proposal**:
1. ANOVA decomposition to separate shared/group/residual
2. Tucker subspace construction only on residual components
3. Optimize in Tucker-constrained residual space

This reduces the optimization problem from "find m that balances 5 experts" to "find residual corrections within constrained subspace".

### 7.5 Per-Expert Scaling in Tucker Space

Currently all methods use uniform scaling_coefficient=0.1. But interference analysis shows:
- OCR-VQA conflict is 2x any other pair (magnitude 6.15)
- Geometry is the most isolated expert
- Chart is most affected by cross-expert interference

**Proposal**: Optimize per-expert scaling within Tucker framework:
- Add learnable weights w_i to the loss: L = Σ_i w_i · ||(m - τ̃_i) @ L_i^T||² / ||τ_i||²
- Or use micro-capability conflict scores to set initial weights

### 7.6 Full Evaluation

Current results are on 3 benchmarks only (TextVQA, ChartQA, OCRVQA). For paper submission, evaluate on all 7: TextVQA, OCRVQA, VizWiz, GQA, ChartQA, MathVision, MathVista. The additional benchmarks may reveal different tradeoff patterns.

## 8. Priority Ranking for Next Experiments

| Priority | Experiment | Expected Gain | Effort |
|----------|-----------|---------------|--------|
| 1 | Tucker k=60 | Fix OCRVQA regression | Low (just change hyperparameter) |
| 2 | Tucker k=80 | Further OCRVQA improvement | Low |
| 3 | MC-Tucker-wudi2 (A+C) | Best of both worlds | Medium |
| 4 | Adaptive Tucker rank | Optimal per-layer compression | Medium |
| 5 | ANOVA-Tucker (B+C) | Theoretical elegance + performance | High |
| 6 | Full 7-benchmark eval | Paper-ready comparison | Low (just run eval) |
| 7 | Per-expert scaling | Fine-grained control | Medium |
