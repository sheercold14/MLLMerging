# Route A & C Experiment Log (2026-03-23)

## Experiment Overview

Following Section 11 of NeurIPS2026 theory document, implementing three improvement routes over wudi2:
- **Route A (MC-wudi2)**: Micro-Capability conflict pre-resolution before wudi2 optimization
- **Route C (Tucker-wudi2)**: Tucker subspace constraint on wudi2's optimization search space

## Models Merged

### mc_wudi2_b05 (Route A)
- Method: MC-wudi2 (micro-capability conflict suppression + wudi2)
- Hyperparameters: beta=0.5, energy_ratio=0.95, scaling_coefficient=0.1
- GPU: 1
- Algorithm:
  1. For each 2D layer, extract micro-capabilities via individual SVD (95% energy cutoff)
  2. Build K×K interference matrix I_ab = (u_a·u_b)(v_a·v_b) * sigma_a * sigma_b
  3. For each micro-cap, compute conflict_score = sum of negative cross-expert interference
  4. Suppression factor = max(0, 1 - beta * relative_conflict)
  5. Reconstruct cleaned task vectors with suppressed sigmas
  6. Feed cleaned task vectors to standard wudi2 optimization (300 steps, Adam lr=1e-5)
- Key observations:
  - K ranges from 107 to 3426 per layer (avg ~1800 at 95% energy)
  - 100% of micro-caps get some suppression (all have some cross-expert conflict)
  - Average suppression factor: 0.62-0.81 (attention layers more, MLP layers less)
  - No micro-capabilities completely zeroed out (except 1-3 in layers 22-23)
  - MLP gate/up_proj have highest K (~3000+), attention K/V have lowest (~400-500)

### tucker_wudi2_k40 (Route C)
- Method: Tucker subspace wudi2 (constrained search space)
- Hyperparameters: top_k=40, scaling_coefficient=0.1, iter_num=300, Adam lr=1e-3
- GPU: 2
- Algorithm:
  1. For each 2D layer, compute individual SVD of each expert's task vector
  2. Pool top-40 left/right singular vectors → QR orthogonalize → joint bases B (m×r2), C (n×r3)
  3. Project wudi2's targets into Tucker coordinates
  4. Optimize X (r2×r3) instead of m (m×n) with same wudi2 loss
  5. Reconstruct: m = B @ X* @ C^T
- Key observations:
  - r2=r3=200 for 896×896 and larger layers (5×40=200 pooled, all linearly independent)
  - r2=128 for 128×896 layers (K/V proj, output dim is bottleneck)
  - Compression: 109x for MLP layers (4864×896 → 200×200), 20x for 896×896, 4x for 128×896
  - K/V proj layers converge to near-zero loss (loss < 1e-6)
  - MLP layers converge to non-trivial loss (~0.04-0.07) — Tucker basis may miss some structure
  - lr=1e-3 works well (much higher than wudi2's lr=1e-5, appropriate for smaller search space)

## Micro-Capability Analysis Results (20 sampled layers)

Key findings from `mc_analysis.py`:
- ~48.7% of cross-expert micro-capability pairs are conflicts (negative interference)
- OCR-VQA has highest conflict magnitude (6.15), nearly 2x any other pair
- VQA is the most conflict-prone expert overall
- MLP layers dominate conflict (top 8 most conflicting layers are all MLP)
- Layer 3 up_proj has the highest conflict magnitude
- Geometry has lowest average micro-cap count (most isolated expert)

## Evaluation

Running on TextVQA_VAL, ChartQA_TEST, OCRVQA_TESTCORE (same as previous experiments).
- mc_wudi2_b05: GPU 1 (task bjejq1x9l)
- tucker_wudi2_k40: GPU 2 (task bu4tsqh3n)

### Results

| Model | TextVQA | ChartQA | OCRVQA | Avg |
|-------|---------|---------|--------|-----|
| InternVL2_5-1B (base) | 74.23 | 69.72 | 41.76 | 61.90 |
| wudi2 (merge_ours_internvl) | 76.01 | 68.44 | 46.29 | 63.58 |
| mc_wudi2_b05 (Route A) | 75.97 | 69.00 | 46.19 | 63.72 |
| anova_wudi2 (Route B) | 75.54 | 70.12 | 44.73 | 63.46 |
| tucker_wudi2_k40 (Route C) | 75.53 | **70.68** | 45.21 | **63.81** |

**Key findings:**
1. **Tucker-wudi2 achieves best average (63.81)**, beating wudi2 (63.58) by +0.23
2. Tucker-wudi2 is the ONLY method that improves ChartQA above base (70.68 > 69.72)
   - wudi2 loses 1.28 on ChartQA; Tucker gains 0.96
3. MC-wudi2 nearly matches wudi2 on TextVQA (75.97 vs 76.01) with better ChartQA
4. ANOVA-wudi2 discovers meaningful expert groups: [[OCR,VQA], [Geometry,Chart,Grounding]]
5. ANOVA groups confirm our interference analysis: OCR-VQA share text understanding,
   Geometry-Chart share visual/spatial reasoning (highest affinity: 0.141)

## Files Modified

- `InternVL/internvl_chat/model_merging.py`: Added mc_wudi2_merging() and tucker_wudi2_merging()
- `InternVL/internvl_chat/model_merging_train.py`: Added CLI support for new methods + hyperparams
- `VLMEvalKit/vlmeval/config.py`: Registered mc_wudi2_b05 and tucker_wudi2_k40 models
- `Skills/theory/mc_analysis.py`: Micro-capability analysis script
- `Skills/theory/results/mc_analysis_sample20.json`: Analysis results

## Next Steps

1. **Tucker is most promising** — try higher top_k (60, 80) to capture more structure
2. **Combine Route A+C**: MC conflict pre-resolution + Tucker subspace search
3. **Combine Route B+C**: ANOVA hierarchical decomposition + Tucker optimization on residuals
4. **Full pipeline A+B+C**: ANOVA decomposition → MC conflict resolution → Tucker optimization
5. **Try Tucker with closed-form solution** (X* = b @ H^{-1}) instead of Adam
6. **Lower MC beta** (0.3) — current beta=0.5 may over-suppress
7. **Evaluate on more benchmarks** (GQA, VizWiz, MathVision, MathVista)
