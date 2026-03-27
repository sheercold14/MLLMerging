# ETVD Foundational Analysis Report

> Generated: 2026-03-22 | Model: InternVL2.5-1B | 5 Experts | 170 2D layers | 850 basis directions

---

## 1. Ensemble SVD Summary

Task vectors from 5 experts stacked per-layer into T in R^{5 x d}, SVD decomposed.

| Metric | Value |
|--------|-------|
| Total layers analyzed | 170 |
| Total basis directions (5 per layer) | 850 |
| **Unique directions** | **733 (86.2%)** |
| Shared directions (multi-expert, same sign) | 51 (6.0%) |
| Conflict directions (multi-expert, mixed sign) | 66 (7.8%) |
| Mean occupancy | 1.16 |
| Median occupancy | 1.0 |

### Occupancy Histogram

| Occupancy | Count | Fraction |
|-----------|-------|----------|
| 1 (unique) | 733 | 86.2% |
| 2 | 103 | 12.1% |
| 3 | 12 | 1.4% |
| 4 | 2 | 0.2% |
| 5 (all experts) | 0 | 0.0% |

**Key finding**: The vast majority (86%) of basis directions are unique to a single expert. Only ~14% show multi-expert overlap. No direction is shared by all 5 experts.

---

## 2. Pairwise Cosine Similarity

Average cosine similarity across all 170 layers:

|  | OCR | VQA | Geometry | Chart | Grounding |
|---|---|---|---|---|---|
| **OCR** | 1.0000 | 0.0423 | 0.0071 | 0.0257 | **0.1517** |
| **VQA** | 0.0423 | 1.0000 | 0.0065 | 0.0168 | 0.0394 |
| **Geometry** | 0.0071 | 0.0065 | 1.0000 | 0.0121 | 0.0066 |
| **Chart** | 0.0257 | 0.0168 | 0.0121 | 1.0000 | 0.0159 |
| **Grounding** | **0.1517** | 0.0394 | 0.0066 | 0.0159 | 1.0000 |

**Key findings**:
- **OCR-Grounding** have the highest pairwise similarity (0.1517), significantly above others
- **Geometry** is the most isolated expert (all cosines < 0.013)
- Overall cosine values are very low (all < 0.16), confirming task vectors largely live in separate subspaces
- This explains why removing Grounding has minimal effect: it's most similar to OCR, so OCR "covers" its shared directions

---

## 3. Sign Conflict Rate (Magnitude-Weighted)

|  | OCR | VQA | Geometry | Chart | Grounding |
|---|---|---|---|---|---|
| **OCR** | 0 | 0.4668 | 0.4944 | 0.4797 | **0.3818** |
| **VQA** | 0.4668 | 0 | 0.4948 | 0.4867 | 0.4687 |
| **Geometry** | 0.4944 | 0.4948 | 0 | 0.4903 | 0.4948 |
| **Chart** | 0.4797 | 0.4867 | 0.4903 | 0 | 0.4873 |
| **Grounding** | **0.3818** | 0.4687 | 0.4948 | 0.4873 | 0 |

**Key findings**:
- Most pairs have ~49-50% sign conflict (essentially random sign alignment, consistent with near-orthogonality)
- **OCR-Grounding** have the lowest conflict (38.2%), consistent with their higher cosine similarity — they tend to agree on direction
- **Geometry** has ~49.5% conflict with everyone (purely random, confirming maximum independence)

---

## 4. Expert Unique Direction Ownership

| Expert | Unique Directions | Fraction |
|--------|------------------|----------|
| OCR | 166 | 19.5% |
| VQA | 168 | 19.8% |
| Geometry | 158 | 18.6% |
| Chart | 157 | 18.5% |
| Grounding | 167 | 19.6% |

Near-uniform distribution. Each expert owns approximately 1/5 of all unique directions, confirming balanced representation in the ensemble SVD basis.

---

## 5. Per-Component Analysis

Average pairwise cosine similarity by parameter type:

| Component | #Layers | Mean Cosine | Std |
|-----------|---------|-------------|-----|
| attn_k | 24 | **0.0449** | 0.0204 |
| attn_v | 24 | **0.0404** | 0.0146 |
| attn_q | 24 | 0.0343 | 0.0112 |
| attn_o | 24 | 0.0313 | 0.0064 |
| mlp_down | 24 | 0.0265 | 0.0052 |
| mlp_up | 24 | 0.0265 | 0.0052 |
| mlp_gate | 24 | 0.0207 | 0.0041 |
| other | 2 | 0.0575 | 0.0097 |

**Key findings**:
- **Attention layers (especially K, V)** have higher cross-expert similarity than MLP layers
- MLP gate has the lowest similarity (most independent)
- This suggests interference is concentrated in attention mechanisms, not MLP

---

## 6. Layer Depth Analysis

Average pairwise cosine by transformer layer depth:

| Layer | Cosine | Layer | Cosine |
|-------|--------|-------|--------|
| 0 | **0.0464** | 12 | **0.0397** |
| 1 | 0.0329 | 13 | 0.0354 |
| 2 | 0.0288 | 14 | 0.0345 |
| 3 | 0.0342 | 15 | 0.0335 |
| 4 | 0.0234 | 16 | 0.0281 |
| 5 | 0.0237 | 17 | 0.0274 |
| 6 | 0.0286 | 18 | 0.0303 |
| 7 | 0.0316 | 19 | 0.0349 |
| 8 | 0.0325 | 20 | 0.0309 |
| 9 | 0.0303 | 21 | 0.0290 |
| 10 | 0.0348 | 22 | 0.0300 |
| 11 | 0.0307 | 23 | **0.0460** |

**Key findings**:
- **First layer (0) and last layer (23)** show highest cross-expert similarity (U-shaped pattern)
- Middle layers (4-5) have the lowest similarity
- This U-shape suggests: early layers learn shared low-level features, late layers learn shared output formatting, and middle layers are where task-specific specialization happens

---

## 7. Spectral Over-Accumulation

Amplification ratio = singular values of sum(task vectors) / average individual singular values.
Reference: sqrt(5) = 2.24 (orthogonal), 5.0 (fully aligned).

| Layer | Component | Top-5 Amplification |
|-------|-----------|---------------------|
| 0 | q_proj | 2.57, 3.18, 3.04, 2.56, 2.59 |
| 20 | down_proj | 2.14, 2.14, 2.13, 2.00, 2.00 |
| 40 | up_proj | 2.10, 2.11, 2.03, 2.01, 2.04 |
| 60 | gate_proj | 1.98, 2.11, 1.96, 1.89, 1.92 |
| 80 | o_proj | 1.99, 2.07, 2.08, 2.14, 2.19 |
| 100 | v_proj | **2.42, 2.42, 2.28, 2.32, 2.28** |
| 120 | k_proj | 2.22, 2.19, 2.28, 1.98, 2.14 |
| 140 | q_proj | 2.14, 2.24, 2.15, 2.12, 2.18 |
| 160 | down_proj | **2.86, 2.51, 2.28, 2.24, 2.30** |

**Key findings**:
- Most amplification ratios are around **2.0-2.3**, close to sqrt(5)=2.24, confirming **near-orthogonal** task vectors
- **Layer 0 (early attention)** and **Layer 160 (late MLP)** show amplification **above** sqrt(5), indicating partial alignment / over-accumulation
- The top singular value at Layer 0 reaches 3.18 (significantly above 2.24), meaning the dominant direction is shared by multiple experts
- MLP gate layers (Layer 60) show amplification **below** sqrt(5), suggesting even more independence than average

---

## 8. Summary of Findings

### For the ETVD theory:

1. **Occupancy structure is sparse**: 86% unique, 14% shared/conflict. This validates the ETVD decomposition — most directions need no special treatment, only the 14% require occupancy-aware handling.

2. **OCR-Grounding coupling**: The strongest pairwise coupling (cosine=0.1517) explains why removing Grounding is neutral — OCR already covers the shared directions.

3. **Attention > MLP in interference**: K/V attention has 2x the cosine similarity of MLP gate, suggesting merge strategies should prioritize attention layers.

4. **U-shaped depth pattern**: Early/late layers share more, middle layers are task-specific. Layer-wise merge coefficients should vary accordingly.

5. **Moderate over-accumulation**: Amplification around sqrt(5) means task vectors are mostly orthogonal, but specific layers (early attention, late MLP) show genuine over-accumulation that naive sum doesn't handle.

### Implications for method design:

- The 14% non-unique directions are where ETVD's occupancy-aware reconstruction adds value over naive Task Arithmetic
- The attention-vs-MLP and depth patterns suggest per-component or per-layer strategies could improve results
- The OCR-Grounding coupling should be handled specially (e.g., reduced coefficient for redundant shared directions)
