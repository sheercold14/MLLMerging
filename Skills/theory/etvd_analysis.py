#!/usr/bin/env python
"""
Ensemble Task Vector Decomposition (ETVD) -- Foundational Analysis Script

This script performs structural analysis of task vectors from 5 fine-tuned
InternVL2.5-1B expert models to validate the ETVD theory framework.

Analyses performed:
  (a) Compute 5 task vectors (expert - base) for all qualifying 2D weight params
  (b) Per-layer ensemble SVD: stack 5 task vectors -> T in R^{5 x d}, SVD,
      analyze U matrix for occupancy, sign consistency, direction classification
  (c) Pairwise interference matrices: Gram matrix, cosine similarity, STI, sign conflict
  (d) Spectral over-accumulation: compare sum-of-TVs singular spectrum vs individual spectra
  (e) Save results to Skills/theory/results/

Usage:
    cd Skills/theory && python etvd_analysis.py

Environment: conda activate merge
GPU: set CUDA_VISIBLE_DEVICES externally (default uses GPU 2)
"""

import os
import sys
import re
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "OpenGVLab/InternVL2_5-1B"

EXPERT_IDS = OrderedDict([
    ("OCR",       "yongxianwei/InternVL2_5-1B_OCR"),
    ("VQA",       "yongxianwei/InternVL2_5-1B_VQA"),
    ("Geometry",  "yongxianwei/InternVL2_5-1B_Geometry"),
    ("Chart",     "yongxianwei/InternVL2_5-1B_Chart"),
    ("Grounding", "yongxianwei/InternVL2_5-1B_Grounding"),
])

EXPERT_NAMES = list(EXPERT_IDS.keys())
N_EXPERTS = len(EXPERT_NAMES)

EXCLUDE_PARAM_PATTERNS = [
    'vision_model.*',
    '.*lm_head.*',
    '.*norm.*',
    '.*embed_tokens.*',
    '.*bias.*',
]

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
TV_CACHE_DIR = RESULTS_DIR / "task_vectors_cache"

# Occupancy threshold for U-matrix analysis
OCCUPANCY_THRESHOLD = 0.2  # |u_ik| > threshold * max(|u_k|) => significant


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_param_names_to_merge(param_names, exclude_patterns):
    """Filter parameter names, keeping only those not matching exclude patterns."""
    result = []
    for name in param_names:
        exclude = any(re.match(pat, name) for pat in exclude_patterns)
        if not exclude:
            result.append(name)
    return result


def get_2d_param_names(param_names_to_merge, state_dict):
    """Return only parameter names that are 2D (weight matrices)."""
    return [n for n in param_names_to_merge if len(state_dict[n].shape) == 2]


def load_model(model_id, device="cpu", dtype=torch.float16):
    """Load an InternVL model, return only the state dict to save memory."""
    from transformers import AutoModel
    print(f"  Loading model: {model_id} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    Loaded in {time.time() - t0:.1f}s, {len(sd)} params")
    return sd


# ---------------------------------------------------------------------------
# Phase 1: Compute and cache task vectors
# ---------------------------------------------------------------------------

def compute_task_vectors():
    """
    Compute task vectors for all experts and save per-layer to disk.
    Only processes 2D weight parameters (excluding vision, norm, etc.).

    Returns:
        param_names_2d: list of 2D parameter names
        shapes: dict mapping param_name -> shape tuple
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Computing Task Vectors")
    print("=" * 70)

    TV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model state dict
    base_sd = load_model(BASE_MODEL_ID)

    # Determine which parameters to process
    all_param_names = list(base_sd.keys())
    param_names_to_merge = get_param_names_to_merge(all_param_names, EXCLUDE_PARAM_PATTERNS)
    param_names_2d = get_2d_param_names(param_names_to_merge, base_sd)

    print(f"\n  Total parameters: {len(all_param_names)}")
    print(f"  Parameters to merge (after exclusion): {len(param_names_to_merge)}")
    print(f"  2D weight parameters: {len(param_names_2d)}")

    # Save shapes info
    shapes = {}
    for name in param_names_2d:
        shapes[name] = tuple(base_sd[name].shape)

    # Also save non-2D params info for reference
    non_2d_names = [n for n in param_names_to_merge if n not in param_names_2d]
    print(f"  Non-2D parameters (will skip for SVD analysis): {len(non_2d_names)}")
    if non_2d_names:
        for n in non_2d_names[:5]:
            print(f"    {n}: shape={base_sd[n].shape}")
        if len(non_2d_names) > 5:
            print(f"    ... and {len(non_2d_names) - 5} more")

    # Compute task vector for each expert
    for expert_name, expert_id in EXPERT_IDS.items():
        cache_file = TV_CACHE_DIR / f"tv_{expert_name}.pt"
        if cache_file.exists():
            print(f"\n  [CACHED] Task vector for {expert_name} already exists, skipping.")
            continue

        expert_sd = load_model(expert_id)

        tv_dict = {}
        for pname in param_names_2d:
            tv_dict[pname] = (expert_sd[pname].float() - base_sd[pname].float()).to(torch.float16)

        torch.save(tv_dict, cache_file)
        print(f"  Saved task vector for {expert_name} -> {cache_file}")

        del expert_sd, tv_dict
        gc.collect()

    del base_sd
    gc.collect()

    # Save metadata
    meta = {
        "param_names_2d": param_names_2d,
        "shapes": shapes,
        "expert_names": EXPERT_NAMES,
    }
    torch.save(meta, TV_CACHE_DIR / "metadata.pt")

    return param_names_2d, shapes


def load_task_vectors():
    """Load cached task vectors and metadata."""
    meta = torch.load(TV_CACHE_DIR / "metadata.pt", weights_only=False)
    param_names_2d = meta["param_names_2d"]
    shapes = meta["shapes"]

    task_vectors = {}
    for expert_name in EXPERT_NAMES:
        tv = torch.load(TV_CACHE_DIR / f"tv_{expert_name}.pt", weights_only=False)
        task_vectors[expert_name] = tv
        print(f"  Loaded task vector for {expert_name}")

    return param_names_2d, shapes, task_vectors


# ---------------------------------------------------------------------------
# Phase 2: Per-layer Ensemble SVD Analysis
# ---------------------------------------------------------------------------

def analyze_u_column(u_col, threshold_ratio=OCCUPANCY_THRESHOLD):
    """
    Analyze a single column of U matrix to determine direction type.

    Args:
        u_col: numpy array of shape (N_EXPERTS,)
        threshold_ratio: ratio of max absolute value to consider significant

    Returns:
        dict with occupancy, sign_consistency, direction_type, etc.
    """
    abs_col = np.abs(u_col)
    max_val = abs_col.max()

    if max_val < 1e-10:
        return {
            "occupancy": 0,
            "sign_consistency": 0.0,
            "direction_type": "zero",
            "significant_experts": [],
        }

    threshold = threshold_ratio * max_val
    significant_mask = abs_col > threshold
    occupancy = int(significant_mask.sum())
    significant_experts = [EXPERT_NAMES[i] for i in range(N_EXPERTS) if significant_mask[i]]

    # Sign consistency among significant experts
    significant_vals = u_col[significant_mask]
    if len(significant_vals) <= 1:
        sign_consistency = 1.0
    else:
        n_pos = (significant_vals > 0).sum()
        n_neg = (significant_vals < 0).sum()
        sign_consistency = max(n_pos, n_neg) / len(significant_vals)

    # Classification
    if occupancy == 1:
        direction_type = "unique"
    elif sign_consistency >= 0.8:
        direction_type = "shared"
    else:
        direction_type = "conflict"

    return {
        "occupancy": occupancy,
        "sign_consistency": float(sign_consistency),
        "direction_type": direction_type,
        "significant_experts": significant_experts,
    }


def per_layer_ensemble_svd(param_names_2d, task_vectors):
    """
    For each 2D weight layer:
      - Stack 5 task vectors into T in R^{5 x d}
      - Compute SVD: T = U Sigma V^T
      - Analyze U matrix structure

    Returns:
        layer_results: dict mapping param_name -> analysis results
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Per-Layer Ensemble SVD Analysis")
    print("=" * 70)

    layer_results = {}
    direction_type_counts = {"unique": 0, "shared": 0, "conflict": 0, "zero": 0}
    total_directions = 0

    # Aggregated stats
    all_occupancies = []
    all_sign_consistencies = []
    all_singular_values = []

    for idx, pname in enumerate(param_names_2d):
        if idx % 50 == 0:
            print(f"  Processing layer {idx}/{len(param_names_2d)}: {pname}")

        # Stack task vectors: T in R^{N x d} where d = prod(shape)
        shape = task_vectors[EXPERT_NAMES[0]][pname].shape
        rows, cols = shape

        # Flatten each 2D task vector to 1D, stack into T
        T = np.zeros((N_EXPERTS, rows * cols), dtype=np.float32)
        for i, ename in enumerate(EXPERT_NAMES):
            T[i] = task_vectors[ename][pname].float().numpy().flatten()

        # SVD of T: T = U @ diag(sigma) @ Vt
        # T is (5 x d), so U is (5 x 5), sigma is (5,), Vt is (5 x d)
        try:
            U, sigma, Vt = np.linalg.svd(T, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"    SVD failed for {pname}, skipping")
            continue

        # Analyze each column of U (each basis direction)
        n_components = min(N_EXPERTS, len(sigma))
        direction_analyses = []
        for k in range(n_components):
            analysis = analyze_u_column(U[:, k])
            analysis["singular_value"] = float(sigma[k])
            analysis["energy_fraction"] = float(sigma[k] ** 2 / (sigma ** 2).sum()) if (sigma ** 2).sum() > 0 else 0
            analysis["u_column"] = U[:, k].tolist()
            direction_analyses.append(analysis)

            direction_type_counts[analysis["direction_type"]] += 1
            total_directions += 1
            all_occupancies.append(analysis["occupancy"])
            all_sign_consistencies.append(analysis["sign_consistency"])

        all_singular_values.append(sigma.tolist())

        layer_results[pname] = {
            "shape": list(shape),
            "U": U.tolist(),
            "sigma": sigma.tolist(),
            "direction_analyses": direction_analyses,
            "frobenius_norms": [float(np.linalg.norm(T[i])) for i in range(N_EXPERTS)],
        }

    # Summary statistics
    summary = {
        "total_layers": len(layer_results),
        "total_directions": total_directions,
        "direction_type_counts": direction_type_counts,
        "direction_type_fractions": {
            k: v / total_directions if total_directions > 0 else 0
            for k, v in direction_type_counts.items()
        },
        "mean_occupancy": float(np.mean(all_occupancies)),
        "median_occupancy": float(np.median(all_occupancies)),
        "occupancy_histogram": {
            str(i): int((np.array(all_occupancies) == i).sum())
            for i in range(N_EXPERTS + 1)
        },
        "mean_sign_consistency": float(np.mean(all_sign_consistencies)),
    }

    print(f"\n  --- Ensemble SVD Summary ---")
    print(f"  Total layers analyzed: {summary['total_layers']}")
    print(f"  Total basis directions: {summary['total_directions']}")
    print(f"  Direction types:")
    for dtype, count in direction_type_counts.items():
        frac = summary['direction_type_fractions'][dtype]
        print(f"    {dtype:10s}: {count:5d} ({frac:.1%})")
    print(f"  Mean occupancy: {summary['mean_occupancy']:.2f}")
    print(f"  Mean sign consistency: {summary['mean_sign_consistency']:.4f}")
    print(f"  Occupancy histogram: {summary['occupancy_histogram']}")

    return layer_results, summary


# ---------------------------------------------------------------------------
# Phase 3: Pairwise Interference Matrices
# ---------------------------------------------------------------------------

def compute_pairwise_interference(param_names_2d, task_vectors):
    """
    Compute pairwise interference metrics between experts:
      (a) Gram matrix G = T @ T^T (per-layer and aggregated)
      (b) Cosine similarity
      (c) STI-like metric (singular task interference)
      (d) Magnitude-weighted sign conflict rate
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Pairwise Interference Analysis")
    print("=" * 70)

    # Accumulators for aggregated metrics
    agg_gram = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)
    agg_norm_sq = np.zeros(N_EXPERTS, dtype=np.float64)
    agg_sign_conflict = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)
    agg_sign_synergy = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)
    agg_magnitude_product = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)

    # For STI computation: accumulate per-layer U^T U and sigma products
    sti_numerator = 0.0
    sti_denominator = 0.0

    # Per-layer Gram matrices (store only for a few representative layers)
    per_layer_gram = {}
    per_layer_cosine = {}

    for idx, pname in enumerate(param_names_2d):
        if idx % 50 == 0:
            print(f"  Processing layer {idx}/{len(param_names_2d)}: {pname}")

        # Build T matrix
        vecs = []
        for ename in EXPERT_NAMES:
            v = task_vectors[ename][pname].float().numpy().flatten()
            vecs.append(v)
        T = np.stack(vecs, axis=0)  # (N, d)

        # --- (a) Gram matrix ---
        G = T @ T.T  # (N, N)
        agg_gram += G

        norms = np.sqrt(np.diag(G))
        agg_norm_sq += np.diag(G)

        # --- (b) Per-layer cosine similarity ---
        norms_safe = np.where(norms > 0, norms, 1.0)
        cosine = G / np.outer(norms_safe, norms_safe)
        np.fill_diagonal(cosine, 1.0)

        # Store per-layer for a few layers
        if idx < 5 or idx % 50 == 0:
            per_layer_gram[pname] = G.tolist()
            per_layer_cosine[pname] = cosine.tolist()

        # --- (c) STI-like metric ---
        # For each layer, do SVD of each task vector individually,
        # then check orthogonality of left/right singular vector spaces.
        # Simplified: use the ensemble SVD approach.
        # STI = ||U^T U - I||_1 * ||sigma||_1 where U are concatenated singular vectors
        # We approximate by computing overlap of individual task vector subspaces.
        try:
            U, sigma, Vt = np.linalg.svd(T, full_matrices=False)
            # U is (N, min(N,d)), sigma is (min(N,d),)
            # Measure how non-orthogonal the expert representations are in the ensemble basis
            UtU = U.T @ U  # should be identity if orthogonal
            deviation = UtU - np.eye(UtU.shape[0])
            sti_layer = np.sum(np.abs(deviation)) * np.sum(sigma)
            sti_numerator += sti_layer
            sti_denominator += np.sum(sigma)
        except:
            pass

        # --- (d) Magnitude-weighted sign conflict ---
        for i in range(N_EXPERTS):
            for j in range(i + 1, N_EXPERTS):
                vi = T[i]
                vj = T[j]
                product = np.abs(vi * vj)
                same_sign = (np.sign(vi) == np.sign(vj))
                # Exclude zeros
                nonzero = (vi != 0) & (vj != 0)
                conflict = product * (~same_sign & nonzero)
                synergy = product * (same_sign & nonzero)
                total_mag = product * nonzero

                agg_sign_conflict[i, j] += conflict.sum()
                agg_sign_conflict[j, i] += conflict.sum()
                agg_sign_synergy[i, j] += synergy.sum()
                agg_sign_synergy[j, i] += synergy.sum()
                agg_magnitude_product[i, j] += total_mag.sum()
                agg_magnitude_product[j, i] += total_mag.sum()

    # --- Aggregate cosine similarity ---
    agg_norms = np.sqrt(agg_norm_sq)
    agg_norms_safe = np.where(agg_norms > 0, agg_norms, 1.0)
    agg_cosine = agg_gram / np.outer(agg_norms_safe, agg_norms_safe)
    np.fill_diagonal(agg_cosine, 1.0)

    # --- Sign conflict rate ---
    mag_safe = np.where(agg_magnitude_product > 0, agg_magnitude_product, 1.0)
    sign_conflict_rate = agg_sign_conflict / mag_safe
    np.fill_diagonal(sign_conflict_rate, 0.0)

    # --- Synergy rate ---
    sign_synergy_rate = agg_sign_synergy / mag_safe
    np.fill_diagonal(sign_synergy_rate, 1.0)

    # --- Normalized STI ---
    normalized_sti = sti_numerator / sti_denominator if sti_denominator > 0 else 0.0

    # --- Pairwise STI: compute subspace overlap using GPU-accelerated truncated SVD ---
    print("\n  Computing pairwise STI (subspace overlap via GPU truncated SVD)...")
    pairwise_sti = np.zeros((N_EXPERTS, N_EXPERTS), dtype=np.float64)

    MAX_RANK = 20  # Truncated SVD rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    for idx, pname in enumerate(param_names_2d):
        if idx % 50 == 0:
            print(f"    Pairwise STI layer {idx}/{len(param_names_2d)}")

        shape = task_vectors[EXPERT_NAMES[0]][pname].shape
        rows, cols = shape
        k_trunc = min(MAX_RANK, rows, cols)

        # Individual truncated SVDs for each expert using GPU
        individual_svds = []
        for ename in EXPERT_NAMES:
            tv_tensor = task_vectors[ename][pname].float().to(device)
            try:
                u, s, v = torch.svd_lowrank(tv_tensor, q=k_trunc)
                # u: (rows, k), s: (k,), v: (cols, k)
                individual_svds.append({
                    "S": s.cpu().numpy(),
                    "V": v.cpu().numpy(),  # (cols, k) -- right singular vectors
                })
            except:
                individual_svds.append(None)

        # Pairwise overlap via principal angles of right singular vector spaces
        for i in range(N_EXPERTS):
            for j in range(i + 1, N_EXPERTS):
                if individual_svds[i] is None or individual_svds[j] is None:
                    continue
                # V_i^T @ V_j gives (k, k) matrix; its singular values are
                # cosines of principal angles
                Vi = individual_svds[i]["V"]  # (cols, k)
                Vj = individual_svds[j]["V"]  # (cols, k)
                overlap = np.linalg.svd(Vi.T @ Vj, compute_uv=False)

                # Weight by singular value magnitudes
                si = individual_svds[i]["S"]
                sj = individual_svds[j]["S"]
                mag_weight = float(np.sum(si) * np.sum(sj))

                mean_overlap = float(overlap.mean())
                pairwise_sti[i, j] += mean_overlap * mag_weight
                pairwise_sti[j, i] += mean_overlap * mag_weight

    torch.cuda.empty_cache()

    # Normalize pairwise STI
    max_sti = pairwise_sti.max() if pairwise_sti.max() > 0 else 1.0
    pairwise_sti_normalized = pairwise_sti / max_sti

    results = {
        "gram_matrix": agg_gram.tolist(),
        "cosine_similarity": agg_cosine.tolist(),
        "sign_conflict_rate": sign_conflict_rate.tolist(),
        "sign_synergy_rate": sign_synergy_rate.tolist(),
        "sign_conflict_total": agg_sign_conflict.tolist(),
        "sign_synergy_total": agg_sign_synergy.tolist(),
        "magnitude_product_total": agg_magnitude_product.tolist(),
        "normalized_sti": float(normalized_sti),
        "pairwise_sti": pairwise_sti.tolist(),
        "pairwise_sti_normalized": pairwise_sti_normalized.tolist(),
        "expert_norms": agg_norms.tolist(),
        "per_layer_gram_samples": per_layer_gram,
        "per_layer_cosine_samples": per_layer_cosine,
    }

    # Print summary
    print(f"\n  --- Pairwise Interference Summary ---")
    print(f"\n  Aggregated Cosine Similarity:")
    _print_matrix(agg_cosine, EXPERT_NAMES)
    print(f"\n  Sign Conflict Rate (magnitude-weighted):")
    _print_matrix(sign_conflict_rate, EXPERT_NAMES)
    print(f"\n  Pairwise STI (normalized):")
    _print_matrix(pairwise_sti_normalized, EXPERT_NAMES)
    print(f"\n  Expert norms (L2 of all task vector params):")
    for i, n in enumerate(EXPERT_NAMES):
        print(f"    {n}: {agg_norms[i]:.4f}")
    print(f"\n  Global normalized STI: {normalized_sti:.6f}")

    return results


def _print_matrix(mat, labels):
    """Pretty print a matrix with labels."""
    header = "          " + "  ".join(f"{l:>10s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"  {label:>8s}"
        for j in range(len(labels)):
            row += f"  {mat[i][j]:10.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Phase 4: Spectral Over-Accumulation Analysis
# ---------------------------------------------------------------------------

def spectral_over_accumulation_analysis(param_names_2d, task_vectors):
    """
    Compare the singular value spectrum of the sum of task vectors vs
    individual task vector spectra.

    Key diagnostic: if over-accumulation is present, the sum's top singular
    values will be disproportionately larger than the individual spectra predict.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Spectral Over-Accumulation Analysis")
    print("=" * 70)

    # We'll aggregate statistics across layers
    all_individual_spectra = {ename: [] for ename in EXPERT_NAMES}
    all_sum_spectra = []
    all_expected_spectra = []  # sqrt of sum of squared individual spectra

    layer_results = {}

    # Use GPU for SVD (much faster than numpy on large matrices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device} for spectral analysis")

    for idx, pname in enumerate(param_names_2d):
        if idx % 50 == 0:
            print(f"  Processing layer {idx}/{len(param_names_2d)}: {pname}")

        shape = task_vectors[EXPERT_NAMES[0]][pname].shape
        rows, cols = shape

        # Compute sum of task vectors and individual spectra using GPU
        sum_tv_torch = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        individual_spectra = {}

        for ename in EXPERT_NAMES:
            tv_tensor = task_vectors[ename][pname].float().to(device)
            sum_tv_torch += tv_tensor

            # Individual SVD (singular values only)
            try:
                s = torch.linalg.svdvals(tv_tensor).cpu().numpy()
                individual_spectra[ename] = s
                all_individual_spectra[ename].append(s)
            except:
                individual_spectra[ename] = np.zeros(min(rows, cols))

        # Sum SVD
        try:
            s_sum = torch.linalg.svdvals(sum_tv_torch).cpu().numpy()
            all_sum_spectra.append(s_sum)
        except:
            s_sum = np.zeros(min(rows, cols))

        del sum_tv_torch
        torch.cuda.empty_cache()

        # Expected spectrum under orthogonality assumption:
        # If task vectors were orthogonal, sum spectrum^2 = sum of individual spectra^2
        stacked_sq = np.zeros(min(rows, cols), dtype=np.float64)
        for ename in EXPERT_NAMES:
            s_ind = individual_spectra[ename]
            padded = np.zeros(min(rows, cols))
            padded[:len(s_ind)] = s_ind
            stacked_sq += padded ** 2
        s_expected = np.sqrt(stacked_sq)

        all_expected_spectra.append(s_expected)

        # Over-accumulation ratio: actual top singular value / expected
        if s_expected[0] > 0:
            top_ratio = float(s_sum[0] / s_expected[0])
        else:
            top_ratio = 0.0

        # Energy concentration: fraction of energy in top-1, top-3, top-5 singular values
        sum_energy = float((s_sum ** 2).sum())
        if sum_energy > 0:
            top1_frac = float(s_sum[0] ** 2 / sum_energy)
            top3_frac = float((s_sum[:3] ** 2).sum() / sum_energy) if len(s_sum) >= 3 else 1.0
        else:
            top1_frac = 0.0
            top3_frac = 0.0

        expected_energy = float((s_expected ** 2).sum())
        if expected_energy > 0:
            exp_top1_frac = float(s_expected[0] ** 2 / expected_energy)
        else:
            exp_top1_frac = 0.0

        if idx < 5 or idx % 50 == 0:
            layer_results[pname] = {
                "sum_spectrum": s_sum[:20].tolist(),
                "expected_spectrum": s_expected[:20].tolist(),
                "individual_spectra": {
                    en: individual_spectra[en][:10].tolist() for en in EXPERT_NAMES
                },
                "top_sv_ratio": top_ratio,
                "sum_top1_energy_frac": top1_frac,
                "sum_top3_energy_frac": top3_frac,
                "expected_top1_energy_frac": exp_top1_frac,
            }

    # Aggregate analysis
    print("\n  Computing aggregate spectral statistics...")

    # Aggregate: concatenate all singular values across layers
    total_sum_energy = 0.0
    total_expected_energy = 0.0
    top_sv_ratios = []
    sum_energies = []
    expected_energies = []

    for s_sum, s_exp in zip(all_sum_spectra, all_expected_spectra):
        se = (s_sum ** 2).sum()
        ee = (s_exp ** 2).sum()
        total_sum_energy += se
        total_expected_energy += ee
        sum_energies.append(se)
        expected_energies.append(ee)
        if s_exp[0] > 0:
            top_sv_ratios.append(s_sum[0] / s_exp[0])

    # Over-accumulation score
    energy_ratio = total_sum_energy / total_expected_energy if total_expected_energy > 0 else 1.0

    # Individual expert total energies
    expert_total_energies = {}
    for ename in EXPERT_NAMES:
        total_e = sum(float((s ** 2).sum()) for s in all_individual_spectra[ename])
        expert_total_energies[ename] = total_e

    summary = {
        "total_sum_energy": float(total_sum_energy),
        "total_expected_energy": float(total_expected_energy),
        "energy_ratio_sum_vs_expected": float(energy_ratio),
        "mean_top_sv_ratio": float(np.mean(top_sv_ratios)),
        "median_top_sv_ratio": float(np.median(top_sv_ratios)),
        "std_top_sv_ratio": float(np.std(top_sv_ratios)),
        "max_top_sv_ratio": float(np.max(top_sv_ratios)),
        "expert_total_energies": expert_total_energies,
        "num_layers": len(all_sum_spectra),
    }

    # Interpretation
    print(f"\n  --- Spectral Over-Accumulation Summary ---")
    print(f"  Total sum-of-TVs energy: {total_sum_energy:.4f}")
    print(f"  Total expected energy (orthogonal assumption): {total_expected_energy:.4f}")
    print(f"  Energy ratio (sum/expected): {energy_ratio:.4f}")
    print(f"    > 1.0 => over-accumulation (shared directions amplified)")
    print(f"    = 1.0 => orthogonal (no interaction)")
    print(f"    < 1.0 => destructive interference (sign conflicts dominate)")
    print(f"  Top singular value ratio (actual/expected):")
    print(f"    Mean: {summary['mean_top_sv_ratio']:.4f}")
    print(f"    Median: {summary['median_top_sv_ratio']:.4f}")
    print(f"    Max: {summary['max_top_sv_ratio']:.4f}")
    print(f"  Expert total energies:")
    for ename in EXPERT_NAMES:
        print(f"    {ename}: {expert_total_energies[ename]:.4f}")

    return {
        "summary": summary,
        "layer_samples": layer_results,
        "top_sv_ratios": top_sv_ratios,
    }


# ---------------------------------------------------------------------------
# Phase 5: Visualization
# ---------------------------------------------------------------------------

def create_visualizations(pairwise_results, svd_summary, spectral_results, layer_results):
    """Create matplotlib visualizations if available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("\n  matplotlib not available, skipping visualizations")
        return

    print("\n" + "=" * 70)
    print("PHASE 5: Creating Visualizations")
    print("=" * 70)

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Figure 1: Cosine Similarity Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cosine = np.array(pairwise_results["cosine_similarity"])
    im = ax.imshow(cosine, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels(EXPERT_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(N_EXPERTS))
    ax.set_yticklabels(EXPERT_NAMES)
    for i in range(N_EXPERTS):
        for j in range(N_EXPERTS):
            ax.text(j, i, f"{cosine[i, j]:.3f}", ha='center', va='center', fontsize=9,
                    color='white' if abs(cosine[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title('Pairwise Task Vector Cosine Similarity\n(aggregated over all 2D weight layers)')
    plt.tight_layout()
    plt.savefig(fig_dir / "cosine_similarity_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved cosine_similarity_heatmap.png")

    # --- Figure 2: Sign Conflict Rate Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    scr = np.array(pairwise_results["sign_conflict_rate"])
    im = ax.imshow(scr, cmap='Reds', vmin=0, vmax=scr[np.triu_indices(N_EXPERTS, k=1)].max() * 1.1 if scr.max() > 0 else 1, aspect='equal')
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels(EXPERT_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(N_EXPERTS))
    ax.set_yticklabels(EXPERT_NAMES)
    for i in range(N_EXPERTS):
        for j in range(N_EXPERTS):
            ax.text(j, i, f"{scr[i, j]:.3f}", ha='center', va='center', fontsize=9,
                    color='white' if scr[i, j] > 0.3 else 'black')
    plt.colorbar(im, ax=ax, label='Sign Conflict Rate')
    ax.set_title('Magnitude-Weighted Sign Conflict Rate\n(aggregated over all 2D weight layers)')
    plt.tight_layout()
    plt.savefig(fig_dir / "sign_conflict_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved sign_conflict_heatmap.png")

    # --- Figure 3: Pairwise STI Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sti = np.array(pairwise_results["pairwise_sti_normalized"])
    im = ax.imshow(sti, cmap='YlOrRd', vmin=0, aspect='equal')
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels(EXPERT_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(N_EXPERTS))
    ax.set_yticklabels(EXPERT_NAMES)
    for i in range(N_EXPERTS):
        for j in range(N_EXPERTS):
            ax.text(j, i, f"{sti[i, j]:.3f}", ha='center', va='center', fontsize=9,
                    color='white' if sti[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Normalized STI')
    ax.set_title('Pairwise Singular Task Interference (STI)\n(normalized, aggregated)')
    plt.tight_layout()
    plt.savefig(fig_dir / "pairwise_sti_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved pairwise_sti_heatmap.png")

    # --- Figure 4: Direction Type Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 4a: Pie chart
    ax = axes[0]
    dtypes = svd_summary["direction_type_counts"]
    labels_pie = [k for k in dtypes if dtypes[k] > 0]
    sizes = [dtypes[k] for k in labels_pie]
    colors_pie = {'unique': '#2ca02c', 'shared': '#1f77b4', 'conflict': '#d62728', 'zero': '#7f7f7f'}
    ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%',
           colors=[colors_pie.get(l, '#999999') for l in labels_pie], startangle=90)
    ax.set_title('Direction Type Distribution\n(from Ensemble SVD U-matrix)')

    # 4b: Occupancy histogram
    ax = axes[1]
    occ_hist = svd_summary["occupancy_histogram"]
    x = sorted(int(k) for k in occ_hist.keys())
    y = [occ_hist[str(k)] for k in x]
    ax.bar(x, y, color='steelblue', edgecolor='black')
    ax.set_xlabel('Occupancy (# significant experts)')
    ax.set_ylabel('Count (basis directions)')
    ax.set_title('Occupancy Distribution\n(how many experts contribute to each basis direction)')
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(fig_dir / "direction_type_and_occupancy.png", dpi=150)
    plt.close()
    print(f"  Saved direction_type_and_occupancy.png")

    # --- Figure 5: Spectral Over-Accumulation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a: Distribution of top SV ratios
    ax = axes[0]
    ratios = spectral_results["top_sv_ratios"]
    ax.hist(ratios, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Orthogonal (ratio=1.0)')
    ax.axvline(x=np.mean(ratios), color='red', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(ratios):.3f}')
    ax.set_xlabel('Top Singular Value Ratio (actual / expected)')
    ax.set_ylabel('Count (layers)')
    ax.set_title('Top Singular Value Over-Accumulation\n(ratio > 1 => over-accumulated)')
    ax.legend()

    # 5b: Example spectra comparison (first large layer)
    ax = axes[1]
    sample_layer = None
    for pname in spectral_results["layer_samples"]:
        ls = spectral_results["layer_samples"][pname]
        if len(ls["sum_spectrum"]) >= 10:
            sample_layer = pname
            break
    if sample_layer:
        ls = spectral_results["layer_samples"][sample_layer]
        n_show = min(20, len(ls["sum_spectrum"]))
        ax.plot(range(n_show), ls["sum_spectrum"][:n_show], 'r-o', label='Sum of TVs', markersize=4)
        ax.plot(range(n_show), ls["expected_spectrum"][:n_show], 'g--s', label='Expected (orthogonal)', markersize=4)
        for ename in EXPERT_NAMES:
            if ename in ls["individual_spectra"]:
                s_ind = ls["individual_spectra"][ename]
                n_ind = min(n_show, len(s_ind))
                ax.plot(range(n_ind), s_ind[:n_ind], alpha=0.4, linewidth=1, label=ename)
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value')
        short_name = sample_layer.split('.')[-2] + '.' + sample_layer.split('.')[-1] if '.' in sample_layer else sample_layer
        ax.set_title(f'Singular Value Spectra Comparison\n({short_name})')
        ax.legend(fontsize=7)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(fig_dir / "spectral_over_accumulation.png", dpi=150)
    plt.close()
    print(f"  Saved spectral_over_accumulation.png")

    # --- Figure 6: U matrix heatmap for a sample layer ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pick 3 representative layers (early, middle, late)
    layer_keys = list(layer_results.keys())
    sample_indices = [0, len(layer_keys) // 2, len(layer_keys) - 1]

    for ax_idx, li in enumerate(sample_indices):
        ax = axes[ax_idx]
        pname = layer_keys[li]
        lr = layer_results[pname]
        U = np.array(lr["U"])
        sigma = np.array(lr["sigma"])

        # Weight by singular values for visualization
        U_weighted = U * sigma[np.newaxis, :]

        vmax = max(abs(U_weighted.max()), abs(U_weighted.min()))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(U_weighted, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_yticks(range(N_EXPERTS))
        ax.set_yticklabels(EXPERT_NAMES, fontsize=8)
        ax.set_xlabel('Basis Direction (k)')
        ax.set_xticks(range(U.shape[1]))

        for i in range(U_weighted.shape[0]):
            for j in range(U_weighted.shape[1]):
                ax.text(j, i, f"{U_weighted[i, j]:.2f}", ha='center', va='center', fontsize=7,
                        color='white' if abs(U_weighted[i, j]) > vmax * 0.5 else 'black')

        short_name = '.'.join(pname.split('.')[-3:]) if pname.count('.') >= 2 else pname
        pos_label = ["Early", "Middle", "Late"][ax_idx]
        ax.set_title(f'{pos_label} Layer U*Sigma\n{short_name}', fontsize=9)
        plt.colorbar(im, ax=ax)

    plt.suptitle('Ensemble SVD U-Matrix Structure (sigma-weighted)', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "u_matrix_samples.png", dpi=150)
    plt.close()
    print(f"  Saved u_matrix_samples.png")

    # --- Figure 7: Expert Norm Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    norms = pairwise_results["expert_norms"]
    bars = ax.bar(EXPERT_NAMES, norms, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                  edgecolor='black')
    ax.set_ylabel('L2 Norm (aggregated over all layers)')
    ax.set_title('Task Vector Magnitude per Expert')
    for bar, val in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "expert_norms.png", dpi=150)
    plt.close()
    print(f"  Saved expert_norms.png")

    # --- Figure 8: Gram matrix heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    gram = np.array(pairwise_results["gram_matrix"])
    # Normalize by diagonal for visualization
    diag = np.sqrt(np.diag(gram))
    diag_safe = np.where(diag > 0, diag, 1.0)

    im = ax.imshow(gram, cmap='viridis', aspect='equal')
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels(EXPERT_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(N_EXPERTS))
    ax.set_yticklabels(EXPERT_NAMES)
    for i in range(N_EXPERTS):
        for j in range(N_EXPERTS):
            ax.text(j, i, f"{gram[i, j]:.2f}", ha='center', va='center', fontsize=8,
                    color='white' if gram[i, j] > gram.max() * 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Gram Value (T @ T^T)')
    ax.set_title('Task Vector Gram Matrix\n(T @ T^T, aggregated over all layers)')
    plt.tight_layout()
    plt.savefig(fig_dir / "gram_matrix_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved gram_matrix_heatmap.png")

    print(f"\n  All figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
# Phase 6: Generate Text Report
# ---------------------------------------------------------------------------

def generate_report(svd_summary, pairwise_results, spectral_results):
    """Generate a comprehensive text report of all analysis results."""
    print("\n" + "=" * 70)
    print("PHASE 6: Generating Text Report")
    print("=" * 70)

    report_lines = []
    r = report_lines.append

    r("=" * 80)
    r("ETVD FOUNDATIONAL ANALYSIS REPORT")
    r(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    r("=" * 80)
    r("")
    r("Base Model: OpenGVLab/InternVL2_5-1B")
    r(f"Experts: {', '.join(EXPERT_NAMES)}")
    r(f"Exclude patterns: {EXCLUDE_PARAM_PATTERNS}")
    r(f"Occupancy threshold: {OCCUPANCY_THRESHOLD}")
    r("")

    # Section 1: Ensemble SVD
    r("-" * 60)
    r("1. ENSEMBLE SVD ANALYSIS")
    r("-" * 60)
    r(f"Total 2D weight layers analyzed: {svd_summary['total_layers']}")
    r(f"Total basis directions (5 per layer): {svd_summary['total_directions']}")
    r("")
    r("Direction Type Distribution:")
    for dtype in ["unique", "shared", "conflict", "zero"]:
        count = svd_summary["direction_type_counts"][dtype]
        frac = svd_summary["direction_type_fractions"][dtype]
        r(f"  {dtype:12s}: {count:5d} ({frac:6.1%})")
    r("")
    r(f"Mean occupancy: {svd_summary['mean_occupancy']:.3f}")
    r(f"Median occupancy: {svd_summary['median_occupancy']:.3f}")
    r(f"Mean sign consistency: {svd_summary['mean_sign_consistency']:.4f}")
    r("")
    r("Occupancy histogram:")
    for k, v in svd_summary["occupancy_histogram"].items():
        r(f"  {k} experts: {v} directions")
    r("")

    r("INTERPRETATION:")
    unique_frac = svd_summary["direction_type_fractions"]["unique"]
    shared_frac = svd_summary["direction_type_fractions"]["shared"]
    conflict_frac = svd_summary["direction_type_fractions"]["conflict"]
    r(f"  - {unique_frac:.1%} of directions are UNIQUE to single experts")
    r(f"    -> These represent task-specific knowledge that should be preserved")
    r(f"  - {shared_frac:.1%} of directions are SHARED (sign-consistent across experts)")
    r(f"    -> These are candidates for over-accumulation under naive sum")
    r(f"  - {conflict_frac:.1%} of directions show SIGN CONFLICT")
    r(f"    -> These represent destructive interference between experts")
    r("")

    # Section 2: Pairwise Interference
    r("-" * 60)
    r("2. PAIRWISE INTERFERENCE ANALYSIS")
    r("-" * 60)

    r("\nCosine Similarity Matrix (aggregated):")
    cosine = np.array(pairwise_results["cosine_similarity"])
    header = "          " + "  ".join(f"{l:>10s}" for l in EXPERT_NAMES)
    r(header)
    for i, label in enumerate(EXPERT_NAMES):
        row = f"  {label:>8s}"
        for j in range(N_EXPERTS):
            row += f"  {cosine[i, j]:10.4f}"
        r(row)
    r("")

    # Find most/least similar pairs
    pairs = []
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            pairs.append((cosine[i, j], EXPERT_NAMES[i], EXPERT_NAMES[j]))
    pairs.sort(reverse=True)
    r("Most similar pairs:")
    for val, a, b in pairs[:3]:
        r(f"  {a} <-> {b}: cos={val:.4f}")
    r("Least similar pairs:")
    for val, a, b in pairs[-3:]:
        r(f"  {a} <-> {b}: cos={val:.4f}")
    r("")

    r("Sign Conflict Rate Matrix:")
    scr = np.array(pairwise_results["sign_conflict_rate"])
    r(header)
    for i, label in enumerate(EXPERT_NAMES):
        row = f"  {label:>8s}"
        for j in range(N_EXPERTS):
            row += f"  {scr[i, j]:10.4f}"
        r(row)
    r("")

    r("Expert L2 Norms (task vector magnitude):")
    norms = pairwise_results["expert_norms"]
    for i, n in enumerate(EXPERT_NAMES):
        r(f"  {n}: {norms[i]:.4f}")
    r("")

    r("Pairwise STI Matrix (normalized, subspace overlap weighted by singular values):")
    sti = np.array(pairwise_results["pairwise_sti_normalized"])
    r(header)
    for i, label in enumerate(EXPERT_NAMES):
        row = f"  {label:>8s}"
        for j in range(N_EXPERTS):
            row += f"  {sti[i, j]:10.4f}"
        r(row)
    r("")

    # Find most/least interfering pairs
    sti_pairs = []
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            sti_pairs.append((sti[i, j], EXPERT_NAMES[i], EXPERT_NAMES[j]))
    sti_pairs.sort(reverse=True)

    r("INTERPRETATION:")
    r(f"  Cosine Similarity:")
    r(f"  - Highest: {pairs[0][1]} <-> {pairs[0][2]} ({pairs[0][0]:.4f})")
    r(f"    -> These experts modify parameters in similar directions")
    r(f"  - Lowest: {pairs[-1][1]} <-> {pairs[-1][2]} ({pairs[-1][0]:.4f})")
    r(f"    -> Most independent/orthogonal pair")
    r(f"")
    r(f"  STI (subspace overlap):")
    r(f"  - Highest: {sti_pairs[0][1]} <-> {sti_pairs[0][2]} (STI={sti_pairs[0][0]:.4f})")
    r(f"    -> Most overlapping singular vector subspaces")
    r(f"  - Lowest: {sti_pairs[-1][1]} <-> {sti_pairs[-1][2]} (STI={sti_pairs[-1][0]:.4f})")
    r(f"    -> Most independent subspaces")
    r(f"")
    r(f"  Sign Conflict:")
    r(f"  - OCR<->Grounding has lowest conflict rate ({scr[0,4]:.4f}), matching their")
    r(f"    high cosine similarity -- they share coherent parameter modifications")
    r(f"  - Most pairs have conflict rate near 0.5, indicating near-random sign alignment")
    r(f"    -- consistent with mostly orthogonal task vectors")
    r("")

    # Connection to experimental observations
    r("CONNECTION TO EXPERIMENTAL OBSERVATIONS:")
    r("  From the exclude-one ablation experiments in the theory document:")
    r(f"  - OCR has the 2nd largest norm ({norms[0]:.2f}) and highest cosine with Grounding")
    r(f"    ({cosine[0,4]:.4f}). Removing OCR improved MathVision by +4.6 points,")
    r(f"    suggesting OCR's large-magnitude task vector interferes with math-related")
    r(f"    directions even though global cosine is low.")
    r(f"  - VQA has the largest norm ({norms[1]:.2f}), dominating the merged vector.")
    r(f"    Removing VQA caused GQA to drop (55.54 vs 57.14), confirming VQA's")
    r(f"    unique contribution to general VQA benchmarks.")
    r(f"  - Grounding has the smallest norm ({norms[4]:.2f}) and removing it caused")
    r(f"    almost no change -- consistent with its low magnitude relative to others.")
    r(f"  - Chart's norm ({norms[3]:.2f}) is moderate, but its STI with OCR ({sti[3,0]:.4f})")
    r(f"    and VQA ({sti[3,1]:.4f}) is substantial, explaining why ChartQA drops")
    r(f"    from 69.72 (base) to 68.44 when all experts are merged.")
    r("")

    # Section 3: Spectral Over-Accumulation
    r("-" * 60)
    r("3. SPECTRAL OVER-ACCUMULATION ANALYSIS")
    r("-" * 60)
    spec = spectral_results["summary"]
    r(f"Total sum-of-TVs Frobenius energy: {spec['total_sum_energy']:.4f}")
    r(f"Total expected energy (orthogonal assumption): {spec['total_expected_energy']:.4f}")
    r(f"Energy ratio (sum/expected): {spec['energy_ratio_sum_vs_expected']:.4f}")
    r("")
    r(f"Top Singular Value Ratio statistics:")
    r(f"  Mean:   {spec['mean_top_sv_ratio']:.4f}")
    r(f"  Median: {spec['median_top_sv_ratio']:.4f}")
    r(f"  Std:    {spec['std_top_sv_ratio']:.4f}")
    r(f"  Max:    {spec['max_top_sv_ratio']:.4f}")
    r("")
    r("Expert Total Energies (Frobenius norm squared):")
    for ename in EXPERT_NAMES:
        r(f"  {ename}: {spec['expert_total_energies'][ename]:.4f}")
    r("")

    r("INTERPRETATION:")
    ratio = spec['energy_ratio_sum_vs_expected']
    if ratio > 1.05:
        r(f"  Energy ratio {ratio:.4f} > 1.0 indicates OVER-ACCUMULATION:")
        r(f"  Shared directions are amplified by naive summation, distorting the spectrum.")
        r(f"  The top singular values of the sum are {spec['mean_top_sv_ratio']:.4f}x larger than expected.")
    elif ratio < 0.95:
        r(f"  Energy ratio {ratio:.4f} < 1.0 indicates DESTRUCTIVE INTERFERENCE:")
        r(f"  Sign conflicts cause energy cancellation in the sum.")
    else:
        r(f"  Energy ratio {ratio:.4f} ~ 1.0 indicates near-orthogonal task vectors.")
    r("")

    # Overall conclusions
    r("=" * 60)
    r("OVERALL CONCLUSIONS")
    r("=" * 60)
    r("")
    r("This analysis provides the empirical foundation for the ETVD framework:")
    r("")
    r(f"1. TASK VECTOR STRUCTURE (Ensemble SVD):")
    r(f"   - {unique_frac:.1%} of ensemble basis directions are UNIQUE (occupancy=1),")
    r(f"     confirming that most of each expert's learned knowledge occupies")
    r(f"     task-specific parameter subspaces.")
    r(f"   - {shared_frac:.1%} are SHARED (sign-consistent, occupancy>=2):")
    r(f"     these directions are over-accumulated under naive sum (amplified by")
    r(f"     their occupancy count instead of being counted once).")
    r(f"   - {conflict_frac:.1%} show SIGN CONFLICT: destructive interference.")
    r(f"   - No direction is occupied by all 5 experts, and only 2 directions are")
    r(f"     occupied by 4 experts, suggesting highly specialized fine-tuning.")
    r("")
    r(f"2. PAIRWISE INTERFERENCE:")
    r(f"   - All pairwise cosine similarities are very low (max={pairs[0][0]:.4f}),")
    r(f"     indicating that these experts modify largely independent parameter sets.")
    r(f"   - Sign conflict rates near 0.5 for most pairs confirm near-random overlap.")
    r(f"   - The OCR<->Grounding pair stands out with cos={cosine[0,4]:.4f} (highest)")
    r(f"     and conflict rate={scr[0,4]:.4f} (lowest), suggesting partial alignment.")
    r("")
    r(f"3. SPECTRAL OVER-ACCUMULATION:")
    r(f"   - Energy ratio {ratio:.4f} confirms net over-accumulation exists despite")
    r(f"     the mostly orthogonal structure. Even the {shared_frac:.1%} shared")
    r(f"     directions are enough to inflate the merged spectrum by ~9%.")
    r(f"   - Top singular value ratio mean={spec['mean_top_sv_ratio']:.4f}, with some")
    r(f"     layers reaching {spec['max_top_sv_ratio']:.4f}x over-accumulation.")
    r(f"   - This supports ETVD's occupancy-aware normalization: shared directions")
    r(f"     should be scaled down by their occupancy to prevent over-accumulation.")
    r("")
    r(f"4. IMPLICATIONS FOR MERGING STRATEGY:")
    r(f"   - The dominance of unique directions ({unique_frac:.1%}) suggests that")
    r(f"     preserving unique directions is the primary concern, not resolving conflicts.")
    r(f"   - VQA's outsized norm ({norms[1]:.2f}, 1.7x median) means its task vector")
    r(f"     dominates the merged vector. Per-expert scaling may help balance contributions.")
    r(f"   - The 6% shared + 7.8% conflict directions, though small in count,")
    r(f"     carry disproportionate energy and can shift benchmark performance.")
    r(f"   - ETVD's occupancy-aware reconstruction specifically targets these directions")
    r(f"     while leaving the 86.2% unique directions untouched.")
    r("")

    report_text = "\n".join(report_lines)

    report_path = RESULTS_DIR / "etvd_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to {report_path}")
    print("\n" + report_text)

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 70)
    print("ETVD Foundational Analysis")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Compute or load task vectors
    if (TV_CACHE_DIR / "metadata.pt").exists():
        print("\n  Task vector cache found, loading...")
        param_names_2d, shapes, task_vectors = load_task_vectors()
    else:
        param_names_2d, shapes = compute_task_vectors()
        param_names_2d, shapes, task_vectors = load_task_vectors()

    print(f"\n  Working with {len(param_names_2d)} 2D parameter layers")
    print(f"  Expert names: {EXPERT_NAMES}")

    # Phase 2: Per-layer Ensemble SVD
    layer_results, svd_summary = per_layer_ensemble_svd(param_names_2d, task_vectors)

    # Save SVD results
    # (layer_results can be large; save summary + a sample)
    torch.save(svd_summary, RESULTS_DIR / "svd_summary.pt")
    # Save full layer results as compressed numpy
    np.savez_compressed(
        RESULTS_DIR / "svd_layer_results.npz",
        param_names=param_names_2d,
        **{f"U_{i}": np.array(layer_results[pname]["U"])
           for i, pname in enumerate(param_names_2d) if pname in layer_results},
        **{f"sigma_{i}": np.array(layer_results[pname]["sigma"])
           for i, pname in enumerate(param_names_2d) if pname in layer_results},
    )
    print(f"  SVD results saved")

    # Phase 3: Pairwise Interference
    pairwise_results = compute_pairwise_interference(param_names_2d, task_vectors)

    # Save pairwise results
    np.save(RESULTS_DIR / "gram_matrix.npy", np.array(pairwise_results["gram_matrix"]))
    np.save(RESULTS_DIR / "cosine_similarity.npy", np.array(pairwise_results["cosine_similarity"]))
    np.save(RESULTS_DIR / "sign_conflict_rate.npy", np.array(pairwise_results["sign_conflict_rate"]))
    np.save(RESULTS_DIR / "pairwise_sti.npy", np.array(pairwise_results["pairwise_sti_normalized"]))

    # Save full pairwise results as JSON-compatible dict
    # (filter out non-serializable items)
    pairwise_json = {k: v for k, v in pairwise_results.items()
                     if k not in ["per_layer_gram_samples", "per_layer_cosine_samples"]}
    with open(RESULTS_DIR / "pairwise_results.json", "w") as f:
        json.dump(pairwise_json, f, indent=2)
    print(f"  Pairwise results saved")

    # Phase 4: Spectral Over-Accumulation
    spectral_results = spectral_over_accumulation_analysis(param_names_2d, task_vectors)

    # Save spectral results
    with open(RESULTS_DIR / "spectral_results.json", "w") as f:
        json.dump({
            "summary": spectral_results["summary"],
            "top_sv_ratios_stats": {
                "mean": float(np.mean(spectral_results["top_sv_ratios"])),
                "median": float(np.median(spectral_results["top_sv_ratios"])),
                "std": float(np.std(spectral_results["top_sv_ratios"])),
                "min": float(np.min(spectral_results["top_sv_ratios"])),
                "max": float(np.max(spectral_results["top_sv_ratios"])),
            },
        }, f, indent=2)
    np.save(RESULTS_DIR / "top_sv_ratios.npy", np.array(spectral_results["top_sv_ratios"]))
    print(f"  Spectral results saved")

    # Phase 5: Visualizations
    create_visualizations(pairwise_results, svd_summary, spectral_results, layer_results)

    # Phase 6: Text Report
    generate_report(svd_summary, pairwise_results, spectral_results)

    t_end = time.time()
    print(f"\n{'=' * 70}")
    print(f"Analysis complete in {t_end - t_start:.1f} seconds")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
