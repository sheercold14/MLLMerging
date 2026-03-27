"""
ETVD Merge: Ensemble Task Vector Decomposition Merge Method

Implements occupancy-aware spectrum reconstruction for model merging.
Instead of naive sum (Task Arithmetic), decomposes task vectors into
shared/unique/conflict directions via ensemble SVD, and normalizes
by occupancy to prevent over-accumulation while preserving unique directions.
"""

import os
import sys
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

# Add InternVL path for model loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../InternVL/internvl_chat'))

DEFAULT_EXCLUDE_PATTERNS = [
    "vision_model.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*embed_tokens.*",
    ".*bias.*",
]


def get_param_names_to_merge(input_param_names, exclude_param_names_regex):
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(re.match(p, param_name) for p in exclude_param_names_regex)
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def etvd_merge(
    base_state_dict,
    expert_state_dicts,
    expert_names,
    exclude_param_names_regex,
    scaling_coefficient=1.0,
    occupancy_threshold=0.3,
    conflict_strategy="majority",  # "majority", "suppress", "mean"
    verbose=True,
):
    """
    ETVD Merge: Ensemble Task Vector Decomposition.

    For each 2D weight layer:
    1. Compute task vectors: τ_i = expert_i - base
    2. Stack into T ∈ R^{N×d}, do SVD: T = UΣV^T
    3. Classify each basis direction by occupancy and sign consistency
    4. Reconstruct with occupancy-aware weighting

    Args:
        base_state_dict: base model state dict
        expert_state_dicts: list of expert state dicts
        expert_names: list of expert name strings
        exclude_param_names_regex: patterns to exclude
        scaling_coefficient: global scaling for merged task vector
        occupancy_threshold: threshold on |u_ik| to count as "occupied"
        conflict_strategy: how to handle conflicting directions
        verbose: print progress
    Returns:
        merged_state_dict with updated parameters
    """
    N = len(expert_state_dicts)
    param_names = get_param_names_to_merge(
        list(base_state_dict.keys()), exclude_param_names_regex
    )

    merged_state_dict = {}
    stats = {
        "unique": 0, "shared": 0, "conflict": 0, "zero": 0,
        "total_directions": 0, "layers_2d": 0, "layers_other": 0,
    }

    for param_name in tqdm(param_names, desc="ETVD Merge", disable=not verbose):
        base_param = base_state_dict[param_name].float()
        task_vectors = []
        for sd in expert_state_dicts:
            tv = sd[param_name].float() - base_param
            task_vectors.append(tv)

        shape = base_param.shape

        if len(shape) == 2:
            stats["layers_2d"] += 1
            # Flatten each task vector for this layer
            tvs_flat = torch.stack([tv.flatten() for tv in task_vectors])  # (N, d)
            T = tvs_flat.numpy()

            # Ensemble SVD
            U, S, Vt = np.linalg.svd(T, full_matrices=False)  # U:(N,r), S:(r,), Vt:(r,d)
            r = len(S)

            # Classify and weight each direction
            weighted_coeffs = np.zeros(r)
            for k in range(r):
                u_col = U[:, k]
                abs_u = np.abs(u_col)
                occupancy = int(np.sum(abs_u > occupancy_threshold))
                stats["total_directions"] += 1

                if occupancy == 0:
                    # Zero direction (shouldn't happen with proper threshold)
                    stats["zero"] += 1
                    weighted_coeffs[k] = 0.0
                elif occupancy == 1:
                    # Unique direction: keep at full strength
                    stats["unique"] += 1
                    weighted_coeffs[k] = np.sum(u_col)  # = the single dominant value
                else:
                    # Multi-expert direction: check sign consistency
                    significant = abs_u > occupancy_threshold
                    signs = np.sign(u_col[significant])
                    sign_sum = np.sum(signs)

                    if np.abs(sign_sum) == occupancy:
                        # All agree on sign: shared beneficial direction
                        stats["shared"] += 1
                        # Normalize by occupancy to prevent over-accumulation
                        weighted_coeffs[k] = np.sum(u_col) / occupancy
                    else:
                        # Sign conflict
                        stats["conflict"] += 1
                        if conflict_strategy == "majority":
                            # Keep majority sign, scale by agreement ratio
                            majority_sign = np.sign(sign_sum) if sign_sum != 0 else 1.0
                            agreement = np.abs(sign_sum) / occupancy
                            weighted_coeffs[k] = majority_sign * np.mean(abs_u[significant]) * agreement
                        elif conflict_strategy == "suppress":
                            weighted_coeffs[k] = 0.0
                        elif conflict_strategy == "mean":
                            weighted_coeffs[k] = np.mean(u_col[significant])

            # Reconstruct merged task vector: Σ_k w_k * σ_k * v_k
            merged_tv_flat = np.zeros(T.shape[1])
            for k in range(r):
                merged_tv_flat += weighted_coeffs[k] * S[k] * Vt[k]

            merged_tv = torch.from_numpy(merged_tv_flat).reshape(shape).to(base_param.dtype)
            merged_state_dict[param_name] = base_param + scaling_coefficient * merged_tv

        else:
            # Non-2D: simple average of task vectors
            stats["layers_other"] += 1
            avg_tv = task_vectors[0].clone()
            for i in range(1, N):
                avg_tv += task_vectors[i]
            avg_tv /= N
            merged_state_dict[param_name] = base_param + scaling_coefficient * avg_tv

    if verbose:
        print(f"\n--- ETVD Merge Stats ---")
        print(f"  2D layers: {stats['layers_2d']}, other: {stats['layers_other']}")
        print(f"  Total directions: {stats['total_directions']}")
        print(f"    Unique:   {stats['unique']} ({stats['unique']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Shared:   {stats['shared']} ({stats['shared']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Conflict: {stats['conflict']} ({stats['conflict']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Zero:     {stats['zero']}")
        print(f"  Occupancy threshold: {occupancy_threshold}")
        print(f"  Conflict strategy: {conflict_strategy}")
        print(f"  Scaling coefficient: {scaling_coefficient}")

    return merged_state_dict


def etvd_v2_merge(
    base_state_dict,
    expert_state_dicts,
    expert_names,
    exclude_param_names_regex,
    scaling_coefficient=1.0,
    occupancy_threshold=0.3,
    conflict_strategy="majority",
    verbose=True,
):
    """
    ETVD-v2: Conflict-Only Intervention.

    Key change from ETVD-v1: shared directions (multi-expert, same sign) are kept
    at FULL sum (same as TA), not divided by occupancy. Only conflict directions
    receive intervention. This preserves beneficial universal improvements while
    resolving destructive interference.

    Rationale: ETVD-v1 suppresses shared directions by dividing by occupancy,
    but shared directions represent improvements all experts agree on. Suppressing
    them removes beneficial knowledge. Only conflict directions (sign disagreement)
    need intervention.
    """
    N = len(expert_state_dicts)
    param_names = get_param_names_to_merge(
        list(base_state_dict.keys()), exclude_param_names_regex
    )

    merged_state_dict = {}
    stats = {
        "unique": 0, "shared": 0, "conflict": 0, "zero": 0,
        "total_directions": 0, "layers_2d": 0, "layers_other": 0,
    }

    for param_name in tqdm(param_names, desc="ETVD-v2 Merge", disable=not verbose):
        base_param = base_state_dict[param_name].float()
        task_vectors = []
        for sd in expert_state_dicts:
            tv = sd[param_name].float() - base_param
            task_vectors.append(tv)

        shape = base_param.shape

        if len(shape) == 2:
            stats["layers_2d"] += 1
            tvs_flat = torch.stack([tv.flatten() for tv in task_vectors])  # (N, d)
            T = tvs_flat.numpy()

            U, S, Vt = np.linalg.svd(T, full_matrices=False)
            r = len(S)

            weighted_coeffs = np.zeros(r)
            for k in range(r):
                u_col = U[:, k]
                abs_u = np.abs(u_col)
                occupancy = int(np.sum(abs_u > occupancy_threshold))
                stats["total_directions"] += 1

                ta_coeff = np.sum(u_col)  # TA's coefficient for this direction

                if occupancy == 0:
                    stats["zero"] += 1
                    weighted_coeffs[k] = ta_coeff  # keep TA behavior
                elif occupancy == 1:
                    # Unique: keep full sum (= TA behavior)
                    stats["unique"] += 1
                    weighted_coeffs[k] = ta_coeff
                else:
                    significant = abs_u > occupancy_threshold
                    signs = np.sign(u_col[significant])
                    sign_sum = np.sum(signs)

                    if np.abs(sign_sum) == occupancy:
                        # Shared (sign-consistent): keep full sum (= TA behavior)
                        stats["shared"] += 1
                        weighted_coeffs[k] = ta_coeff
                    else:
                        # Conflict: intervene
                        stats["conflict"] += 1
                        if conflict_strategy == "majority":
                            majority_sign = np.sign(sign_sum) if sign_sum != 0 else 1.0
                            agreement = np.abs(sign_sum) / occupancy
                            weighted_coeffs[k] = majority_sign * np.mean(abs_u[significant]) * agreement
                        elif conflict_strategy == "suppress":
                            weighted_coeffs[k] = 0.0
                        elif conflict_strategy == "mean":
                            weighted_coeffs[k] = np.mean(u_col[significant])

            merged_tv_flat = np.zeros(T.shape[1])
            for k in range(r):
                merged_tv_flat += weighted_coeffs[k] * S[k] * Vt[k]

            merged_tv = torch.from_numpy(merged_tv_flat).reshape(shape).to(base_param.dtype)
            merged_state_dict[param_name] = base_param + scaling_coefficient * merged_tv
        else:
            # Non-2D: full sum (same as TA)
            stats["layers_other"] += 1
            tv_sum = task_vectors[0].clone()
            for i in range(1, N):
                tv_sum += task_vectors[i]
            merged_state_dict[param_name] = base_param + scaling_coefficient * tv_sum

    if verbose:
        print(f"\n--- ETVD-v2 Merge Stats ---")
        print(f"  2D layers: {stats['layers_2d']}, other: {stats['layers_other']}")
        print(f"  Total directions: {stats['total_directions']}")
        print(f"    Unique:   {stats['unique']} ({stats['unique']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Shared:   {stats['shared']} ({stats['shared']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Conflict: {stats['conflict']} ({stats['conflict']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Zero:     {stats['zero']}")
        print(f"  Occupancy threshold: {occupancy_threshold}")
        print(f"  Conflict strategy: {conflict_strategy}")
        print(f"  Scaling coefficient: {scaling_coefficient}")

    return merged_state_dict


def etvd_norm_merge(
    base_state_dict,
    expert_state_dicts,
    expert_names,
    exclude_param_names_regex,
    scaling_coefficient=1.0,
    occupancy_threshold=0.3,
    conflict_strategy="majority",
    norm_target="mean",  # "mean", "median", "max"
    verbose=True,
):
    """
    ETVD-Norm: ETVD with per-layer norm matching.

    Instead of a global scaling coefficient, this variant normalizes the merged
    task vector per-layer so its Frobenius norm matches a target derived from
    the individual expert task vectors. This provides principled, adaptive
    magnitude control per layer.

    norm_target: how to compute target norm from individual expert norms
        "mean" - match average expert TV norm (conservative, balanced)
        "median" - match median expert TV norm
        "max" - match maximum expert TV norm (aggressive)
    """
    N = len(expert_state_dicts)
    param_names = get_param_names_to_merge(
        list(base_state_dict.keys()), exclude_param_names_regex
    )

    merged_state_dict = {}
    stats = {
        "unique": 0, "shared": 0, "conflict": 0, "zero": 0,
        "total_directions": 0, "layers_2d": 0, "layers_other": 0,
        "norm_ratios": [],
    }

    for param_name in tqdm(param_names, desc="ETVD-Norm Merge", disable=not verbose):
        base_param = base_state_dict[param_name].float()
        task_vectors = []
        for sd in expert_state_dicts:
            tv = sd[param_name].float() - base_param
            task_vectors.append(tv)

        shape = base_param.shape

        # Compute individual expert TV norms for this layer
        tv_norms = [tv.norm().item() for tv in task_vectors]
        if norm_target == "mean":
            target_norm = np.mean(tv_norms)
        elif norm_target == "median":
            target_norm = np.median(tv_norms)
        elif norm_target == "max":
            target_norm = np.max(tv_norms)

        if len(shape) == 2:
            stats["layers_2d"] += 1
            tvs_flat = torch.stack([tv.flatten() for tv in task_vectors])  # (N, d)
            T = tvs_flat.numpy()

            U, S, Vt = np.linalg.svd(T, full_matrices=False)
            r = len(S)

            weighted_coeffs = np.zeros(r)
            for k in range(r):
                u_col = U[:, k]
                abs_u = np.abs(u_col)
                occupancy = int(np.sum(abs_u > occupancy_threshold))
                stats["total_directions"] += 1

                if occupancy == 0:
                    stats["zero"] += 1
                    weighted_coeffs[k] = 0.0
                elif occupancy == 1:
                    stats["unique"] += 1
                    weighted_coeffs[k] = np.sum(u_col)
                else:
                    significant = abs_u > occupancy_threshold
                    signs = np.sign(u_col[significant])
                    sign_sum = np.sum(signs)

                    if np.abs(sign_sum) == occupancy:
                        stats["shared"] += 1
                        weighted_coeffs[k] = np.sum(u_col) / occupancy
                    else:
                        stats["conflict"] += 1
                        if conflict_strategy == "majority":
                            majority_sign = np.sign(sign_sum) if sign_sum != 0 else 1.0
                            agreement = np.abs(sign_sum) / occupancy
                            weighted_coeffs[k] = majority_sign * np.mean(abs_u[significant]) * agreement
                        elif conflict_strategy == "suppress":
                            weighted_coeffs[k] = 0.0
                        elif conflict_strategy == "mean":
                            weighted_coeffs[k] = np.mean(u_col[significant])

            merged_tv_flat = np.zeros(T.shape[1])
            for k in range(r):
                merged_tv_flat += weighted_coeffs[k] * S[k] * Vt[k]

            merged_tv = torch.from_numpy(merged_tv_flat).reshape(shape)

            # Per-layer norm matching
            current_norm = merged_tv.norm().item()
            if current_norm > 1e-10 and target_norm > 1e-10:
                norm_ratio = target_norm / current_norm
                stats["norm_ratios"].append(norm_ratio)
                merged_tv = merged_tv * norm_ratio

            merged_state_dict[param_name] = (base_param + scaling_coefficient * merged_tv).to(base_param.dtype)

        else:
            stats["layers_other"] += 1
            avg_tv = task_vectors[0].clone()
            for i in range(1, N):
                avg_tv += task_vectors[i]
            avg_tv /= N

            # Norm match for non-2D too
            current_norm = avg_tv.norm().item()
            if current_norm > 1e-10 and target_norm > 1e-10:
                norm_ratio = target_norm / current_norm
                avg_tv = avg_tv * norm_ratio

            merged_state_dict[param_name] = (base_param + scaling_coefficient * avg_tv).to(base_param.dtype)

    if verbose:
        ratios = stats["norm_ratios"]
        print(f"\n--- ETVD-Norm Merge Stats ---")
        print(f"  2D layers: {stats['layers_2d']}, other: {stats['layers_other']}")
        print(f"  Total directions: {stats['total_directions']}")
        print(f"    Unique:   {stats['unique']} ({stats['unique']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Shared:   {stats['shared']} ({stats['shared']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"    Conflict: {stats['conflict']} ({stats['conflict']/max(stats['total_directions'],1)*100:.1f}%)")
        print(f"  Norm target: {norm_target}")
        if ratios:
            print(f"  Norm ratios: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}, "
                  f"min={np.min(ratios):.4f}, max={np.max(ratios):.4f}")
        print(f"  Scaling coefficient: {scaling_coefficient}")

    return merged_state_dict


def task_arithmetic_merge(base_state_dict, expert_state_dicts, exclude_param_names_regex, scaling_coefficient=1.0):
    """Baseline: Task Arithmetic (simple sum of task vectors)."""
    N = len(expert_state_dicts)
    param_names = get_param_names_to_merge(list(base_state_dict.keys()), exclude_param_names_regex)
    merged = {}
    for param_name in tqdm(param_names, desc="Task Arithmetic"):
        base_param = base_state_dict[param_name].float()
        tv_sum = torch.zeros_like(base_param)
        for sd in expert_state_dicts:
            tv_sum += sd[param_name].float() - base_param
        merged[param_name] = base_param + scaling_coefficient * tv_sum
    return merged


def ta_norm_merge(base_state_dict, expert_state_dicts, exclude_param_names_regex,
                  scaling_coefficient=1.0, norm_target="mean", verbose=True):
    """
    TA with per-layer norm matching (ablation: no ETVD decomposition).

    Sums task vectors like standard TA, but normalizes the merged task vector
    per-layer to match the target norm of individual expert task vectors.
    This isolates the effect of per-layer norm matching from ETVD's occupancy
    decomposition.
    """
    N = len(expert_state_dicts)
    param_names = get_param_names_to_merge(list(base_state_dict.keys()), exclude_param_names_regex)
    merged = {}
    norm_ratios = []

    for param_name in tqdm(param_names, desc="TA-Norm", disable=not verbose):
        base_param = base_state_dict[param_name].float()
        task_vectors = []
        tv_sum = torch.zeros_like(base_param)
        for sd in expert_state_dicts:
            tv = sd[param_name].float() - base_param
            task_vectors.append(tv)
            tv_sum += tv

        # Compute target norm from individual TVs
        tv_norms = [tv.norm().item() for tv in task_vectors]
        if norm_target == "mean":
            target = np.mean(tv_norms)
        elif norm_target == "median":
            target = np.median(tv_norms)
        elif norm_target == "max":
            target = np.max(tv_norms)

        # Normalize merged TV to match target
        current_norm = tv_sum.norm().item()
        if current_norm > 1e-10 and target > 1e-10:
            ratio = target / current_norm
            norm_ratios.append(ratio)
            tv_sum = tv_sum * ratio

        merged[param_name] = (base_param + scaling_coefficient * tv_sum).to(base_param.dtype)

    if verbose:
        print(f"\n--- TA-Norm Stats ---")
        print(f"  Layers: {len(param_names)}")
        print(f"  Norm target: {norm_target}")
        if norm_ratios:
            print(f"  Norm ratios: mean={np.mean(norm_ratios):.4f}, std={np.std(norm_ratios):.4f}, "
                  f"min={np.min(norm_ratios):.4f}, max={np.max(norm_ratios):.4f}")
        print(f"  Scaling coefficient: {scaling_coefficient}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="ETVD Merge for InternVL models")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--merge-models", nargs="+", required=True, help="Expert model paths")
    parser.add_argument("--expert-names", nargs="+", default=None, help="Expert names (for logging)")
    parser.add_argument("--output-path", required=True, help="Output directory")
    parser.add_argument("--method", default="etvd", choices=["etvd", "etvd_norm", "task_arithmetic"])
    parser.add_argument("--scaling-coefficient", type=float, default=1.0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.3)
    parser.add_argument("--conflict-strategy", default="majority", choices=["majority", "suppress", "mean"])
    parser.add_argument("--norm-target", default="mean", choices=["mean", "median", "max"])
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]
    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    from transformers import AutoModel, AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)

    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModel.from_pretrained(args.base_model, torch_dtype=torch_dtype, trust_remote_code=True).eval()
    base_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # Load expert models
    expert_sds = []
    expert_names = args.expert_names or [f"expert_{i}" for i in range(len(args.merge_models))]
    for i, path in enumerate(args.merge_models):
        print(f"Loading expert {expert_names[i]}: {path}")
        model = AutoModel.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).eval()
        expert_sds.append({k: v.cpu() for k, v in model.state_dict().items()})
        del model
        torch.cuda.empty_cache()

    # Merge
    if args.method == "etvd":
        merged_params = etvd_merge(
            base_sd, expert_sds, expert_names, exclude_patterns,
            scaling_coefficient=args.scaling_coefficient,
            occupancy_threshold=args.occupancy_threshold,
            conflict_strategy=args.conflict_strategy,
        )
    elif args.method == "etvd_norm":
        merged_params = etvd_norm_merge(
            base_sd, expert_sds, expert_names, exclude_patterns,
            scaling_coefficient=args.scaling_coefficient,
            occupancy_threshold=args.occupancy_threshold,
            conflict_strategy=args.conflict_strategy,
            norm_target=args.norm_target,
        )
    elif args.method == "task_arithmetic":
        merged_params = task_arithmetic_merge(
            base_sd, expert_sds, exclude_patterns,
            scaling_coefficient=args.scaling_coefficient,
        )

    # Apply merged params to base model
    full_sd = base_model.state_dict()
    for k, v in merged_params.items():
        if k in full_sd:
            full_sd[k] = v.to(torch_dtype)
    base_model.load_state_dict(full_sd)

    # Save
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving to {args.output_path}")
    base_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
