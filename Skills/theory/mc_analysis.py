"""
Micro-Capability Analysis & Route A (MC-wudi2) Implementation

Section 11.1 of NeurIPS2026 theory document:
1. Extract micro-capabilities via individual SVD of each expert's task vector
2. Build K×K interference matrix I_ab = (v_a·v_b)(u_a·u_b) * sigma_a * sigma_b
3. Classify: same-expert (within), positive cross-expert (redundant), negative cross-expert (conflict)
4. Report statistics per layer and aggregated
"""

import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, "/data/shichao/era-2026/MLLMerging/InternVL/internvl_chat")
from model_merging import get_param_names_to_merge

DEFAULT_EXCLUDE_PATTERNS = [
    "vision_model.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*embed_tokens.*",
    ".*bias.*",
]

EXPERT_NAMES = ["OCR", "VQA", "Geometry", "Chart", "Grounding"]
EXPERT_PATHS = [
    "yongxianwei/InternVL2_5-1B_OCR",
    "yongxianwei/InternVL2_5-1B_VQA",
    "yongxianwei/InternVL2_5-1B_Geometry",
    "yongxianwei/InternVL2_5-1B_Chart",
    "yongxianwei/InternVL2_5-1B_Grounding",
]
BASE_PATH = "OpenGVLab/InternVL2_5-1B"


def load_state_dicts():
    """Load base and expert state dicts (CPU, float32)."""
    from transformers import AutoModel

    print("Loading base model...")
    base_model = AutoModel.from_pretrained(BASE_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    base_sd = {k: v.float().cpu() for k, v in base_model.state_dict().items()}
    del base_model

    expert_sds = {}
    for name, path in zip(EXPERT_NAMES, EXPERT_PATHS):
        print(f"Loading expert: {name}...")
        model = AutoModel.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True)
        expert_sds[name] = {k: v.float().cpu() for k, v in model.state_dict().items()}
        del model

    torch.cuda.empty_cache()
    return base_sd, expert_sds


def get_2d_param_names(base_sd):
    """Get names of 2D parameters to merge (excluding vision, lm_head, norm, etc.)."""
    all_names = list(base_sd.keys())
    mergeable = get_param_names_to_merge(all_names, DEFAULT_EXCLUDE_PATTERNS)
    return [n for n in mergeable if len(base_sd[n].shape) == 2]


def extract_micro_capabilities(task_vector, top_k=None, energy_ratio=0.95):
    """
    Extract micro-capabilities from a single task vector via SVD.

    Returns list of (sigma, u_col, v_row) tuples.
    If top_k is None, use energy_ratio to determine cutoff.
    """
    U, S, Vt = torch.linalg.svd(task_vector, full_matrices=False)

    if top_k is None:
        # Determine k by energy ratio
        total_energy = torch.sum(S ** 2).item()
        cumulative = torch.cumsum(S ** 2, dim=0)
        k = torch.searchsorted(cumulative, energy_ratio * total_energy).item() + 1
        k = max(k, 1)
        k = min(k, S.shape[0])
    else:
        k = min(top_k, S.shape[0])

    caps = []
    for j in range(k):
        caps.append((S[j].item(), U[:, j], Vt[j, :]))
    return caps, k


def build_interference_matrix(all_caps, expert_indices):
    """
    Build K×K interference matrix.
    I_ab = (u_a · u_b) * (v_a · v_b) * sigma_a * sigma_b

    all_caps: list of (sigma, u, v) tuples
    expert_indices: list of expert index for each capability

    Returns: interference matrix (K×K), expert_indices
    """
    K = len(all_caps)
    device = all_caps[0][1].device
    # Stack for efficient computation
    sigmas = torch.tensor([c[0] for c in all_caps], device=device)
    U_mat = torch.stack([c[1] for c in all_caps])  # K × m
    V_mat = torch.stack([c[2] for c in all_caps])  # K × n

    # Compute overlaps
    u_overlap = U_mat @ U_mat.T  # K × K
    v_overlap = V_mat @ V_mat.T  # K × K

    # Interference: element-wise product of overlaps × sigma products
    sigma_outer = sigmas.unsqueeze(1) * sigmas.unsqueeze(0)
    I_matrix = u_overlap * v_overlap * sigma_outer

    return I_matrix, sigmas


def analyze_layer(task_vectors, expert_names, top_k=None, energy_ratio=0.95):
    """
    Analyze a single layer's micro-capability interference structure.

    task_vectors: list of N tensors, each (m, n)
    Returns: dict with analysis results
    """
    N = len(task_vectors)
    all_caps = []
    expert_indices = []
    caps_per_expert = []

    for i in range(N):
        caps, k = extract_micro_capabilities(task_vectors[i], top_k=top_k, energy_ratio=energy_ratio)
        caps_per_expert.append(k)
        for cap in caps:
            all_caps.append(cap)
            expert_indices.append(i)

    K = len(all_caps)
    expert_indices = torch.tensor(expert_indices)

    I_matrix, sigmas = build_interference_matrix(all_caps, expert_indices)

    # Classify interactions
    stats = {
        "total_caps": K,
        "caps_per_expert": caps_per_expert,
        "within_expert": {"count": 0, "pos": 0, "neg": 0, "total_mag": 0.0},
        "cross_expert": {"count": 0, "pos": 0, "neg": 0, "total_mag": 0.0},
        "cross_conflict": {"count": 0, "total_mag": 0.0},
        "cross_redundant": {"count": 0, "total_mag": 0.0},
        "pairwise_conflict": defaultdict(float),  # (i,j) -> total conflict magnitude
    }

    for a in range(K):
        for b in range(a + 1, K):
            val = I_matrix[a, b].item()
            ea, eb = expert_indices[a].item(), expert_indices[b].item()

            if ea == eb:
                stats["within_expert"]["count"] += 1
                stats["within_expert"]["total_mag"] += abs(val)
                if val > 0:
                    stats["within_expert"]["pos"] += 1
                else:
                    stats["within_expert"]["neg"] += 1
            else:
                stats["cross_expert"]["count"] += 1
                stats["cross_expert"]["total_mag"] += abs(val)
                if val > 0:
                    stats["cross_expert"]["pos"] += 1
                    stats["cross_redundant"]["count"] += 1
                    stats["cross_redundant"]["total_mag"] += val
                else:
                    stats["cross_expert"]["neg"] += 1
                    stats["cross_conflict"]["count"] += 1
                    stats["cross_conflict"]["total_mag"] += abs(val)
                    # Track pairwise
                    pair = (min(ea, eb), max(ea, eb))
                    stats["pairwise_conflict"][(pair[0], pair[1])] += abs(val)

    return stats, I_matrix, all_caps, expert_indices


def run_analysis(base_sd, expert_sds, top_k=None, energy_ratio=0.95,
                 sample_layers=None, device="cuda:1"):
    """
    Run full micro-capability analysis across all 2D layers.

    sample_layers: if int, sample that many layers; if None, do all
    """
    param_names = get_2d_param_names(base_sd)
    print(f"Total 2D layers to analyze: {len(param_names)}")

    if sample_layers is not None and sample_layers < len(param_names):
        # Sample evenly across layers
        indices = np.linspace(0, len(param_names) - 1, sample_layers, dtype=int)
        param_names = [param_names[i] for i in indices]
        print(f"Sampling {len(param_names)} layers for analysis")

    N = len(EXPERT_NAMES)

    # Aggregated statistics
    agg = {
        "total_caps": 0,
        "total_within_pairs": 0,
        "total_cross_pairs": 0,
        "total_conflict_pairs": 0,
        "total_redundant_pairs": 0,
        "conflict_ratio": 0.0,
        "pairwise_conflict_agg": defaultdict(float),
        "caps_per_expert_agg": defaultdict(list),
        "layer_stats": [],
    }

    for param_name in tqdm(param_names, desc="Analyzing layers"):
        # Compute task vectors for this layer
        task_vectors = []
        for name in EXPERT_NAMES:
            tv = (expert_sds[name][param_name] - base_sd[param_name]).to(device)
            task_vectors.append(tv)

        stats, I_matrix, all_caps, expert_indices = analyze_layer(
            task_vectors, EXPERT_NAMES, top_k=top_k, energy_ratio=energy_ratio
        )

        # Move tensors back to CPU to free GPU
        for i in range(len(all_caps)):
            sigma, u, v = all_caps[i]
            all_caps[i] = (sigma, u.cpu(), v.cpu())
        del I_matrix
        for tv in task_vectors:
            del tv

        layer_info = {
            "name": param_name,
            "shape": list(base_sd[param_name].shape),
            "total_caps": stats["total_caps"],
            "caps_per_expert": stats["caps_per_expert"],
            "cross_conflict_count": stats["cross_conflict"]["count"],
            "cross_redundant_count": stats["cross_redundant"]["count"],
            "cross_conflict_mag": stats["cross_conflict"]["total_mag"],
            "cross_redundant_mag": stats["cross_redundant"]["total_mag"],
        }
        agg["layer_stats"].append(layer_info)
        agg["total_caps"] += stats["total_caps"]
        agg["total_within_pairs"] += stats["within_expert"]["count"]
        agg["total_cross_pairs"] += stats["cross_expert"]["count"]
        agg["total_conflict_pairs"] += stats["cross_conflict"]["count"]
        agg["total_redundant_pairs"] += stats["cross_redundant"]["count"]

        for (i, j), mag in stats["pairwise_conflict"].items():
            pair_name = f"{EXPERT_NAMES[i]}-{EXPERT_NAMES[j]}"
            agg["pairwise_conflict_agg"][pair_name] += mag

        for idx, k in enumerate(stats["caps_per_expert"]):
            agg["caps_per_expert_agg"][EXPERT_NAMES[idx]].append(k)

    # Compute ratios
    total_cross = agg["total_conflict_pairs"] + agg["total_redundant_pairs"]
    agg["conflict_ratio"] = agg["total_conflict_pairs"] / max(total_cross, 1)

    return agg


def print_analysis_report(agg):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("MICRO-CAPABILITY INTERFERENCE ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nLayers analyzed: {len(agg['layer_stats'])}")
    print(f"Total micro-capabilities extracted: {agg['total_caps']}")
    print(f"Average per layer: {agg['total_caps'] / len(agg['layer_stats']):.1f}")

    print(f"\nWithin-expert pairs: {agg['total_within_pairs']}")
    print(f"Cross-expert pairs:  {agg['total_cross_pairs']}")
    print(f"  - Redundant (positive): {agg['total_redundant_pairs']} "
          f"({agg['total_redundant_pairs']/max(agg['total_cross_pairs'],1)*100:.1f}%)")
    print(f"  - Conflict (negative):  {agg['total_conflict_pairs']} "
          f"({agg['conflict_ratio']*100:.1f}%)")

    print(f"\nMicro-caps per expert (avg across layers):")
    for name in EXPERT_NAMES:
        vals = agg["caps_per_expert_agg"][name]
        print(f"  {name:12s}: {np.mean(vals):.1f} (min={np.min(vals)}, max={np.max(vals)})")

    print(f"\nPairwise conflict magnitude (sum across all layers):")
    sorted_pairs = sorted(agg["pairwise_conflict_agg"].items(), key=lambda x: -x[1])
    for pair_name, mag in sorted_pairs:
        print(f"  {pair_name:25s}: {mag:.4f}")

    # Top 5 most conflicting layers
    sorted_layers = sorted(agg["layer_stats"], key=lambda x: -x["cross_conflict_mag"])
    print(f"\nTop 10 layers by conflict magnitude:")
    for layer in sorted_layers[:10]:
        print(f"  {layer['name']:60s} shape={layer['shape']}  "
              f"caps={layer['total_caps']}  conflict_mag={layer['cross_conflict_mag']:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=None, help="Fixed top-k per expert (default: use energy ratio)")
    parser.add_argument("--energy-ratio", type=float, default=0.95, help="Energy ratio for auto k selection")
    parser.add_argument("--sample-layers", type=int, default=None, help="Number of layers to sample (default: all)")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    base_sd, expert_sds = load_state_dicts()

    agg = run_analysis(
        base_sd, expert_sds,
        top_k=args.top_k,
        energy_ratio=args.energy_ratio,
        sample_layers=args.sample_layers,
        device=args.device,
    )

    print_analysis_report(agg)

    if args.output:
        # Convert to JSON-serializable
        save_data = {
            "total_caps": agg["total_caps"],
            "total_within_pairs": agg["total_within_pairs"],
            "total_cross_pairs": agg["total_cross_pairs"],
            "total_conflict_pairs": agg["total_conflict_pairs"],
            "total_redundant_pairs": agg["total_redundant_pairs"],
            "conflict_ratio": agg["conflict_ratio"],
            "pairwise_conflict_agg": dict(agg["pairwise_conflict_agg"]),
            "caps_per_expert_agg": {k: [int(x) for x in v] for k, v in agg["caps_per_expert_agg"].items()},
            "layer_stats": agg["layer_stats"],
        }
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
