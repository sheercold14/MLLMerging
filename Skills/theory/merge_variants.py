"""
Merge variants for targeted experiments.
Uses CPU-only model loading to avoid GPU contention.
"""

import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../InternVL/internvl_chat'))

from etvd_merge import task_arithmetic_merge, etvd_v2_merge, DEFAULT_EXCLUDE_PATTERNS


def weighted_ta_merge(base_state_dict, expert_state_dicts, exclude_param_names_regex,
                      expert_weights, verbose=True):
    """Task Arithmetic with per-expert weights: base + Σ_i α_i * τ_i"""
    from etvd_merge import get_param_names_to_merge
    from tqdm import tqdm

    param_names = get_param_names_to_merge(list(base_state_dict.keys()), exclude_param_names_regex)
    merged = {}
    for param_name in tqdm(param_names, desc="Weighted TA", disable=not verbose):
        base_param = base_state_dict[param_name].float()
        weighted_sum = torch.zeros_like(base_param)
        for i, sd in enumerate(expert_state_dicts):
            tv = sd[param_name].float() - base_param
            weighted_sum += expert_weights[i] * tv
        merged[param_name] = base_param + weighted_sum
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["ta_no_geometry", "ta_no_grounding", "ta_weighted_v1",
                                 "etvd_v2_s01", "ta_4expert_sweep"])
    parser.add_argument("--output-dir", default="/data/shichao/data/InternVL_merged")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    base_path = "OpenGVLab/InternVL2_5-1B"
    all_expert_paths = [
        "yongxianwei/InternVL2_5-1B_OCR",
        "yongxianwei/InternVL2_5-1B_VQA",
        "yongxianwei/InternVL2_5-1B_Geometry",
        "yongxianwei/InternVL2_5-1B_Chart",
        "yongxianwei/InternVL2_5-1B_Grounding",
    ]
    all_expert_names = ["OCR", "VQA", "Geometry", "Chart", "Grounding"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)

    print(f"Loading base model (CPU): {base_path}")
    base_model = AutoModel.from_pretrained(base_path, torch_dtype=torch.float16,
                                            trust_remote_code=True, device_map="cpu").eval()
    base_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}

    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    # Define which experts to load based on experiment
    if args.experiment == "ta_no_geometry":
        expert_indices = [0, 1, 3, 4]  # OCR, VQA, Chart, Grounding
        configs = [("ta_no_geometry_s01", 0.1)]
    elif args.experiment == "ta_no_grounding":
        expert_indices = [0, 1, 2, 3]  # OCR, VQA, Geometry, Chart
        configs = [("ta_no_grounding_s01", 0.1)]
    elif args.experiment == "ta_weighted_v1":
        expert_indices = [0, 1, 2, 3, 4]
        # Based on interference analysis: boost Chart, reduce Geometry/Grounding
        configs = [("ta_weighted_v1", None)]  # special handling below
    elif args.experiment == "ta_4expert_sweep":
        expert_indices = [0, 1, 3, 4]  # OCR, VQA, Chart, Grounding (no Geometry)
        configs = [
            ("ta_no_geometry_s01", 0.1),
            ("ta_no_geometry_s015", 0.15),
            ("ta_no_geometry_s02", 0.2),
        ]
    else:
        expert_indices = list(range(5))
        configs = []

    # Load needed experts
    expert_sds = []
    expert_names = []
    for i in expert_indices:
        print(f"Loading expert {all_expert_names[i]} (CPU): {all_expert_paths[i]}")
        model = AutoModel.from_pretrained(all_expert_paths[i], torch_dtype=torch.float16,
                                          trust_remote_code=True, device_map="cpu").eval()
        expert_sds.append({k: v.cpu() for k, v in model.state_dict().items()})
        expert_names.append(all_expert_names[i])
        del model

    for output_name, sc in configs:
        output_path = os.path.join(args.output_dir, output_name)
        if os.path.exists(os.path.join(output_path, "config.json")):
            print(f"\nSkipping {output_name} (already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"Merging: {output_name}")
        print(f"{'='*60}")

        if args.experiment == "ta_weighted_v1":
            # Weighted TA: α_OCR=0.1, α_VQA=0.1, α_Geometry=0.05, α_Chart=0.15, α_Grounding=0.05
            weights = [0.1, 0.1, 0.05, 0.15, 0.05]
            merged_params = weighted_ta_merge(
                base_sd, expert_sds, exclude_patterns,
                expert_weights=weights,
            )
        else:
            merged_params = task_arithmetic_merge(
                base_sd, expert_sds, exclude_patterns,
                scaling_coefficient=sc,
            )

        # Apply merged params
        full_sd = base_model.state_dict()
        for k, v in merged_params.items():
            if k in full_sd:
                full_sd[k] = v.to(torch.float16)
        base_model.load_state_dict(full_sd)

        os.makedirs(output_path, exist_ok=True)
        print(f"Saving to {output_path}")
        base_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Reload base for next iteration
        base_model.load_state_dict({k: v.to(torch.float16) for k, v in base_sd.items()}, strict=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
