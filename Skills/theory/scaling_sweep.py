"""
Scaling coefficient sweep for ETVD and Task Arithmetic merging.
Loads models once, merges with multiple coefficients, saves all variants.
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../InternVL/internvl_chat'))

from etvd_merge import etvd_merge, etvd_v2_merge, etvd_norm_merge, task_arithmetic_merge, ta_norm_merge, DEFAULT_EXCLUDE_PATTERNS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling-coefficients", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--methods", nargs="+", default=["etvd", "ta"])
    parser.add_argument("--output-dir", default="/data/shichao/data/InternVL_merged")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from transformers import AutoModel, AutoTokenizer

    base_path = "OpenGVLab/InternVL2_5-1B"
    expert_paths = [
        "yongxianwei/InternVL2_5-1B_OCR",
        "yongxianwei/InternVL2_5-1B_VQA",
        "yongxianwei/InternVL2_5-1B_Geometry",
        "yongxianwei/InternVL2_5-1B_Chart",
        "yongxianwei/InternVL2_5-1B_Grounding",
    ]
    expert_names = ["OCR", "VQA", "Geometry", "Chart", "Grounding"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)

    print(f"Loading base model: {base_path}")
    base_model = AutoModel.from_pretrained(base_path, torch_dtype=torch.float16, trust_remote_code=True).eval()
    base_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}

    expert_sds = []
    for i, path in enumerate(expert_paths):
        print(f"Loading expert {expert_names[i]}: {path}")
        model = AutoModel.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True).eval()
        expert_sds.append({k: v.cpu() for k, v in model.state_dict().items()})
        del model
        torch.cuda.empty_cache()

    exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)

    for method in args.methods:
        for sc in args.scaling_coefficients:
            sc_str = f"{sc:.1f}".replace(".", "")
            output_name = f"{method}_s{sc_str}_all5"
            output_path = os.path.join(args.output_dir, output_name)

            if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "config.json")):
                print(f"\nSkipping {output_name} (already exists)")
                continue

            print(f"\n{'='*60}")
            print(f"Merging: method={method}, scaling={sc}")
            print(f"Output: {output_path}")
            print(f"{'='*60}")

            if method == "etvd":
                merged_params = etvd_merge(
                    base_sd, expert_sds, expert_names, exclude_patterns,
                    scaling_coefficient=sc,
                    occupancy_threshold=0.3,
                    conflict_strategy="majority",
                )
            elif method == "ta":
                merged_params = task_arithmetic_merge(
                    base_sd, expert_sds, exclude_patterns,
                    scaling_coefficient=sc,
                )
            elif method.startswith("etvd_norm"):
                # etvd_norm_mean, etvd_norm_median, etvd_norm_max
                norm_target = method.split("_")[-1] if method.count("_") >= 2 else "mean"
                merged_params = etvd_norm_merge(
                    base_sd, expert_sds, expert_names, exclude_patterns,
                    scaling_coefficient=sc,
                    occupancy_threshold=0.3,
                    conflict_strategy="majority",
                    norm_target=norm_target,
                )
            elif method == "etvd_v2":
                merged_params = etvd_v2_merge(
                    base_sd, expert_sds, expert_names, exclude_patterns,
                    scaling_coefficient=sc,
                    occupancy_threshold=0.3,
                    conflict_strategy="majority",
                )
            elif method.startswith("ta_norm"):
                # ta_norm_mean, ta_norm_median
                norm_target = method.split("_")[-1] if method.count("_") >= 2 else "mean"
                merged_params = ta_norm_merge(
                    base_sd, expert_sds, exclude_patterns,
                    scaling_coefficient=sc,
                    norm_target=norm_target,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Apply merged params to base model
            full_sd = base_model.state_dict()
            for k, v in merged_params.items():
                if k in full_sd:
                    full_sd[k] = v.to(torch.float16)
            base_model.load_state_dict(full_sd)

            os.makedirs(output_path, exist_ok=True)
            print(f"Saving to {output_path}")
            base_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Reload base weights for next iteration
            base_model.load_state_dict({k: v.to(torch.float16) for k, v in base_sd.items()}, strict=False)

    print("\nAll merges complete!")


if __name__ == "__main__":
    main()
