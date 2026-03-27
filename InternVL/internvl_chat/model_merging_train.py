import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer

from model_merging import (
    anova_wudi2_merging,
    copy_params_to_model,
    iso_merging,
    mask_model_weights,
    mc_wudi2_merging,
    mc_tucker_wudi2_cf_merging,
    svd_merging,
    task_arithmetic,
    ties_merging,
    tucker_wudi2_cf_merging,
    tucker_wudi2_merging,
    wudi_merging,
    wudi_merging2,
)


DEFAULT_EXCLUDE_PATTERNS = [
    "vision_model.*",
    ".*lm_head.*",
    ".*norm.*",
    ".*embed_tokens.*",
    ".*bias.*",
]

MERGE_METHOD_CHOICES = [
    "task_arithmetic",
    "ties",
    "dare ta",
    "dare ties",
    "svd",
    "iso",
    "wudi",
    "wudi2",
    "mc_wudi2",
    "anova_wudi2",
    "tucker_wudi2",
    "tucker_wudi2_cf",
    "mc_tucker_wudi2_cf",
]


def parse_dtype(dtype_name: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def load_model(model_path: str, torch_dtype: torch.dtype):
    print(f"Loading model: {model_path}")
    return AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).eval().cuda()


def merge_models(
    base_model_path: str,
    merge_model_paths: list[str],
    output_path: str,
    merge_method: str = "wudi2",
    scaling_coefficient: float = 1.0,
    exclude_param_names_regex: list[str] | None = None,
    torch_dtype: torch.dtype = torch.float16,
    mc_beta: float = 0.5,
    mc_energy_ratio: float = 0.95,
    tucker_top_k: int = 40,
):
    if not merge_model_paths:
        raise ValueError("merge_model_paths cannot be empty.")

    if exclude_param_names_regex is None:
        exclude_param_names_regex = list(DEFAULT_EXCLUDE_PATTERNS)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    base_model = load_model(base_model_path, torch_dtype=torch_dtype)
    base_state_dict = base_model.state_dict()

    models_to_merge = [load_model(model_path, torch_dtype=torch_dtype) for model_path in merge_model_paths]

    if merge_method == "task_arithmetic":
        merged_params = task_arithmetic(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "ties":
        merged_params = ties_merging(base_model, models_to_merge, exclude_param_names_regex, 0.8, float(scaling_coefficient))
    elif merge_method == "dare ta":
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            for new_model_to_merge, weight_mask_rate in zip(models_to_merge, weight_mask_rates):
                masked_param_dict = mask_model_weights(
                    finetuned_model=new_model_to_merge,
                    pretrained_model=base_model,
                    exclude_param_names_regex=exclude_param_names_regex,
                    weight_format="delta_weight",
                    weight_mask_rate=weight_mask_rate,
                    use_weight_rescale=True,
                    mask_strategy="random",
                )
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        merged_params = task_arithmetic(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "dare ties":
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            for new_model_to_merge, weight_mask_rate in zip(models_to_merge, weight_mask_rates):
                masked_param_dict = mask_model_weights(
                    finetuned_model=new_model_to_merge,
                    pretrained_model=base_model,
                    exclude_param_names_regex=exclude_param_names_regex,
                    weight_format="delta_weight",
                    weight_mask_rate=weight_mask_rate,
                    use_weight_rescale=True,
                    mask_strategy="random",
                )
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        merged_params = ties_merging(base_model, models_to_merge, exclude_param_names_regex, 0.8, float(scaling_coefficient))
    elif merge_method == "svd":
        merged_params = svd_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "iso":
        merged_params = iso_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "wudi":
        merged_params = wudi_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "wudi2":
        merged_params = wudi_merging2(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "mc_wudi2":
        merged_params = mc_wudi2_merging(base_model, models_to_merge, exclude_param_names_regex,
                                          float(scaling_coefficient), beta=mc_beta, energy_ratio=mc_energy_ratio)
    elif merge_method == "anova_wudi2":
        merged_params = anova_wudi2_merging(base_model, models_to_merge, exclude_param_names_regex,
                                             float(scaling_coefficient))
    elif merge_method == "tucker_wudi2":
        merged_params = tucker_wudi2_merging(base_model, models_to_merge, exclude_param_names_regex,
                                              float(scaling_coefficient), top_k=tucker_top_k)
    elif merge_method == "tucker_wudi2_cf":
        merged_params = tucker_wudi2_cf_merging(base_model, models_to_merge, exclude_param_names_regex,
                                                 float(scaling_coefficient), top_k=tucker_top_k)
    elif merge_method == "mc_tucker_wudi2_cf":
        merged_params = mc_tucker_wudi2_cf_merging(base_model, models_to_merge, exclude_param_names_regex,
                                                    float(scaling_coefficient), beta=mc_beta,
                                                    energy_ratio=mc_energy_ratio, top_k=tucker_top_k)
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")

    for key, value in merged_params.items():
        if key in base_state_dict:
            base_state_dict[key] = value

    base_model.load_state_dict(base_state_dict)

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    for model in models_to_merge:
        del model
    del base_model
    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description="Merge InternVL models with a configurable model list.")
    parser.add_argument("--base-model", required=True, help="Base model path.")
    parser.add_argument("--merge-models", nargs="+", required=True, help="One or more model paths to merge into the base model.")
    parser.add_argument("--output-path", required=True, help="Directory to save the merged model.")
    parser.add_argument(
        "--merge-method",
        default="wudi2",
        choices=MERGE_METHOD_CHOICES,
        help="Merge method.",
    )
    parser.add_argument("--scaling-coefficient", type=float, default=1.0, help="Scaling coefficient used during merging.")
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        dest="exclude_patterns",
        default=None,
        help="Regex pattern for parameters to exclude. Can be specified multiple times.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype used when loading models.",
    )
    parser.add_argument("--mc-beta", type=float, default=0.5, help="MC-wudi2: conflict suppression strength (0-1)")
    parser.add_argument("--mc-energy-ratio", type=float, default=0.95, help="MC-wudi2: SVD energy ratio for micro-cap extraction")
    parser.add_argument("--tucker-top-k", type=int, default=40, help="Tucker-wudi2: per-expert SVD rank for basis construction")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    merge_models(
        base_model_path=args.base_model,
        merge_model_paths=args.merge_models,
        output_path=args.output_path,
        merge_method=args.merge_method,
        scaling_coefficient=args.scaling_coefficient,
        exclude_param_names_regex=args.exclude_patterns,
        torch_dtype=parse_dtype(args.torch_dtype),
        mc_beta=args.mc_beta,
        mc_energy_ratio=args.mc_energy_ratio,
        tucker_top_k=args.tucker_top_k,
    )


if __name__ == "__main__":
    main()
