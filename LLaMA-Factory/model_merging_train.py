import argparse
import os
import re

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex)
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


class TaskVector:
    def __init__(
        self,
        pretrained_model: nn.Module = None,
        finetuned_model: nn.Module = None,
        exclude_param_names_regex: list = None,
        task_vector_param_dict: dict = None,
    ):
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(pretrained_param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex,
            )
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict, f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
        return merged_params


def ties_merging(
    merged_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list,
    param_value_mask_rate: float = 0.8,
    scaling_coefficient: float = 1.0,
):
    def mask_smallest_magnitude_param_values(param_tensor: torch.Tensor, param_value_mask_rate: float = 0.8):
        original_dtype = param_tensor.dtype
        param_tensor = param_tensor.float()
        num_mask_params = int(param_tensor.numel() * param_value_mask_rate)
        flattened = param_tensor.reshape(-1)
        kth_value = flattened.abs().kthvalue(k=num_mask_params).values
        mask = param_tensor.abs() >= kth_value
        return (param_tensor * mask).to(original_dtype)

    def get_param_signs(param_tensors: list):
        param_sum = sum(param_tensors)
        param_signs = torch.sign(param_sum)
        if (param_signs == 0).any():
            majority_sign = torch.sign(param_signs.sum())
            param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(param_tensors: list, param_signs: torch.Tensor):
        preserved_params = []
        for param in param_tensors:
            preserve_mask = ((param_signs > 0) & (param > 0)) | ((param_signs < 0) & (param < 0))
            preserved_params.append(param * preserve_mask)
        num_preserved = sum((p != 0).float() for p in preserved_params)
        merged_param = sum(preserved_params) / torch.clamp(num_preserved, min=1.0)
        return merged_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()),
        exclude_param_names_regex=exclude_param_names_regex,
    )

    print("Creating task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)

    merged_params = {}
    for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
        with torch.no_grad():
            param_vectors = [task_vector[param_name] for task_vector in models_to_merge_task_vectors]
            masked_param_vectors = [
                mask_smallest_magnitude_param_values(param, param_value_mask_rate)
                for param in param_vectors
            ]
            param_signs = get_param_signs(masked_param_vectors)
            merged_delta = disjoint_merge(masked_param_vectors, param_signs)
            merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * merged_delta

    return merged_params


def copy_params_to_model(params: dict, model: nn.Module):
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    original_dtype = input_tensor.dtype
    input_tensor = input_tensor.float()
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)


def mask_model_weights(
    finetuned_model: nn.Module,
    pretrained_model: nn.Module,
    exclude_param_names_regex: list,
    weight_format: str,
    weight_mask_rate: float,
    use_weight_rescale: bool,
    mask_strategy: str,
):
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(param_dict.keys()),
            exclude_param_names_regex=exclude_param_names_regex,
        )
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(
                input_tensor=param_value,
                mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale,
                mask_strategy=mask_strategy,
            )

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            masked_param_dict = new_task_vector.combine_with_pretrained_model(
                pretrained_model=pretrained_model,
                scaling_coefficient=1.0,
            )

    return masked_param_dict


def task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    with torch.no_grad():
        merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
        for index in range(2, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient,
        )

    return merged_params


def svd_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()),
        exclude_param_names_regex=exclude_param_names_regex,
    )

    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)

    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
            torch.cuda.empty_cache()
            param_shape = models_to_merge_task_vectors[0][param_name].shape

            if len(param_shape) == 2 and param_name == "lm_head.weight":
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                sum_u = None
                sum_s = None
                sum_v = None

                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    vec = task_vector_dict[param_name].to(device).float()
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)

                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]

                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                merged_param = torch.linalg.multi_dot([u_u, v_u, torch.diag(sum_s), u_v, v_v]).to(original_dtype).cpu()
                merged_task_vector_dict[param_name] = merged_param
            else:
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient,
        )

    return merged_params


def iso_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    with torch.no_grad():
        merged_task_vector = models_to_merge_task_vectors[0]
        for index in range(1, len(models_to_merge_task_vectors)):
            merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]

    for param_name, param_value in merged_task_vector.task_vector_param_dict.items():
        original_dtype = param_value.dtype
        param_value = param_value.cuda().to(torch.float32)
        u, s, v = torch.linalg.svd(param_value, full_matrices=False)
        avg_singular_value = torch.mean(s)
        avg_s = torch.diag(torch.full_like(s, avg_singular_value))
        merged_param = torch.linalg.multi_dot([u, avg_s, v]).to(original_dtype).cpu()
        merged_task_vector.task_vector_param_dict[param_name] = merged_param

    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )
    return merged_params


def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_redundant_task_vector(param_name, vectors, iter_num=300, num_chunks=2):
        original_dtype = vectors.dtype
        vectors = vectors.float().cuda()
        model_num, m, n = vectors.shape
        models_per_chunk = (model_num + num_chunks - 1) // num_chunks
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(vectors.reshape(model_num, -1), p=2, dim=-1))
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            optimizer.zero_grad()
            total_loss = 0.0
            for chunk_idx in range(num_chunks):
                start_model = chunk_idx * models_per_chunk
                end_model = min((chunk_idx + 1) * models_per_chunk, model_num)
                vectors_chunk = vectors[start_model:end_model, :, :]
                chunk_norms = l2_norms[start_model:end_model]
                disturbing_vectors = merging_vector.unsqueeze(0) - vectors_chunk
                inner_product = torch.matmul(disturbing_vectors, vectors_chunk.transpose(1, 2))
                chunk_loss = torch.sum(torch.square(inner_product) / chunk_norms.unsqueeze(-1).unsqueeze(-1))
                total_loss += chunk_loss
            if i % 10 == 0:
                print(f"Step {i}, loss: {total_loss.item()}")
            total_loss.backward()
            optimizer.step()
        return merging_vector.data.detach().to(original_dtype).cpu()

    merged_task_vector_dict = {}
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([task_vector.task_vector_param_dict[param_name] for task_vector in models_to_merge_task_vectors])
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )
    return merged_params


def wudi_merging2(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(
            pretrained_model=merged_model,
            finetuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            s_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            s_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(s_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2)
            del u, s, v, u2, s2, v2, s_matrix, s_mask, v_mask
        low_rank = torch.stack(low_rank_list).to(original_dtype)
        taskvector = torch.stack(taskvector_list).to(original_dtype)

        merging_vector = torch.nn.Parameter(average_vector.to(original_dtype))
        optimizer = torch.optim.SGD([merging_vector], lr=1e-4, momentum=0.9)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1)).to(original_dtype)
        del vectors, low_rank_list, taskvector_list
        torch.cuda.empty_cache()

        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return merging_vector.data.detach().cpu()

    merged_task_vector_dict = {}
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([task_vector.task_vector_param_dict[param_name] for task_vector in models_to_merge_task_vectors])
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient,
    )
    return merged_params


def parse_dtype(dtype_name: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def merge_models(
    base_model_path: str,
    merge_model_paths: list[str],
    output_path: str,
    merge_method: str = "wudi2",
    scaling_coefficient: float = 1.0,
    exclude_param_names_regex: list | None = None,
    torch_dtype: torch.dtype = torch.float16,
):
    if not merge_model_paths:
        raise ValueError("merge_model_paths cannot be empty.")

    if exclude_param_names_regex is None:
        exclude_param_names_regex = [
            "visual..*",
            ".*embed_tokens.*",
            ".*lm_head.*",
            ".*norm.*",
            ".*bias.*",
        ]

    print(f"Loading base model: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).eval()

    print("Loading models to merge...")
    models_to_merge = []
    for model_path in merge_model_paths:
        print(f"  - {model_path}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()
        models_to_merge.append(model)

    base_state_dict = base_model.state_dict()

    if merge_method == "task_arithmetic":
        merged_params = task_arithmetic(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "ties":
        merged_params = ties_merging(base_model, models_to_merge, exclude_param_names_regex, 0.8, float(scaling_coefficient))
    elif merge_method == "dare ta":
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
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
        merged_params = task_arithmetic(base_model, new_models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "dare ties":
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
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
        merged_params = ties_merging(base_model, new_models_to_merge, exclude_param_names_regex, 0.8, float(scaling_coefficient))
    elif merge_method == "svd":
        merged_params = svd_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "iso":
        merged_params = iso_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "wudi":
        merged_params = wudi_merging(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    elif merge_method == "wudi2":
        merged_params = wudi_merging2(base_model, models_to_merge, exclude_param_names_regex, float(scaling_coefficient))
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")

    for key in merged_params:
        if key in base_state_dict:
            base_state_dict[key] = merged_params[key]

    base_model.load_state_dict(base_state_dict)
    base_model = base_model.cuda()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    base_model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description="Merge Qwen2-VL models with a configurable model list.")
    parser.add_argument("--base-model", required=True, help="Base model path.")
    parser.add_argument("--merge-models", nargs="+", required=True, help="One or more model paths to merge into the base model.")
    parser.add_argument("--output-path", required=True, help="Directory to save the merged model.")
    parser.add_argument(
        "--merge-method",
        default="wudi2",
        choices=["task_arithmetic", "ties", "dare ta", "dare ties", "svd", "iso", "wudi", "wudi2"],
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
    )


if __name__ == "__main__":
    main()
