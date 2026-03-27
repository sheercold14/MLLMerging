import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import re
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params

def ties_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
    """
    ties merging method (layer-by-layer implementation to save memory)
    :param merged_model: nn.Module, the merged model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """
    def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
        """
        convert parameter dictionary in task vector to a single vector
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

    def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
        """
        convert a single vector to parameter dictionary in task vector
        :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
        :param task_vector: TaskVector, task vector
        :return:
        """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params)
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # Convert to float32 to support kthvalue operation
        flattened_models_to_merge_param = flattened_models_to_merge_param.float()
        
        num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)
        
        # Calculate the threshold
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
        
        # Create mask and apply
        mask = flattened_models_to_merge_param.abs() >= kth_values
        
        # Apply mask and convert back to original dtype
        return (flattened_models_to_merge_param * mask).to(flattened_models_to_merge_param.dtype)

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
        get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

        return merged_flattened_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

    flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for task_vector in models_to_merge_task_vectors]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)

        # merged parameter dictionary
        merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

    return merged_params

def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
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
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor.to(original_dtype)

def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict

def task_arithmetic(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            merged_task_vector = models_to_merge_task_vectors[0] + models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + models_to_merge_task_vectors[index]
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params

def svd_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    SVD merging method that uses Singular Value Decomposition to merge models.
    Args:
        merged_model: nn.Module, the base model to merge into  
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to merge the task vectors
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    # Get the parameter names to merge
    pretrained_param_dict = {param_name: param_value for param_name, param_value in merged_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(pretrained_param_dict.keys()), 
        exclude_param_names_regex=exclude_param_names_regex
    )
    
    # Compute task vectors
    print("Computing task vectors...")
    models_to_merge_task_vectors = []
    for model_to_merge in models_to_merge:
        task_vector_dict = {}
        for param_name in param_names_to_merge:
            # Compute difference as task vector
            task_vector_dict[param_name] = model_to_merge.state_dict()[param_name] - merged_model.state_dict()[param_name]
        models_to_merge_task_vectors.append(task_vector_dict)
    
    sv_reduction = 1.0 / len(models_to_merge)
    device = torch.device("cuda")
    first_param_name = list(models_to_merge_task_vectors[0].keys())[0]
    original_dtype = models_to_merge_task_vectors[0][first_param_name].dtype
    print("Computing SVD merging...")

    with torch.no_grad():
        merged_task_vector_dict = {}
        # Process each parameter
        for param_name in tqdm(param_names_to_merge, desc="Processing model parameters"):
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
            
            # Check parameter shape
            param_shape = models_to_merge_task_vectors[0][param_name].shape
            
            if len(param_shape) == 2 and param_name == 'lm_head.weight':
                print(f"Processing parameter {param_name}, shape: {param_shape}")
                # Apply SVD merging for 2D tensors
                
                # Create temporary variables to store merged results
                sum_u = None
                sum_s = None
                sum_v = None
                
                # Process each model's task vector
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors):
                    # Move parameter to GPU for computation
                    vec = task_vector_dict[param_name].to(device).float()
                    
                    # Compute SVD
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    
                    # Compute reduced index
                    reduced_index_s = int(s.shape[0] * sv_reduction)
                    
                    # Initialize and prepare storage for the first model
                    if i == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    
                    # Store important components for each model
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
                
                # Compute final merged parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                # Compute merged result and move back to CPU
                merged_param = torch.linalg.multi_dot([
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                ]).to(original_dtype).cpu()
                
                # Store merged parameter
                merged_task_vector_dict[param_name] = merged_param
                
            else:
                # Use simple averaging for non-2D tensors
                merged_param = models_to_merge_task_vectors[0][param_name].clone()
                for i, task_vector_dict in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                merged_task_vector_dict[param_name] = merged_param

        # Create merged task vector and combine with base model
        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=merged_model,
            scaling_coefficient=scaling_coefficient
        )
        
    return merged_params

def iso_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    ISO merging method, uses SVD and equalizes singular values to reduce interference between task vectors

    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, models to be merged
        exclude_param_names_regex: list, regex patterns for parameter names to exclude
        scaling_coefficient: float, scaling coefficient for merging task vectors
    Returns:
        dict: merged parameter dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    
    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge, exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        # Get parameter shape from the first task vector
        param_shape = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape
        
        if len(param_shape) == 2:
            # For 2D parameters, perform SVD merging
            with torch.no_grad():
                merged_param_value = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
                for index in range(1, len(models_to_merge_task_vectors)):
                    merged_param_value = merged_param_value + models_to_merge_task_vectors[index].task_vector_param_dict[param_name]
            
            # SVD and equalize singular values
            original_dtype = merged_param_value.dtype
            merged_param_value = merged_param_value.cuda().to(torch.float32)
            u, s, v = torch.linalg.svd(merged_param_value, full_matrices=False)
            avg_singular_value = torch.mean(s)
            avg_s = torch.diag(torch.full_like(s, avg_singular_value))
            
            merged_param = torch.linalg.multi_dot([
                u, avg_s, v
            ]).to(original_dtype)
            
            # Store merged parameter
            merged_task_vector_dict[param_name] = merged_param
        else:
            # For non-2D parameters, compute the average of all task vectors
            print(param_name)
            with torch.no_grad():
                merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
                for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                    vec = task_vector.task_vector_param_dict[param_name]
                    merged_param += (vec - merged_param) / (i + 1)
                
                merged_task_vector_dict[param_name] = merged_param

    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    return merged_params

def wudi_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Wudi merging method that optimizes a merging vector to minimize interference between task vectors
    
    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32).cuda()
       
        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - vectors
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            print(f"Step {i}, loss: {loss.item()}")
            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)#.cpu()
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector
    
    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def wudi_merging2(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
    """
    Wudi merging2 method that optimizes a merging vector to minimize interference between task vectors
    
    Args:
        merged_model: nn.Module, the base model to merge into
        models_to_merge: list, individual models that need to be merged
        exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        scaling_coefficient: float, scaling coefficient to apply to the final merged vector
    Returns:
        dict: merged parameters dictionary
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model, 
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]
    
    def get_redundant_task_vector(param_name, vectors, iter_num=300):
        """
        Optimize a merging vector to minimize interference between task vectors
        
        Args:
            param_name: str, name of the parameter
            vectors: torch.Tensor, stacked task vectors to merge
            iter_num: int, number of optimization iterations
        Returns:
            torch.Tensor: optimized merging vector
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)

        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / vectors.shape[0])
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask  # (n, n)
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)  # m x n
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        # Initialize with sum of vectors as starting point
        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        
        # Compute L2 norms for normalization
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
       
        # Optimization loop
        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            # Calculate disturbing vectors
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            # Calculate inner products
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            # Calculate loss
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 10 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return merging_vector.data.detach().to(original_dtype)
    
    merged_task_vector_dict = {}
    
    # Process each parameter
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            
            # Stack task vectors for this parameter
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name] 
                for task_vector in models_to_merge_task_vectors
            ])
            
            # Get optimized merging vector
            merging_vector = get_redundant_task_vector(param_name, values, iter_num=300)
            merged_task_vector_dict[param_name] = merging_vector

    # Handle non-attention weights using simple averaging for completeness
    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict.keys():
        if param_name not in merged_task_vector_dict:
            print(f"Using simple averaging for {param_name}")
            merged_param = models_to_merge_task_vectors[0].task_vector_param_dict[param_name].clone()
            for i, task_vector in enumerate(models_to_merge_task_vectors[1:], 1):
                vec = task_vector.task_vector_param_dict[param_name]
                merged_param += (vec - merged_param) / (i + 1)
            merged_task_vector_dict[param_name] = merged_param
    
    # Create merged task vector and combine with base model
    merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_dict)
    merged_params = merged_task_vector.combine_with_pretrained_model(
        pretrained_model=merged_model,
        scaling_coefficient=scaling_coefficient
    )
    
    return merged_params

def mc_wudi2_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                     scaling_coefficient: float = 1.0, beta: float = 0.5, energy_ratio: float = 0.95):
    """
    Route A: Micro-Capability aware wudi2 (MC-wudi2).

    Pre-resolves cross-expert conflicts at the micro-capability level before
    feeding cleaned task vectors to wudi2's optimization.

    Algorithm:
    1. For each 2D layer, extract micro-capabilities via individual SVD
    2. Build K×K interference matrix I_ab = (u_a·u_b)(v_a·v_b) * σ_a * σ_b
    3. Suppress conflicting micro-capabilities (reduce σ for high-conflict caps)
    4. Reconstruct cleaned task vectors
    5. Run wudi2 optimization on cleaned task vectors
    """
    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    def suppress_conflicts(vectors, beta=0.5, energy_ratio=0.95):
        """
        Suppress conflicting micro-capabilities across experts.

        Args:
            vectors: (N, m, n) stacked task vectors
            beta: conflict suppression strength (0=no suppression, 1=aggressive)
            energy_ratio: SVD energy threshold for micro-capability extraction
        Returns:
            cleaned vectors (N, m, n)
        """
        N, m, n = vectors.shape
        device = vectors.device

        # Extract micro-capabilities for each expert
        all_sigmas = []
        all_us = []
        all_vs = []
        expert_ids = []
        expert_k_ranges = []  # (start, end) indices for each expert

        idx = 0
        for i in range(N):
            U_i, S_i, Vt_i = torch.linalg.svd(vectors[i], full_matrices=False)
            total_energy = torch.sum(S_i ** 2)
            cumulative = torch.cumsum(S_i ** 2, dim=0)
            k = torch.searchsorted(cumulative, energy_ratio * total_energy).item() + 1
            k = max(k, 1)
            k = min(k, S_i.shape[0])

            all_sigmas.append(S_i[:k])
            all_us.append(U_i[:, :k])
            all_vs.append(Vt_i[:k, :])
            expert_ids.extend([i] * k)
            expert_k_ranges.append((idx, idx + k))
            idx += k

        # Stack all micro-capabilities
        sigmas = torch.cat(all_sigmas)  # (K,)
        K = sigmas.shape[0]
        expert_ids = torch.tensor(expert_ids, device=device)

        # Build interference matrix efficiently using batched ops
        # U overlap: need block-wise computation to avoid OOM for large K
        # For K~1800, K×K = 3.24M entries — manageable
        U_cat = torch.cat(all_us, dim=1).T  # (K, m)
        V_cat = torch.cat(all_vs, dim=0)    # (K, n)

        u_overlap = U_cat @ U_cat.T  # (K, K)
        v_overlap = V_cat @ V_cat.T  # (K, K)
        sigma_outer = sigmas.unsqueeze(1) * sigmas.unsqueeze(0)
        I_matrix = u_overlap * v_overlap * sigma_outer

        # Compute conflict scores for each micro-capability
        # conflict_score_a = sum of |I_{a,b}| for b from different experts where I_{a,b} < 0
        suppression_factors = torch.ones(K, device=device)

        for i in range(N):
            start_i, end_i = expert_k_ranges[i]
            # Mask for other experts' capabilities
            other_mask = expert_ids != i  # (K,)

            for a_local in range(end_i - start_i):
                a = start_i + a_local
                # Get conflict values with other experts
                cross_vals = I_matrix[a] * other_mask.float()
                # Only count negative (conflict) interactions
                conflict_vals = torch.clamp(-cross_vals, min=0)
                conflict_score = conflict_vals.sum().item()

                if conflict_score > 0:
                    # Normalize by the capability's own contribution
                    own_contribution = sigmas[a].item() ** 2
                    relative_conflict = conflict_score / (own_contribution + 1e-10)
                    suppression = max(0.0, 1.0 - beta * relative_conflict)
                    suppression_factors[a] = suppression

        # Reconstruct cleaned task vectors
        cleaned = torch.zeros_like(vectors)
        for i in range(N):
            start_i, end_i = expert_k_ranges[i]
            suppressed_s = all_sigmas[i] * suppression_factors[start_i:end_i]
            cleaned[i] = all_us[i] @ torch.diag(suppressed_s) @ all_vs[i]

        n_suppressed = (suppression_factors < 1.0).sum().item()
        n_zeroed = (suppression_factors == 0.0).sum().item()
        avg_suppression = suppression_factors.mean().item()
        print(f"  MC: K={K}, suppressed={n_suppressed}/{K} "
              f"({n_suppressed/K*100:.1f}%), zeroed={n_zeroed}, "
              f"avg_factor={avg_suppression:.3f}")

        del I_matrix, u_overlap, v_overlap, sigma_outer, U_cat, V_cat
        return cleaned

    def get_redundant_task_vector_mc(param_name, vectors, beta=0.5,
                                     energy_ratio=0.95, iter_num=300):
        """wudi2 optimization with MC pre-processing."""
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)

        # Step 1: Suppress conflicts
        vectors = suppress_conflicts(vectors, beta=beta, energy_ratio=energy_ratio)

        # Step 2: Standard wudi2 optimization on cleaned vectors
        average_vector = vectors.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
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
            S_matrix = torch.zeros(vector.shape[0], vector.shape[1], device=s.device)
            min_dim = min(vector.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_vector)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))

        for i in tqdm(range(iter_num), desc=f"Optimizing {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 50 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return merging_vector.data.detach().to(original_dtype)

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_redundant_task_vector_mc(
                param_name, values, beta=beta, energy_ratio=energy_ratio, iter_num=300
            )
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
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


def anova_wudi2_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                        scaling_coefficient: float = 1.0, occupancy_threshold: float = 0.3,
                        alpha_0: float = 1.0, alpha_g: float = 0.8):
    """
    Route B: ANOVA-wudi2 — hierarchical decomposition + wudi2 optimization.

    Replaces wudi2's crude global-average decomposition with ETVD-guided
    hierarchical ANOVA decomposition:
      tau_i = mu + (mu_g(i) - mu) + residual_i

    The shared (mu) and group (mu_g - mu) components are deterministically merged.
    Only the residuals go through wudi2's optimization, giving it a cleaner target.

    Algorithm:
    1. Pre-processing: discover expert groups via ETVD U-matrix across all layers
    2. Per layer: ANOVA decomposition → deterministic shared merge + wudi2 on residuals
    """
    assert isinstance(scaling_coefficient, float)
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    N = len(models_to_merge)

    # Step 1: Discover expert groups via ETVD U-matrix across all layers
    print("ANOVA: Discovering expert groups via joint SVD...")
    affinity = torch.zeros(N, N)

    param_names_2d = [
        p for p in models_to_merge_task_vectors[0].task_vector_param_dict
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[p].shape) == 2
        and "lm_head" not in p
    ]

    for param_name in tqdm(param_names_2d, desc="Computing affinity"):
        vectors = torch.stack([
            tv.task_vector_param_dict[param_name].float()
            for tv in models_to_merge_task_vectors
        ])  # N × m × n
        # Flatten and stack for joint SVD
        T = vectors.reshape(N, -1)  # N × d
        U, S, Vt = torch.linalg.svd(T, full_matrices=False)

        # For each singular direction, find which experts participate
        for k in range(min(N, S.shape[0])):
            significant = torch.abs(U[:, k]) > occupancy_threshold
            if significant.sum() >= 2:
                idxs = torch.where(significant)[0]
                for ii in range(len(idxs)):
                    for jj in range(ii + 1, len(idxs)):
                        affinity[idxs[ii], idxs[jj]] += 1
                        affinity[idxs[jj], idxs[ii]] += 1

    # Simple group discovery: threshold-based clustering
    # Normalize affinity by number of layers
    affinity = affinity / len(param_names_2d)
    print(f"ANOVA: Affinity matrix (normalized by {len(param_names_2d)} layers):")
    for i in range(N):
        row = [f"{affinity[i, j]:.3f}" for j in range(N)]
        print(f"  Expert {i}: [{', '.join(row)}]")

    # Hierarchical clustering: merge pairs with affinity > median
    # Use simple greedy: find highest affinity pair, merge, repeat
    groups = [[i] for i in range(N)]  # start with each expert in own group
    flat_affinities = []
    for i in range(N):
        for j in range(i + 1, N):
            flat_affinities.append((affinity[i, j].item(), i, j))
    flat_affinities.sort(reverse=True)

    # Merge groups with affinity above threshold (top quartile)
    if flat_affinities:
        merge_threshold = flat_affinities[len(flat_affinities) // 4][0]  # top 25%
        merged_ids = list(range(N))
        for aff_val, i, j in flat_affinities:
            if aff_val < merge_threshold:
                break
            gi, gj = merged_ids[i], merged_ids[j]
            if gi != gj:
                # Merge group j into group i
                for k in range(N):
                    if merged_ids[k] == gj:
                        merged_ids[k] = gi

        # Build final groups
        group_map = {}
        for i in range(N):
            gid = merged_ids[i]
            if gid not in group_map:
                group_map[gid] = []
            group_map[gid].append(i)
        groups = list(group_map.values())

    print(f"ANOVA: Discovered {len(groups)} groups: {groups}")

    # Build expert-to-group mapping
    expert_to_group = {}
    for g_idx, group in enumerate(groups):
        for expert_idx in group:
            expert_to_group[expert_idx] = g_idx

    # Step 2: Per-layer ANOVA decomposition + wudi2 on residuals
    def get_anova_wudi2_vector(param_name, vectors, iter_num=300):
        """ANOVA decomposition + wudi2 optimization on residuals."""
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)
        N_loc = vectors.shape[0]

        # ANOVA decomposition
        mu = vectors.mean(dim=0)  # global mean
        mu_g_list = []
        for g_idx, group in enumerate(groups):
            group_mean = torch.stack([vectors[i] for i in group]).mean(dim=0)
            mu_g_list.append(group_mean)

        residuals = torch.stack([
            vectors[i] - mu_g_list[expert_to_group[i]]
            for i in range(N_loc)
        ])  # N × m × n

        # Deterministic shared component
        m_shared = alpha_0 * mu
        for g_idx, group in enumerate(groups):
            m_shared = m_shared + alpha_g * (mu_g_list[g_idx] - mu) * (len(group) / N_loc)

        # Compute energy distribution for logging
        mu_energy = torch.norm(mu).item()
        group_energy = sum(
            torch.norm(mu_g_list[g] - mu).item() * len(groups[g])
            for g in range(len(groups))
        ) / N_loc
        residual_energy = torch.norm(residuals).item() / N_loc
        print(f"  ANOVA: mu_energy={mu_energy:.4f}, group_energy={group_energy:.4f}, "
              f"residual_energy={residual_energy:.4f}")

        # Standard wudi2 optimization on RESIDUALS (not full task vectors)
        average_residual = residuals.mean(dim=0)
        low_rank_list = []
        taskvector_list = []
        for i in range(N_loc):
            r = residuals[i]
            u, s, v = torch.linalg.svd(r, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(r - average_residual, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N_loc)
            u2 = u2[:, :reduced_index_s]
            s2 = s2[:reduced_index_s]
            v2 = v2[:reduced_index_s, :]
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(r.shape[0], r.shape[1], device=s.device)
            min_dim = min(r.shape)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_list.append(S_matrix @ v)
            taskvector_list.append(u2 @ torch.diag_embed(s2) @ v2 + average_residual)
        low_rank = torch.stack(low_rank_list)
        taskvector = torch.stack(taskvector_list)

        merging_vector = torch.nn.Parameter(torch.sum(residuals, dim=0))
        optimizer = torch.optim.Adam([merging_vector], lr=1e-5)
        l2_norms = torch.square(torch.norm(residuals.reshape(N_loc, -1), p=2, dim=-1))
        # Avoid division by zero for very small residuals
        l2_norms = torch.clamp(l2_norms, min=1e-10)

        for i in tqdm(range(iter_num), desc=f"ANOVA-opt {param_name}", leave=False):
            disturbing_vectors = merging_vector.unsqueeze(0) - taskvector
            inner_product = torch.matmul(disturbing_vectors, low_rank.transpose(1, 2))
            loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
            if i % 50 == 0:
                print(f"Step {i}, loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        m_residual = merging_vector.data.detach()

        # Combine: shared + scaled residual
        result = m_shared + m_residual
        return result.to(original_dtype)

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_anova_wudi2_vector(param_name, values, iter_num=300)
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
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


def tucker_wudi2_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                         scaling_coefficient: float = 1.0, top_k: int = 40, iter_num: int = 300):
    """
    Route C: Tucker subspace wudi2.

    Constrains wudi2's optimization to the joint low-rank subspace spanned by
    the experts' task vectors, reducing search space by ~100x.

    Algorithm:
    1. For each 2D layer, compute individual SVDs → extract top-k left/right singular vectors
    2. Pool and orthogonalize → joint output basis B, joint input basis C
    3. Project task vectors and wudi2 targets into (r2 × r3) subspace
    4. Optimize in low-dimensional space
    5. Reconstruct: m = B @ X* @ C^T
    """
    assert isinstance(scaling_coefficient, float)
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    def get_tucker_merging_vector(param_name, vectors, top_k=40, iter_num=300):
        """
        Optimize merging vector in Tucker subspace.
        """
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)
        N, m, n = vectors.shape

        # Step 1: Build joint bases via pooled SVD
        U_pool_list = []
        V_pool_list = []
        for i in range(N):
            Ui, Si, Vti = torch.linalg.svd(vectors[i], full_matrices=False)
            k = min(top_k, Si.shape[0])
            U_pool_list.append(Ui[:, :k])
            V_pool_list.append(Vti[:k, :].T)  # n × k

        U_pool = torch.cat(U_pool_list, dim=1)  # m × (N*k)
        V_pool = torch.cat(V_pool_list, dim=1)  # n × (N*k)

        # Orthogonalize via QR
        B, _ = torch.linalg.qr(U_pool)  # m × r2
        C, _ = torch.linalg.qr(V_pool)  # n × r3
        r2, r3 = B.shape[1], C.shape[1]
        print(f"  Tucker: r2={r2}, r3={r3} (vs full {m}×{n}={m*n}), "
              f"compression={m*n/(r2*r3):.0f}x")

        # Step 2: wudi2-style target construction (full space, same as original wudi2)
        average_vector = vectors.mean(dim=0)
        target_full_list = []
        low_rank_full_list = []

        for i in range(N):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N)

            # Cleaned target (same as wudi2)
            u2_k = u2[:, :reduced_index_s]
            s2_k = s2[:reduced_index_s]
            v2_k = v2[:reduced_index_s, :]
            target_i = u2_k @ torch.diag(s2_k) @ v2_k + average_vector

            # Low-rank projection subspace (same as wudi2)
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(m, n, device=s.device)
            min_dim = min(m, n)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_i = S_matrix @ v  # m × n

            target_full_list.append(target_i)
            low_rank_full_list.append(low_rank_i)

        targets_full = torch.stack(target_full_list)    # N × m × n
        low_ranks = torch.stack(low_rank_full_list)     # N × m × n
        l2_norms = torch.square(torch.norm(vectors.reshape(N, -1), p=2, dim=-1))

        # Step 3: Optimize X in Tucker subspace
        # Project targets into Tucker coords for initialization
        targets_proj = torch.stack([B.T @ targets_full[i] @ C for i in range(N)])
        X = torch.nn.Parameter(targets_proj.mean(dim=0))  # r2 × r3
        optimizer = torch.optim.Adam([X], lr=1e-3)

        # Pre-compute low_rank projections for efficiency: L_i @ C (m×n @ n×r3 = m×r3)
        # loss = ||(B@X@C^T - target_i) @ L_i^T||^2
        #      = ||B@X@C^T@L_i^T - target_i@L_i^T||^2
        # Pre-compute target_i @ L_i^T (m × m)
        target_L_list = [(targets_full[i] @ low_ranks[i].T).detach() for i in range(N)]

        for step in tqdm(range(iter_num), desc=f"Tucker-opt {param_name}", leave=False):
            m_full = B @ X @ C.T  # m × n
            loss = 0.0
            for i in range(N):
                proj = m_full @ low_ranks[i].T - target_L_list[i]
                loss = loss + torch.sum(proj ** 2) / l2_norms[i]

            if step % 50 == 0:
                print(f"Step {step}, loss: {loss.item():.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reconstruct final merging vector
        result = B @ X.data @ C.T
        return result.detach().to(original_dtype)

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_tucker_merging_vector(
                param_name, values, top_k=top_k, iter_num=iter_num
            )
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
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


def tucker_wudi2_cf_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                            scaling_coefficient: float = 1.0, top_k: int = 40):
    """
    Tucker subspace wudi2 with closed-form solution.

    Instead of 300-step Adam, solves X* = G @ H^{-1} exactly.
    The wudi2 loss is quadratic in X within the Tucker subspace,
    so the optimum is a linear system solve.
    """
    assert isinstance(scaling_coefficient, float)
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    def get_tucker_cf_vector(param_name, vectors, top_k=40):
        """Closed-form Tucker subspace merging."""
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)
        N, m, n = vectors.shape

        # Step 1: Build joint bases via pooled SVD (same as iterative Tucker)
        U_pool_list = []
        V_pool_list = []
        for i in range(N):
            Ui, Si, Vti = torch.linalg.svd(vectors[i], full_matrices=False)
            k = min(top_k, Si.shape[0])
            U_pool_list.append(Ui[:, :k])
            V_pool_list.append(Vti[:k, :].T)  # n × k

        U_pool = torch.cat(U_pool_list, dim=1)  # m × (N*k)
        V_pool = torch.cat(V_pool_list, dim=1)  # n × (N*k)

        # Orthogonalize via QR
        B, _ = torch.linalg.qr(U_pool)  # m × r2
        C, _ = torch.linalg.qr(V_pool)  # n × r3
        r2, r3 = B.shape[1], C.shape[1]
        print(f"  Tucker-CF: r2={r2}, r3={r3} (vs full {m}×{n}={m*n}), "
              f"compression={m*n/(r2*r3):.0f}x")

        # Step 2: Compute wudi2 targets and projection matrices
        average_vector = vectors.mean(dim=0)
        l2_norms = torch.sum(vectors.reshape(N, -1) ** 2, dim=-1)  # (N,)

        # Accumulate H (r3 × r3) and G (r2 × r3)
        H = torch.zeros(r3, r3, device=vectors.device, dtype=torch.float32)
        G = torch.zeros(r2, r3, device=vectors.device, dtype=torch.float32)

        for i in range(N):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N)

            # Cleaned target T_i (same as wudi2)
            u2_k = u2[:, :reduced_index_s]
            s2_k = s2[:reduced_index_s]
            v2_k = v2[:reduced_index_s, :]
            target_i = u2_k @ torch.diag(s2_k) @ v2_k + average_vector  # m × n

            # Low-rank projection L_i (same as wudi2)
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(m, n, device=s.device)
            min_dim = min(m, n)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_i = S_matrix @ v  # m × n

            n_i = l2_norms[i]

            # Efficient computation avoiding m×m intermediates:
            # Z_i = L_i @ C  (m × r3)
            Z_i = low_rank_i @ C  # m × r3

            # H += Z_i^T @ Z_i / n_i  (r3 × r3)
            H += Z_i.T @ Z_i / n_i

            # G += (B^T @ T_i) @ (L_i^T @ Z_i) / n_i  (r2 × r3)
            BtT = B.T @ target_i    # r2 × n
            LtZ = low_rank_i.T @ Z_i  # n × r3
            G += BtT @ LtZ / n_i

        # Step 3: Closed-form solution X* = G @ H^{-1}
        # Tikhonov regularization for rank-deficient H (esp. K/V layers)
        reg = 1e-6 * torch.trace(H) / r3
        H_reg = H + reg * torch.eye(r3, device=H.device, dtype=H.dtype)
        X_star = G @ torch.linalg.inv(H_reg)  # r2 × r3

        # Compute residual loss for diagnostics
        m_full = B @ X_star @ C.T
        loss = 0.0
        # Re-derive targets for loss computation
        for i in range(N):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N)
            u2_k = u2[:, :reduced_index_s]
            s2_k = s2[:reduced_index_s]
            v2_k = v2[:reduced_index_s, :]
            target_i = u2_k @ torch.diag(s2_k) @ v2_k + average_vector
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(m, n, device=s.device)
            min_dim = min(m, n)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_i = S_matrix @ v
            proj = m_full @ low_rank_i.T - target_i @ low_rank_i.T
            loss += torch.sum(proj ** 2) / l2_norms[i]
        print(f"  Closed-form loss: {loss.item():.6f}")

        result = B @ X_star @ C.T
        return result.detach().to(original_dtype)

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_tucker_cf_vector(
                param_name, values, top_k=top_k
            )
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
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


def mc_tucker_wudi2_cf_merging(merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                                scaling_coefficient: float = 1.0, beta: float = 0.5,
                                energy_ratio: float = 0.95, top_k: int = 40):
    """
    Route A+C: MC conflict suppression + Tucker closed-form.

    Combines micro-capability conflict resolution (Route A) with
    Tucker subspace closed-form solution (Route C).
    """
    assert isinstance(scaling_coefficient, float)
    models_to_merge_task_vectors = [
        TaskVector(pretrained_model=merged_model,
                  finetuned_model=model_to_merge,
                  exclude_param_names_regex=exclude_param_names_regex)
        for model_to_merge in models_to_merge
    ]

    def suppress_conflicts(vectors, beta=0.5, energy_ratio=0.95):
        """Suppress conflicting micro-capabilities across experts (same as mc_wudi2)."""
        N, m, n = vectors.shape
        device = vectors.device
        all_sigmas, all_us, all_vs = [], [], []
        expert_ids = []
        expert_k_ranges = []
        idx = 0
        for i in range(N):
            U_i, S_i, Vt_i = torch.linalg.svd(vectors[i], full_matrices=False)
            total_energy = torch.sum(S_i ** 2)
            cumulative = torch.cumsum(S_i ** 2, dim=0)
            k = torch.searchsorted(cumulative, energy_ratio * total_energy).item() + 1
            k = max(k, 1)
            k = min(k, S_i.shape[0])
            all_sigmas.append(S_i[:k])
            all_us.append(U_i[:, :k])
            all_vs.append(Vt_i[:k, :])
            expert_ids.extend([i] * k)
            expert_k_ranges.append((idx, idx + k))
            idx += k

        sigmas = torch.cat(all_sigmas)
        K = sigmas.shape[0]
        expert_ids_t = torch.tensor(expert_ids, device=device)
        U_cat = torch.cat(all_us, dim=1).T
        V_cat = torch.cat(all_vs, dim=0)
        u_overlap = U_cat @ U_cat.T
        v_overlap = V_cat @ V_cat.T
        sigma_outer = sigmas.unsqueeze(1) * sigmas.unsqueeze(0)
        I_matrix = u_overlap * v_overlap * sigma_outer

        suppression_factors = torch.ones(K, device=device)
        for i in range(N):
            start_i, end_i = expert_k_ranges[i]
            other_mask = expert_ids_t != i
            for a_local in range(end_i - start_i):
                a = start_i + a_local
                cross_vals = I_matrix[a] * other_mask.float()
                conflict_vals = torch.clamp(-cross_vals, min=0)
                conflict_score = conflict_vals.sum().item()
                if conflict_score > 0:
                    own_contribution = sigmas[a].item() ** 2
                    relative_conflict = conflict_score / (own_contribution + 1e-10)
                    suppression = max(0.0, 1.0 - beta * relative_conflict)
                    suppression_factors[a] = suppression

        cleaned = torch.zeros_like(vectors)
        for i in range(N):
            start_i, end_i = expert_k_ranges[i]
            suppressed_s = all_sigmas[i] * suppression_factors[start_i:end_i]
            cleaned[i] = all_us[i] @ torch.diag(suppressed_s) @ all_vs[i]

        n_suppressed = (suppression_factors < 1.0).sum().item()
        avg_suppression = suppression_factors.mean().item()
        print(f"  MC: K={K}, suppressed={n_suppressed}/{K} "
              f"({n_suppressed/K*100:.1f}%), avg_factor={avg_suppression:.3f}")
        del I_matrix, u_overlap, v_overlap, sigma_outer, U_cat, V_cat
        return cleaned

    def get_mc_tucker_cf_vector(param_name, vectors, beta=0.5, energy_ratio=0.95, top_k=40):
        """MC conflict suppression followed by Tucker closed-form."""
        original_dtype = vectors.dtype
        vectors = vectors.to(torch.float32)
        N, m, n = vectors.shape

        # Step 1: MC conflict suppression
        vectors = suppress_conflicts(vectors, beta=beta, energy_ratio=energy_ratio)

        # Step 2: Build Tucker bases from CLEANED vectors
        U_pool_list, V_pool_list = [], []
        for i in range(N):
            Ui, Si, Vti = torch.linalg.svd(vectors[i], full_matrices=False)
            k = min(top_k, Si.shape[0])
            U_pool_list.append(Ui[:, :k])
            V_pool_list.append(Vti[:k, :].T)

        B, _ = torch.linalg.qr(torch.cat(U_pool_list, dim=1))
        C, _ = torch.linalg.qr(torch.cat(V_pool_list, dim=1))
        r2, r3 = B.shape[1], C.shape[1]
        print(f"  MC-Tucker-CF: r2={r2}, r3={r3}, compression={m*n/(r2*r3):.0f}x")

        # Step 3: wudi2 targets from cleaned vectors
        average_vector = vectors.mean(dim=0)
        l2_norms = torch.sum(vectors.reshape(N, -1) ** 2, dim=-1)
        H = torch.zeros(r3, r3, device=vectors.device, dtype=torch.float32)
        G = torch.zeros(r2, r3, device=vectors.device, dtype=torch.float32)

        for i in range(N):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N)
            u2_k = u2[:, :reduced_index_s]
            s2_k = s2[:reduced_index_s]
            v2_k = v2[:reduced_index_s, :]
            target_i = u2_k @ torch.diag(s2_k) @ v2_k + average_vector
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(m, n, device=s.device)
            min_dim = min(m, n)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_i = S_matrix @ v
            n_i = l2_norms[i]
            Z_i = low_rank_i @ C
            H += Z_i.T @ Z_i / n_i
            BtT = B.T @ target_i
            LtZ = low_rank_i.T @ Z_i
            G += BtT @ LtZ / n_i

        # Step 4: Closed-form solution with Tikhonov regularization
        reg = 1e-6 * torch.trace(H) / r3
        H_reg = H + reg * torch.eye(r3, device=H.device, dtype=H.dtype)
        X_star = G @ torch.linalg.inv(H_reg)
        result = B @ X_star @ C.T

        # Diagnostics
        loss = 0.0
        for i in range(N):
            vector = vectors[i]
            u, s, v = torch.linalg.svd(vector, full_matrices=True)
            u2, s2, v2 = torch.linalg.svd(vector - average_vector, full_matrices=False)
            reduced_index_s = int(s.shape[0] / N)
            u2_k = u2[:, :reduced_index_s]
            s2_k = s2[:reduced_index_s]
            v2_k = v2[:reduced_index_s, :]
            target_i = u2_k @ torch.diag(s2_k) @ v2_k + average_vector
            s_mask = torch.zeros_like(s)
            s_mask[:reduced_index_s] = 1
            s = s * s_mask
            v_mask = torch.zeros_like(v)
            v_mask[:reduced_index_s, :] = 1
            v = v * v_mask
            S_matrix = torch.zeros(m, n, device=s.device)
            min_dim = min(m, n)
            S_matrix[:min_dim, :min_dim] = torch.diag_embed(s)
            low_rank_i = S_matrix @ v
            proj = result @ low_rank_i.T - target_i @ low_rank_i.T
            loss += torch.sum(proj ** 2) / l2_norms[i]
        print(f"  MC-Tucker-CF loss: {loss.item():.6f}")

        return result.detach().to(original_dtype)

    merged_task_vector_dict = {}

    for param_name in models_to_merge_task_vectors[0].task_vector_param_dict:
        if len(models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape) == 2 and "lm_head" not in param_name:
            print(f"Processing {param_name} with shape {models_to_merge_task_vectors[0].task_vector_param_dict[param_name].shape}")
            values = torch.stack([
                task_vector.task_vector_param_dict[param_name]
                for task_vector in models_to_merge_task_vectors
            ])
            merging_vector = get_mc_tucker_cf_vector(
                param_name, values, beta=beta, energy_ratio=energy_ratio, top_k=top_k
            )
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
        scaling_coefficient=scaling_coefficient
    )
    return merged_params


def merge_models(merge_method="wudi2", scaling_coefficient = 0.1, merged_model_name='default_merged_model', merged_list=[]):
    print("Start merging models...")
    base_model = models['a'].cuda()
    base_state_dict = base_model.state_dict()

    models_to_merge = []
    for k in merged_list:
        model = models[k].cuda()
        models_to_merge.append(model)
    
    exclude_param_names_regex = [
        'vision_model.*',
        '.*lm_head.*',
        '.*norm.*',
        '.*embed_tokens.*',
        '.*bias.*'
    ]

    if merge_method == "task_arithmetic":
        print("Running task_arithmetic...")
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "ties":
        print("Running ties_merging...")
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ta":
        print("Running Dare task_arithmetic...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = task_arithmetic(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "dare ties":
        print("Running Dare ties_merging...")
        weight_mask_rates = [0.2 for _ in range(len(models_to_merge))]
        with torch.no_grad():
            new_models_to_merge = models_to_merge
            for new_model_to_merge, weight_mask_rate in zip(new_models_to_merge, weight_mask_rates):
                # for each individual model, mask its weight
                masked_param_dict = mask_model_weights(finetuned_model=new_model_to_merge, pretrained_model=base_model,
                                                        exclude_param_names_regex=exclude_param_names_regex, weight_format="delta_weight",
                                                        weight_mask_rate=weight_mask_rate, use_weight_rescale=True, mask_strategy="random")
                copy_params_to_model(params=masked_param_dict, model=new_model_to_merge)
        
        merged_params = ties_merging(
            merged_model=base_model,
            models_to_merge=new_models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            param_value_mask_rate=0.8,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "svd":
        print("Running tsv_merging...")
        merged_params = svd_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "iso":
        print("Running iso_merging...")
        merged_params = iso_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi":
        print("Running wudi_merging...")
        merged_params = wudi_merging(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    elif merge_method == "wudi2":
        print("Running wudi v2...")
        merged_params = wudi_merging2(
            merged_model=base_model,
            models_to_merge=models_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
            scaling_coefficient=scaling_coefficient
        )
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")
    
    for key in merged_params:
        if key in base_state_dict:
            base_state_dict[key] = merged_params[key]
    base_model.load_state_dict(base_state_dict)
    base_model = base_model.cuda()

    output_path = merged_model_name
    print(f"Saving model to {output_path}")
    base_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    for model in models_to_merge:
        del model
    torch.cuda.empty_cache()
    return base_model

if __name__ == "__main__":
    #####################################################################
    path_a = 'OpenGVLab/InternVL2_5-1B'
    path_b = 'yongxianwei/InternVL2_5-1B_OCR'
    path_c = 'yongxianwei/InternVL2_5-1B_VQA'
    path_d = 'yongxianwei/InternVL2_5-1B_Geometry'
    path_e = 'yongxianwei/InternVL2_5-1B_Chart'
    path_f = 'yongxianwei/InternVL2_5-1B_Grounding'

    tokenizer = AutoTokenizer.from_pretrained(path_a, trust_remote_code=True, use_fast=False)
    models = {
        'a': AutoModel.from_pretrained(path_a, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'b': AutoModel.from_pretrained(path_b, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'c': AutoModel.from_pretrained(path_c, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'd': AutoModel.from_pretrained(path_d, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'e': AutoModel.from_pretrained(path_e, torch_dtype=torch.float16, trust_remote_code=True).eval(),
        'f': AutoModel.from_pretrained(path_f, torch_dtype=torch.float16, trust_remote_code=True).eval(),
    }
    model = merge_models(merged_model_name='merged_exclude_ocr', merged_list=['c', 'd', 'e', 'f'])
    #####################################################################
    # set the max number of tiles in `max_num`
    pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # pure-text conversation
    question = 'Hello, who are you?'
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'Can you tell me a story?'
    response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    # single-image single-round conversation
    question = '<image>\nPlease describe the image shortly.'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')

    # single-image multi-round conversation
    question = '<image>\nPlease describe the image in detail.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'Please write a poem according to the image.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    # multi-image multi-round conversation, combined images
    pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
    pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    question = '<image>\nDescribe the two images in detail.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'What are the similarities and differences between these two images.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    # multi-image multi-round conversation, separate images
    pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
    pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    question = 'What are the similarities and differences between these two images.'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    # batch inference, single image per sample
    pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.float16).cuda()
    pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.float16).cuda()
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
    responses = model.batch_chat(tokenizer, pixel_values,
                                num_patches_list=num_patches_list,
                                questions=questions,
                                generation_config=generation_config)
    for question, response in zip(questions, responses):
        print(f'User: {question}\nAssistant: {response}')
