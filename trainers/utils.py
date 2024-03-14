"""Utilities for the trainer"""

import importlib
from prettytable import PrettyTable
import inspect
import os
import pkgutil

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets

import torch.distributed as dist

def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_folder_structure(path_config):
    """
    Create all necessary folders for training.
    """
    if not os.path.exists(path_config["data_dir"]):
        os.makedirs(path_config["data_dir"])

    if not os.path.exists(path_config["checkpoint_dir"]):
        os.makedirs(path_config["checkpoint_dir"])



def get_classes_from_module(module_name):
    """
    Get a list of classes defined in a module or package.

    Args:
        module_name (str): The name of the module or package.

    Returns:
        list: A list of classes defined in the module or package.
    """
    module = importlib.import_module(module_name)
    classes = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if inspect.getmodule(obj) == module:
            classes.append(obj)

    return classes


def get_classes_from_package(package_name):
    """
    Get a list of classes defined in a package and its subpackages.

    Args:
        package_name (str): The name of the package.

    Returns:
        list: A list of classes defined in the package and its subpackages.
    """
    package = importlib.import_module(package_name)
    classes = get_classes_from_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        classes.extend(get_classes_from_module(module_name))

    return classes


def register_backward_hooks(tensor, module_name):
    """Registers hooks to profile the backward pass of a tensor."""
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

        def backward_hook(grad):
            with torch.autograd.profiler.record_function(f"{module_name}.backward"):
                return grad

        tensor.register_hook(backward_hook)


def profilize(model, classes=None):
    """Recursively add hooks to the model for recording PyTorch profiler traces with module names"""
    if classes is None:
        classes = get_classes_from_package("models")
        classes += get_classes_from_package("models.components.layers")
        print(f"Found classes for profiling: {classes}")

    for module in model.children():
        if isinstance(module, torch.nn.Module):
            profilize(module, classes=classes)
        if isinstance(module, torch.nn.ModuleDict):
            for sub_module in module.values():
                profilize(sub_module, classes=classes)
        if isinstance(module, torch.nn.ModuleList):
            for sub_module in module:
                profilize(sub_module, classes=classes)

    if (
        hasattr(model, "forward")
        and any(isinstance(model, cls) for cls in classes)
        and not hasattr(model, "old_forward")
    ):
        model.old_forward = model.forward
        print(f"added forward profiling wrapper for {model.__class__.__name__}")

        def forward_wrapper(*args, **kwargs):
            nested_module_name = model.__class__.__name__
            with torch.autograd.profiler.record_function(
                f"{nested_module_name}.forward"
            ):
                outputs = model.old_forward(*args, **kwargs)
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    register_backward_hooks(output, nested_module_name)
            else:
                register_backward_hooks(outputs, nested_module_name)
            return outputs

        model.forward = forward_wrapper

def is_dist():
    """
    Check if the current process is distributed.
    """
    return dist.is_initialized()

def aggregate_value(value, device = torch.device("cuda")): 
    """
    Since using DDP, calculation of metrics happen across all GPUs. 
    This function aggregate the loss across all GPUs. 
    """
    if not is_dist():
        return value
    all_loss = torch.tensor([value], device=device)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    return all_loss.item() / dist.get_world_size()
    # return value

def init_print_override():
    '''
    Overriding the print function is useful when running DDP. 
    This way, only rank 0 prints to the console.
    '''
    import builtins as __builtin__
    
    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv('GLOBAL_RANK') == '0':
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print

def restore_print_override(original_print):
    '''
    Restore the original print function.
    '''
    import builtins as __builtin__
    __builtin__.print = original_print




# Function to print evaluation results and benchmark results
def print_evaluation_results(iter_num, eval_results, benchmark_results):
    # Loss, Perplexity
    headers = ['Metric', 'Value']
    table = PrettyTable(headers)
    for metric, value in eval_results.items():
        row = [metric, value]
        table.add_row(row)
    print(f"Iteration {iter_num}")
    print(table)

    # MCQ Benchmark Results
    if "mcq" in benchmark_results:
        benchmark_table = PrettyTable([
            'Benchmark',
            'Accuracy',
            "Path Conf.",
            "Ground Conf.",
            "Num Options",
            "Normalized Accuracy",
            "Normalized Path Conf.",
            ])
        for benchmark, value in benchmark_results["mcq"].items():
            benchmark_table.add_row([
                f"{benchmark}", 
                f"{value['accuracy']:.3f}",
                f"{value['path_confidence']:.3f}",
                f"{value['ground_confidence']:.3f}",
                value['num_options'],
                f"{value['normalized_accuracy']:.3f}",
                f"{value['normalized_path_confidence']:.3f}",
            ])
        print("Benchmark Results")
        if len(benchmark_table._rows) > 0:
            print(benchmark_table)

    # Math Generation Benchmark Results
    if "math_generation" in benchmark_results:
        table = PrettyTable(['Split', 'BaseTok Acc.', 'StochasTok Acc.', 'CharacterTok Acc.'])
        for split, metrics in benchmark_results["math_generation"].items():
            table.add_row([split, metrics["base/answer_found"], metrics["stochastok/answer_found"], metrics["character/answer_found"]])
        print("Math Generation Results")
        print(table)


