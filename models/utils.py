"""
General Model utils
"""

import pandas as pd
from models.model_shell import ModelShell

def analyze_shared_parameters(model1, model2):
    shared_params = 0
    total_params1 = 0
    total_params2 = 0
    
    # Create dictionaries of all parameters for each model
    params1 = {id(p): p for p in model1.parameters()}
    params2 = {id(p): p for p in model2.parameters()}
    
    # Find shared parameters
    shared_ids = set(params1.keys()) & set(params2.keys())
    
    # Count parameters
    for pid in params1:
        total_params1 += params1[pid].numel()
        if pid in shared_ids:
            shared_params += params1[pid].numel()
    
    for pid in params2:
        total_params2 += params2[pid].numel()
    
    return shared_params, (total_params1 + total_params2 - shared_params)


def print_model_stats(model: ModelShell):
    """
    Print relevant model statistics, including the number of parameters
    with and without embeddings for a given PyTorch model, formatted for better readability.
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Check if the parameters are shared
    
    _, lm_head_and_embeddings_params = analyze_shared_parameters(model.embedding_model, model.model_head)
    core_model_params = total_params - lm_head_and_embeddings_params

    # Format the numbers for better readability
    def format_number(n):
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)

    # Prepare the data
    data = {
        "Component": ["Total", "LM Head + Embeddings", "Core Model"],
        "Parameters": [
            format_number(total_params),
            format_number(lm_head_and_embeddings_params),
            format_number(core_model_params),
        ],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Print the table
    print(df.to_string(index=False))


def log_parameter_mse(model1, model2):
    """
    Compute and log the MSE between the current model parameters and the initial model.
    This function computes the mean squared error over all parameters in each group:
    - Entire model
    - core_model
    - embedding_model
    - model_head
    """
    # Variables to accumulate squared error sums and total element counts.
    sum_total, num_total = 0.0, 0
    sum_core, num_core = 0.0, 0
    sum_embed, num_embed = 0.0, 0
    sum_head, num_head = 0.0, 0

    # Assumes the parameter ordering/names are identical between self.model and self.initial_model.
    for (name, param), (_, init_param) in zip(model1.named_parameters(), model2.named_parameters()):
        diff = param - init_param
        sum_sq = diff.pow(2).sum().item()
        n = diff.numel()

        sum_total += sum_sq
        num_total += n

        if "core_model" in name:
            sum_core += sum_sq
            num_core += n
        elif "embedding_model" in name:
            sum_embed += sum_sq
            num_embed += n
        elif "model_head" in name:
            sum_head += sum_sq
            num_head += n

    assert num_total == num_core + num_embed + num_head

    mse_total = sum_total / num_total if num_total > 0 else 0.0
    mse_core = sum_core / num_core if num_core > 0 else 0.0
    mse_embed = sum_embed / num_embed if num_embed > 0 else 0.0
    mse_head = sum_head / num_head if num_head > 0 else 0.0

    log_dict = {
        "param_mse/whole_model": mse_total,
        "param_mse/core_model": mse_core,
        "param_mse/embedding_model": mse_embed,
        "param_mse/model_head": mse_head,
    }
    print(f"Parameter MSE: {log_dict}")
    return log_dict
