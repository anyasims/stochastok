"""
A collection of metrics for evaluating models
"""

import torch

from trainers.utils import aggregate_value

def accuracy_metric(confidences):
    """
    Calculate the accuracy of the model over a path_prob
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        accuracy: float of accuracy
    """
    _, predicted = torch.max(confidences, 1)
    ## aggregate the tensor values
    return aggregate_value((predicted == 0).float().mean())


def path_confidence(confidences):
    """
    Calculate the path confidence of the model.
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        path_confidence: float of path confidences
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0]
    ## aggregate the tensor values
    return aggregate_value(softmaxed.mean())

def ground_confidence(confidences):
    """
    Calculate the confidence of the model on the ground truth
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Returns:
        ground_confidence: float of confidence on ground truth
    See: https://arxiv.org/pdf/2406.04391 - this is equivalent to
    $$P_\\theta^{\\text{Choices}}(\\text{Ground Truth})$$ over the
    Path probabilities. (takeaway 3)
    """
    return confidences[:, 0].mean()

def num_options(confidences):
    """
    Calculate the number of options
    """
    return confidences.shape[1]

def normalized_accuracy(confidences):
    """
    Accuracy of model normalized so that 0 is random guessing and 1 is perfect accuracy
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        accuracy: float of accuracy
    """
    _, predicted = torch.max(confidences, 1)
    accuracy = (predicted == 0).float().mean()
    num_options = confidences.shape[1]
    normalized_accuracy = (accuracy * num_options - 1) / (num_options - 1)
    ## aggregate the tensor values
    return aggregate_value(normalized_accuracy)

def normalized_path_confidence(confidences):
    """
    Path confidence normalized so that 0 is all having the same confidence and 1 is much higher confidence on the ground truth
    Assume that the ground truth is the first element in the list
    Args:
        confidences: (B, N) tensor of confidences
    Outputs:
        path_confidence: float of path confidences
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0].mean()
    num_options = confidences.shape[1]
    normalized_path_confidence = (softmaxed * num_options - 1) / (num_options - 1)
    ## aggregate the tensor values
    return aggregate_value(normalized_path_confidence)


MCQ_METRIC_DICT = {
    "accuracy": accuracy_metric,
    "path_confidence": path_confidence,
    "ground_confidence": ground_confidence,
    "num_options": num_options,
    "normalized_accuracy": normalized_accuracy,
    "normalized_path_confidence": normalized_path_confidence,
    }
