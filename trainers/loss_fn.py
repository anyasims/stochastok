"""Loss Functions for Training

Each loss function should take in output of a model and the target labels
and return the loss value. This need not be the logits."""


import torch


def cross_entropy_loss_fn(logits, y, mask=None, mean_with_mask_length=True):
    """Cross Entropy Loss Function"""
    batch_size, sequence_length, vocab_size = logits.size()
    assert y.shape == (batch_size, sequence_length), f"{y.shape=}, {logits.shape=}"
    assert mask is None or mask.size() == (batch_size, sequence_length)
    # reshaping since torch.nn.functional.cross_entropy only supports one batch dimension
    logits_ = logits.view(batch_size * sequence_length, vocab_size)
    y_ = y.view(batch_size * sequence_length)
    losses = torch.nn.functional.cross_entropy(logits_, y_, reduction="none").reshape(batch_size, sequence_length)
    if mask is None:
        loss = losses.mean(-1)
    elif mean_with_mask_length:
        loss = (losses * mask).sum(-1) / mask.sum(-1)
    else:
        loss = (losses * mask).mean(-1)
    with torch.no_grad():
        logps = torch.nn.functional.log_softmax(logits, dim=-1) 
        entropy = -torch.sum(logps * torch.exp(logps), dim=-1)
        metrics = {
            "loss": loss.mean().item(),
            "perplexity": torch.exp(loss).mean().item(),
            "entropy": entropy.mean().item(),
        }
    metrics = {f"loss/{k}": v for k, v in metrics.items()}
    return loss.mean(), metrics

