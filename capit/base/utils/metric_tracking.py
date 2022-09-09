import torch


def compute_accuracy(logits, targets):
    acc = targets == logits.argmax(-1)
    return torch.mean(acc.type(torch.float32)) * 100
