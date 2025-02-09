import torch


def log_loss(pred, target):
    return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

