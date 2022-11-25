import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    loss = torch.nn.CrossEntropyLoss()
    return loss(output, target)
