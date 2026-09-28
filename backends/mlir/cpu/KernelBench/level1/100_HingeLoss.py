import torch
import torch.nn as nn


class Model(nn.Module):
    mlir_pipeline = "default"
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
