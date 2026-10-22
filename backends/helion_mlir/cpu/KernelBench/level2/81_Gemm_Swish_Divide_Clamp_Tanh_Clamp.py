import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


def _epilogue(x):
    x = x * torch.sigmoid(x)
    x = torch.clamp(x / 2.0, -1.0, 1.0)
    return torch.clamp(torch.tanh(x), -1.0, 1.0)


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(x, self.linear, _epilogue, self._linear_cache)
        return x
