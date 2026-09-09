from types import SimpleNamespace

import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        projection = torch.zeros(32, hidden_size)
        projection[0, :] = 1.0
        self._sum = SimpleNamespace(weight=projection, bias=None, out_features=32)
        self._linear_cache = None
        self._sum_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, torch.sigmoid, self._linear_cache
        )
        x, self._sum_cache = linear(x, self._sum, cache=self._sum_cache)
        return x[:, :1]
