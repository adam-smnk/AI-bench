from types import SimpleNamespace

import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        projection = torch.zeros(32, output_size)
        projection[0, :] = 1.0
        self._sum = SimpleNamespace(weight=projection, bias=None, out_features=32)
        self._linear1_cache = None
        self._linear2_cache = None
        self._sum_cache = None

    def forward(self, x):
        x, self._linear1_cache = linear(
            x, self.linear1, torch.sigmoid, self._linear1_cache
        )
        x, self._linear2_cache = linear(x, self.linear2, torch.exp, self._linear2_cache)
        x, self._sum_cache = linear(x, self._sum, torch.log, self._sum_cache)
        return x[:, 0]
