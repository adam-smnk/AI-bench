import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_sum


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, torch.sigmoid, self._linear_cache
        )
        return row_sum(x)
