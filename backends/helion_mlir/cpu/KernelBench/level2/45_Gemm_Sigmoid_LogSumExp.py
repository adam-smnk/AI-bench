import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_sum


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._linear1_cache = None
        self._linear2_cache = None

    def forward(self, x):
        x, self._linear1_cache = linear(
            x, self.linear1, torch.sigmoid, self._linear1_cache
        )
        x, self._linear2_cache = linear(x, self.linear2, torch.exp, self._linear2_cache)
        return row_sum(x, torch.log)[:, 0]
