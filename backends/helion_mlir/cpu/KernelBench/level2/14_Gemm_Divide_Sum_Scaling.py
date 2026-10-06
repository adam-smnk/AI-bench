from types import SimpleNamespace

import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_sum


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self._matmul_cache = None
        divide = hl.constexpr(2.0)
        scaling_factor = hl.constexpr(scaling_factor)

        def divide_epilogue(x):
            return x / divide

        def scale_epilogue(x):
            return x * scaling_factor

        self._divide_epilogue = divide_epilogue
        self._scale_epilogue = scale_epilogue

    def forward(self, x):
        weight = SimpleNamespace(
            weight=self.weight,
            bias=None,
            out_features=self.weight.shape[0],
        )
        x, self._matmul_cache = linear(
            x, weight, self._divide_epilogue, self._matmul_cache
        )
        return row_sum(x, self._scale_epilogue)
