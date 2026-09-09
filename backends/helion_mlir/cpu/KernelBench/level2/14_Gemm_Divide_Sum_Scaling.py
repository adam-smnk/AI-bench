from types import SimpleNamespace

import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        projection = torch.zeros(32, hidden_size)
        projection[0, :] = 1.0
        self._sum = SimpleNamespace(weight=projection, bias=None, out_features=32)
        self._matmul_cache = None
        self._sum_cache = None
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
        x, self._sum_cache = linear(x, self._sum, self._scale_epilogue, self._sum_cache)
        return x[:, :1]
