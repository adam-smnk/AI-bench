import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None
        multiplier = hl.constexpr(multiplier)
        negative_slope = hl.constexpr(negative_slope)

        def epilogue(x):
            x = x * multiplier
            return torch.where(x >= 0, x, x * negative_slope)

        self._epilogue = epilogue

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, self._epilogue, self._linear_cache
        )
        return x
