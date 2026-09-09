import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None
        subtract_value = hl.constexpr(subtract_value)
        multiply_value = hl.constexpr(multiply_value)

        def epilogue(x):
            return torch.relu((x - subtract_value) * multiply_value)

        self._epilogue = epilogue

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, self._epilogue, self._linear_cache
        )
        return x
