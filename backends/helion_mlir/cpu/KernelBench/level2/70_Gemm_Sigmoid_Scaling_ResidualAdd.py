import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, in_features, hidden_size, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_size)
        self._linear_cache = None
        scaling_factor = hl.constexpr(scaling_factor)

        def epilogue(x):
            return torch.sigmoid(x) * scaling_factor + x

        self._epilogue = epilogue

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, self._epilogue, self._linear_cache
        )
        return x
