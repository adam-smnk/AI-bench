import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import group_norm
from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(
        self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max
    ):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self._linear_cache = None
        lo = hl.constexpr(hardtanh_min)
        hi = hl.constexpr(hardtanh_max)

        def epilogue(y, _):
            return torch.clamp(y, lo, hi)

        self._epilogue = epilogue

    def forward(self, x):
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        gn = self.group_norm
        return group_norm(y, gn.num_groups, gn.weight, gn.bias, gn.eps, self._epilogue)
