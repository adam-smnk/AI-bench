import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scaling_factor,
        hardtanh_min,
        hardtanh_max,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None
        scaling_factor = hl.constexpr(scaling_factor)
        hardtanh_min = hl.constexpr(hardtanh_min)
        hardtanh_max = hl.constexpr(hardtanh_max)

        def epilogue(x):
            x = torch.clamp(x * scaling_factor, hardtanh_min, hardtanh_max)
            return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))

        self._epilogue = epilogue

    def forward(self, x):
        x, self._linear_cache = linear(
            x, self.linear, self._epilogue, self._linear_cache
        )
        return x
