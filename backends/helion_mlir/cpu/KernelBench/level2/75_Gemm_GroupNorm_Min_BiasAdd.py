import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16]))
def _group_norm_min(
    x3: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: hl.constexpr
) -> torch.Tensor:
    """Row minimum of ``GroupNorm(x)`` of ``x3`` ``[M, G, C/G]``; ``weight`` and
    ``bias`` are ``[G, C/G]``. A tile holds whole rows."""
    m, groups, group_size = x3.shape
    hl.specialize(groups)
    hl.specialize(group_size)
    out = torch.empty((m,), dtype=x3.dtype, device=x3.device)
    for tile_m in hl.tile(m):
        y = x3[tile_m, :, :].to(torch.float32)
        mean = y.mean(-1, keepdim=True)
        centered = y - mean
        var = (centered * centered).mean(-1, keepdim=True)
        y = centered * torch.rsqrt(var + eps) * weight[:, :].to(torch.float32) + bias[
            :, :
        ].to(torch.float32)
        out[tile_m] = torch.amin(torch.amin(y, dim=-1), dim=-1).to(x3.dtype)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[64, 256]))
def _outer_add(col: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """``col[:, None] + row[None, :]`` of ``[C]`` and ``[M]``."""
    (c,) = col.shape
    (m,) = row.shape
    out = torch.empty((c, m), dtype=row.dtype, device=row.device)
    for tile_c, tile_m in hl.tile([c, m]):
        out[tile_c, tile_m] = (
            col[tile_c].to(torch.float32)[:, None]
            + row[tile_m].to(torch.float32)[None, :]
        ).to(row.dtype)
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._linear_cache = None

    def forward(self, x):
        bias_shape = tuple(self.bias.shape)
        c = self.bias.numel()
        if len(bias_shape) != 4 or bias_shape[1] != c:
            raise ValueError(f"expected a (1, C, 1, 1) bias, got {bias_shape}")
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        m, n = y.shape
        gn = self.group_norm
        channels = (gn.num_groups, n // gn.num_groups)
        mins = _group_norm_min(
            y.view(m, *channels),
            gn.weight.view(channels),
            gn.bias.view(channels),
            hl.constexpr(gn.eps),
        )
        # [M, 1] + [1, C, 1, 1] -> [1, C, M, 1].
        return _outer_add(self.bias.view(c), mins).view(1, c, m, 1)
