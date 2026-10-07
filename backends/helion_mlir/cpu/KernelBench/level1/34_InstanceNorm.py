import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _instance_norm(x: torch.Tensor, eps: hl.constexpr) -> torch.Tensor:
    """Normalization of each row of ``x`` ``[B*C, H*W]`` in passes over column
    chunks: mean, variance, then the output. A tile of few rows keeps them in
    L2 for the later passes."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].to(torch.float32)
        mean = acc.sum(-1, keepdim=True) * (1.0 / n)
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            d = x[tile_m, tile_n].to(torch.float32) - mean
            acc = acc + d * d
        rstd = torch.rsqrt(acc.sum(-1, keepdim=True) * (1.0 / n) + eps)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n].to(torch.float32)
            out[tile_m, tile_n] = ((v - mean) * rstd).to(x.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_n: int, rows: int) -> helion.Kernel:
    # The column block is registered first.
    key = (block_n, rows)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _instance_norm,
            backend="mlir",
            config=helion.Config(block_sizes=[block_n, rows]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.inorm
        if norm.affine or norm.track_running_stats:
            raise ValueError("only InstanceNorm without affine/running statistics")
        b, c = x.shape[:2]
        x2 = x.contiguous().view(b * c, -1)
        m, n = x2.shape
        # Masked lanes would enter the centered sums: chunks divide the row.
        # 256-wide chunks swept best; rows per tile stay in L2 between passes
        # (<= 256 KiB of f32), at most 16, and every thread gets a tile.
        block_n = math.gcd(n, 256)
        rows = max(1, min(16, (1 << 16) // n, m // torch.get_num_threads()))
        return _kernel(block_n, rows)(x2, hl.constexpr(norm.eps)).view_as(x)
