import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """Row softmax in passes over column chunks: maxima, sums of exponentials,
    then the output. A tile of few rows keeps them in L2 for the later passes."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n].to(torch.float32))
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            e = torch.exp(x[tile_m, tile_n].to(torch.float32) - row_top)
            out[tile_m, tile_n] = (e * inv).to(x.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_n: int, rows: int) -> helion.Kernel:
    # The column block is registered first.
    key = (block_n, rows)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _softmax, backend="mlir", config=helion.Config(block_sizes=[block_n, rows])
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        # Rows per tile: few enough to stay in L2 between passes (<= 1 MiB of f32),
        # at most 16, and every thread gets a tile; 1024-wide chunks swept best.
        rows = max(1, min(16, (1 << 18) // n, m // torch.get_num_threads()))
        # Masked lanes would enter the maxima and sums: chunks divide the row.
        return _kernel(math.gcd(n, 1024), rows)(x)
