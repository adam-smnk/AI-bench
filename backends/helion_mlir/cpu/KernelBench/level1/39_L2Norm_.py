import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """``x / ||x||_2`` per row, in passes over column chunks. A tile of few
    rows keeps them in L2 for the second pass."""
    m, n = x.shape
    hl.specialize(n)
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n].to(torch.float32)
            acc = acc + v * v
        inv = torch.rsqrt(acc.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n].to(torch.float32) * inv).to(
                x.dtype
            )
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_n: int, rows: int) -> helion.Kernel:
    # The column block is registered first.
    key = (block_n, rows)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _l2_normalize,
            backend="mlir",
            config=helion.Config(block_sizes=[block_n, rows]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        # Rows per tile: few enough to stay in L2 between passes (<= 256 KiB of
        # f32), at most 16, and every thread gets a tile; chunks up to 4096 wide
        # (a ragged last chunk is fine: masked lanes load 0, neutral for x^2).
        rows = max(1, min(16, (1 << 16) // n, m // torch.get_num_threads()))
        return _kernel(min(n, 4096), rows)(x)
