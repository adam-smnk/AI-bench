import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _frobenius_normalize(x2: torch.Tensor) -> torch.Tensor:
    """``x2 / ||x2||_F`` of ``[R, L]``: per-row sums of squares, then the
    output scaled by the inverse norm of all rows."""
    r, width = x2.shape
    hl.specialize(r)
    hl.specialize(width)
    partial = torch.empty([r], dtype=torch.float32, device=x2.device)
    out = torch.empty_like(x2)
    for tile_r in hl.tile(r):
        v = x2[tile_r, :].to(torch.float32)
        partial[tile_r] = (v * v).sum(-1)
    hl.barrier()
    for tile_r, tile_l in hl.tile([r, width]):
        inv = torch.rsqrt(partial[:].sum(-1))
        out[tile_r, tile_l] = (x2[tile_r, tile_l].to(torch.float32) * inv).to(x2.dtype)
    return out


_KERNELS: dict[tuple[int, ...], helion.Kernel] = {}


def _kernel(block_sizes: list[int]) -> helion.Kernel:
    key = tuple(block_sizes)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _frobenius_normalize,
            backend="mlir",
            config=helion.Config(block_sizes=block_sizes),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.numel()
        # Rows of the largest power-of-two divisor up to 16384 elements (64 KiB of
        # f32): one partial sum per row, whole rows as output tiles.
        row = min(total & -total, 16384)
        x2 = x.contiguous().view(-1, row)
        return _kernel([1, 1, row])(x2).view_as(x)
