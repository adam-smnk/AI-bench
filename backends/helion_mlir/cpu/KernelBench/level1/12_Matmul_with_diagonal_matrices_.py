import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 1024]))
def _diag_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``diag(a) @ b``: row ``n`` of ``b`` scaled by ``a[n]``."""
    n, m = b.shape
    out = torch.empty_like(b)
    for tile_n, tile_m in hl.tile([n, m]):
        scale = a[tile_n].to(torch.float32)
        out[tile_n, tile_m] = (scale[:, None] * b[tile_n, tile_m].to(torch.float32)).to(
            b.dtype
        )
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _diag_matmul(A, B)
