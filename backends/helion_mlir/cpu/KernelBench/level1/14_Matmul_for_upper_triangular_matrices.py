import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16, 1024]))
def _triu_(x: torch.Tensor) -> torch.Tensor:
    """Zero the part of ``x`` below the diagonal, in place."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        below = tile_n.index[None, :] < tile_m.index[:, None]
        x[tile_m, tile_n] = torch.where(below, 0.0, x[tile_m, tile_n])
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _triu_(matmul(A, B))
