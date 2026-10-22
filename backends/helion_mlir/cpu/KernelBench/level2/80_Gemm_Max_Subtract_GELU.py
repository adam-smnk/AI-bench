import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16]))
def _row_max_sub_gelu(y: torch.Tensor) -> torch.Tensor:
    """``gelu(v - mean(v, dim=1))`` of the row maxima ``v`` ``[M, 1]`` of ``y``."""
    m, n = y.shape
    hl.specialize(n)
    out = torch.empty((m, 1), dtype=y.dtype, device=y.device)
    for tile_m in hl.tile(m):
        v = torch.amax(y[tile_m, :].to(torch.float32), dim=-1, keepdim=True)
        # The mean over v's singleton dim 1 is v.
        out[tile_m, :] = _gelu(v - v).to(y.dtype)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[64]))
def _col_max(y: torch.Tensor) -> torch.Tensor:
    """Column maxima of ``[M, N]`` as ``[1, N]``."""
    m, n = y.shape
    hl.specialize(m)
    out = torch.empty((1, n), dtype=y.dtype, device=y.device)
    for tile_n in hl.tile(n):
        out[0, tile_n] = torch.amax(y[:, tile_n], dim=0)
    return out


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1]))
def _sub_mean_gelu(v: torch.Tensor) -> torch.Tensor:
    """``gelu(v - mean(v, dim=1))`` of ``[R, N]``, one row per tile."""
    r, n = v.shape
    hl.specialize(n)
    out = torch.empty_like(v)
    for tile_r in hl.tile(r):
        row = v[tile_r, :].to(torch.float32)
        out[tile_r, :] = _gelu(row - row.mean(-1, keepdim=True)).to(v.dtype)
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        if self.max_dim == 1:
            return _row_max_sub_gelu(y)
        if self.max_dim == 0:
            return _sub_mean_gelu(_col_max(y))
        raise ValueError(f"max_dim must be 0 or 1, got {self.max_dim}")
