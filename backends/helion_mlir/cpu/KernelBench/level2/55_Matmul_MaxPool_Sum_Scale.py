import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[2, 1024]))
def _pool_sum(y3: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
    """``scale * sum(max(pool window))`` per row of ``y3``, rows of pool
    windows ``[M, N/k, k]``."""
    m, windows, k = y3.shape
    hl.specialize(k)
    out = torch.empty((m,), dtype=y3.dtype, device=y3.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_w in hl.tile(windows):
            pooled = torch.amax(y3[tile_m, tile_w, :].to(torch.float32), dim=-1)
            acc = acc + pooled.sum(-1)
        out[tile_m] = (acc * scale).to(y3.dtype)
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.matmul, cache=self._linear_cache)
        m, n = y.shape
        k = self.kernel_size
        # MaxPool1d drops the incomplete last window.
        y3 = y[:, : n // k * k].view(m, n // k, k)
        return _pool_sum(y3, hl.constexpr(self.scale_factor))
