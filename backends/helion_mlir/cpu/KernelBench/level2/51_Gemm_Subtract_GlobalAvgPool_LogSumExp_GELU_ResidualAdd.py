import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16]))
def _mean_gelu_residual(
    y: torch.Tensor, subtract: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """``x + gelu(mean(y - subtract))`` per row of ``y`` ``[M, N]``; the
    logsumexp over the singleton mean dim is the identity."""
    m, n = y.shape
    _, k = x.shape
    hl.specialize(n)
    hl.specialize(k)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        diff = y[tile_m, :].to(torch.float32) - subtract[:].to(torch.float32)
        mean = diff.mean(-1, keepdim=True)
        gelu = mean * 0.5 * (1.0 + torch.erf(mean * 0.7071067811865476))
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) + gelu).to(x.dtype)
    return out


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.gemm, cache=self._linear_cache)
        return _mean_gelu_residual(y, self.subtract, x)
