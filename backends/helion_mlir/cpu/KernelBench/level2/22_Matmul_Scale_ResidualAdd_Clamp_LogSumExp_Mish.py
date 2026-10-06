import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[16]))
def _logsumexp_mish(y: torch.Tensor) -> torch.Tensor:
    """``l * mish(l)`` of the row logsumexp ``l`` of ``[M, N]``, as ``[M, 1]``."""
    m, n = y.shape
    hl.specialize(n)
    out = torch.empty((m, 1), dtype=y.dtype, device=y.device)
    for tile_m in hl.tile(m):
        row = y[tile_m, :].to(torch.float32)
        top = torch.amax(row, dim=-1, keepdim=True)
        lse = top + torch.log(torch.exp(row - top).sum(-1, keepdim=True))
        mish = lse * torch.tanh(torch.log1p(torch.exp(lse)))
        out[tile_m, :] = (lse * mish).to(y.dtype)
    return out


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self._linear_cache = None
        # x * scale_factor, then x + x.
        scale = hl.constexpr(2.0 * scale_factor)
        lo = hl.constexpr(clamp_min)
        hi = hl.constexpr(clamp_max)

        def epilogue(x):
            return torch.clamp(x * scale, lo, hi)

        self._epilogue = epilogue

    def forward(self, x):
        y, self._linear_cache = linear(
            x, self.matmul, self._epilogue, self._linear_cache
        )
        return _logsumexp_mish(y)
