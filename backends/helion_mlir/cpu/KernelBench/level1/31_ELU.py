import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[4, 4096]))
def _elu(x: torch.Tensor, alpha: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = torch.where(v > 0, v, alpha * (torch.exp(v) - 1.0)).to(x.dtype)
    return out


class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _elu(x, hl.constexpr(self.alpha))
