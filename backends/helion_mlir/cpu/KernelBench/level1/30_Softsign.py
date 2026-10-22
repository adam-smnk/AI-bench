import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[4, 4096]))
def _softsign(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v / (1.0 + torch.abs(v))).to(x.dtype)
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _softsign(x)
