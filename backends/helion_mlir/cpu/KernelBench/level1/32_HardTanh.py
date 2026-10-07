import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[4, 4096]))
def _hardtanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = torch.clamp(x[tile], -1.0, 1.0)
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _hardtanh(x)
