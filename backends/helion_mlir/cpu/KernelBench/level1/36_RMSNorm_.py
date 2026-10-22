import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1, 256]))
def _rms_norm(x3: torch.Tensor, eps: hl.constexpr) -> torch.Tensor:
    """RMS normalization of ``x3`` ``[B, C, H*W]`` over the channels ``C``."""
    b, c, hw = x3.shape
    hl.specialize(c)
    out = torch.empty_like(x3)
    for tile_b, tile_l in hl.tile([b, hw]):
        v = x3[tile_b, :, tile_l].to(torch.float32)
        rms = torch.rsqrt((v * v).sum(1, keepdim=True) * (1.0 / c) + eps)
        out[tile_b, :, tile_l] = (v * rms).to(x3.dtype)
    return out


class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        return _rms_norm(x.contiguous().view(b, c, -1), hl.constexpr(self.eps)).view_as(
            x
        )
