import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


@helion.kernel(backend="mlir", config=helion.Config(block_sizes=[1, 1, 4096]))
def _batch_norm(
    x3: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: hl.constexpr,
) -> torch.Tensor:
    """BatchNorm with inference statistics of ``x3`` ``[B, C, H*W]``."""
    b, c, hw = x3.shape
    out = torch.empty_like(x3)
    for tile_b, tile_c, tile_l in hl.tile([b, c, hw]):
        scale = torch.rsqrt(running_var[tile_c].to(torch.float32) + eps) * weight[
            tile_c
        ].to(torch.float32)
        shift = (
            bias[tile_c].to(torch.float32)
            - running_mean[tile_c].to(torch.float32) * scale
        )
        v = x3[tile_b, tile_c, tile_l].to(torch.float32)
        out[tile_b, tile_c, tile_l] = (
            v * scale[None, :, None] + shift[None, :, None]
        ).to(x3.dtype)
    return out


class Model(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ValueError("BatchNorm with running statistics needs eval mode")
        bn = self.bn
        b, c = x.shape[:2]
        return _batch_norm(
            x.contiguous().view(b, c, -1),
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            hl.constexpr(bn.eps),
        ).view_as(x)
