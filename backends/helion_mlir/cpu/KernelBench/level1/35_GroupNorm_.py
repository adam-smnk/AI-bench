import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _group_norm(
    x4: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: hl.constexpr
) -> torch.Tensor:
    """GroupNorm of ``x4`` ``[B, G, C/G, H*W]``; ``weight``/``bias`` ``[G, C/G]``.
    Per-channel mean and centered sum of squares first (each channel's row stays
    in L2 for the second pass), then the output with group statistics combined
    from them."""
    b, g, p, hw = x4.shape
    hl.specialize(p)
    hl.specialize(hw)
    block_l = hl.register_block_size(hw)
    mean = torch.empty([b, g, p], dtype=torch.float32, device=x4.device)
    m2 = torch.empty([b, g, p], dtype=torch.float32, device=x4.device)
    out = torch.empty_like(x4)
    for tile_b, tile_g, tile_p in hl.tile([b, g, p]):
        acc = hl.zeros([tile_b, tile_g, tile_p, block_l], dtype=torch.float32)
        for tile_l in hl.tile(hw, block_size=block_l):
            acc = acc + x4[tile_b, tile_g, tile_p, tile_l].to(torch.float32)
        mu = acc.sum(-1) * (1.0 / hw)
        acc = hl.zeros([tile_b, tile_g, tile_p, block_l], dtype=torch.float32)
        for tile_l in hl.tile(hw, block_size=block_l):
            d = x4[tile_b, tile_g, tile_p, tile_l].to(torch.float32) - mu[:, :, :, None]
            acc = acc + d * d
        mean[tile_b, tile_g, tile_p] = mu
        m2[tile_b, tile_g, tile_p] = acc.sum(-1)
    hl.barrier()
    for tile_b, tile_g, tile_p, tile_l in hl.tile([b, g, p, hw]):
        channel_mean = mean[tile_b, tile_g, :]
        group_mean = channel_mean.mean(-1, keepdim=True)
        d = channel_mean - group_mean
        var = (m2[tile_b, tile_g, :] + hw * d * d).sum(-1, keepdim=True) * (
            1.0 / (p * hw)
        )
        rstd = torch.rsqrt(var + eps)
        scale = (
            rstd[:, :, :, None]
            * weight[tile_g, tile_p].to(torch.float32)[None, :, :, None]
        )
        v = x4[tile_b, tile_g, tile_p, tile_l].to(torch.float32)
        out[tile_b, tile_g, tile_p, tile_l] = (
            (v - group_mean[:, :, :, None]) * scale
            + bias[tile_g, tile_p].to(torch.float32)[None, :, :, None]
        ).to(x4.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_l: int, out_l: int) -> helion.Kernel:
    # Block sizes: stats chunk, stats tile [B, G, C/G], output tile [B, G, C/G, H*W].
    key = (block_l, out_l)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _group_norm,
            backend="mlir",
            config=helion.Config(block_sizes=[block_l, 1, 1, 1, 1, 1, 1, out_l]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gn = self.gn
        if not gn.affine:
            raise ValueError("only GroupNorm with affine parameters")
        b, c = x.shape[:2]
        groups = gn.num_groups
        channels = (groups, c // groups)
        x4 = x.contiguous().view(b, *channels, -1)
        hw = x4.shape[-1]
        # Masked lanes would enter the centered sums: chunks divide the row.
        # Stats chunks 256 wide (swept best), output tiles up to 4096 wide.
        return _kernel(math.gcd(hw, 256), min(hw, 4096))(
            x4, gn.weight.view(channels), gn.bias.view(channels), hl.constexpr(gn.eps)
        ).view_as(x)
