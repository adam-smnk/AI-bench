import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn


def _layer_norm(
    x3: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: hl.constexpr
) -> torch.Tensor:
    """LayerNorm of rows of ``x3`` ``[B, P, L]`` (each row split into ``P``
    chunks); ``weight``/``bias`` ``[P, L]``. Per-chunk mean and centered sum of
    squares first (a chunk stays in L2 for the second pass; chunks of a row run
    in parallel), then the output with row statistics combined from them."""
    b, p, chunk = x3.shape
    hl.specialize(p)
    hl.specialize(chunk)
    block_l = hl.register_block_size(chunk)
    mean = torch.empty([b, p], dtype=torch.float32, device=x3.device)
    m2 = torch.empty([b, p], dtype=torch.float32, device=x3.device)
    out = torch.empty_like(x3)
    for tile_b, tile_p in hl.tile([b, p]):
        acc = hl.zeros([tile_b, tile_p, block_l], dtype=torch.float32)
        for tile_l in hl.tile(chunk, block_size=block_l):
            acc = acc + x3[tile_b, tile_p, tile_l].to(torch.float32)
        mu = acc.sum(-1) * (1.0 / chunk)
        acc = hl.zeros([tile_b, tile_p, block_l], dtype=torch.float32)
        for tile_l in hl.tile(chunk, block_size=block_l):
            d = x3[tile_b, tile_p, tile_l].to(torch.float32) - mu[:, :, None]
            acc = acc + d * d
        mean[tile_b, tile_p] = mu
        m2[tile_b, tile_p] = acc.sum(-1)
    hl.barrier()
    for tile_b, tile_p, tile_l in hl.tile([b, p, chunk]):
        chunk_mean = mean[tile_b, :]
        row_mean = chunk_mean.mean(-1, keepdim=True)
        d = chunk_mean - row_mean
        var = (m2[tile_b, :] + chunk * d * d).sum(-1, keepdim=True) * (
            1.0 / (p * chunk)
        )
        rstd = torch.rsqrt(var + eps)
        v = x3[tile_b, tile_p, tile_l].to(torch.float32)
        w = weight[tile_p, tile_l].to(torch.float32)
        out[tile_b, tile_p, tile_l] = (
            (v - row_mean[:, :, None]) * rstd[:, :, None] * w[None, :, :]
            + bias[tile_p, tile_l].to(torch.float32)[None, :, :]
        ).to(x3.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_l: int, out_l: int) -> helion.Kernel:
    # Block sizes: stats chunk, stats tile [B, P], output tile [B, P, L].
    key = (block_l, out_l)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _layer_norm,
            backend="mlir",
            config=helion.Config(block_sizes=[block_l, 1, 1, 1, 1, out_l]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln = self.ln
        if ln.weight is None or ln.bias is None:
            raise ValueError("only LayerNorm with affine parameters")
        n = ln.weight.numel()
        # Chunks of a row: parallel work for few rows, and each fits L2.
        chunk = math.gcd(n, 65536)
        chunks = (n // chunk, chunk)
        x3 = x.contiguous().view(-1, *chunks)
        # Masked lanes would enter the centered sums: blocks divide the chunk.
        # Stats blocks 1024 wide (swept best), output tiles up to 4096 wide.
        return _kernel(math.gcd(chunk, 1024), min(chunk, 4096))(
            x3, ln.weight.view(chunks), ln.bias.view(chunks), hl.constexpr(ln.eps)
        ).view_as(x)
