import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

# Growth of a row maximum (in scaled logits) below which the stale maximum is
# kept: exponentials stay below e^8, well within f32 and bf16 range.
_RESCALE_THRESHOLD = 8.0


def _dot_nt(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b.T`` of ``[M, K]`` and ``[N, K]``; bf16 operands in AMX's VNNI
    layouts, ``[M, K/2, 2]`` and ``[K/2, N, 2]``."""
    if not vnni:
        return torch.einsum("mk,nk->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0), b.size(1) // 2, 2).permute(1, 0, 2)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _dot_nn(a: torch.Tensor, b: torch.Tensor, vnni: bool) -> torch.Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``, VNNI layouts as in ``_dot_nt``."""
    if not vnni:
        return torch.einsum("mk,kn->mn", a, b)
    a3 = a.reshape(a.size(0), a.size(1) // 2, 2)
    b3 = b.reshape(b.size(0) // 2, 2, b.size(1)).permute(0, 2, 1)
    return torch.einsum("mcv,cnv->mn", a3, b3)


def _exp_bf16(x: torch.Tensor) -> torch.Tensor:
    """``exp(x)`` of f32 ``x <= _RESCALE_THRESHOLD`` at bf16's accuracy: a cubic
    in ``x / 64``, squared six times. The relative error is about
    ``x^4 / (24 * 64^3)``: below 8e-4 for ``|x| <= 8`` (2.5e-6 at 2), under
    bf16's rounding of the weights. It grows further down, 0.22 at ``x = -32``:
    with ``V`` spanning over about 1e6, such a weight can show in the output.
    The clamp is safely past where the cubic's 64th power underflows (denormal
    from about -74, 0 by -84), so ``-inf`` scores weigh nothing, as with
    ``exp``; the cubic is positive down to it. L1/97 at S2048 D128, EMR: 12.2 ms with the f32-accurate polynomial
    and its clamps, 10.9 with this."""
    y = torch.clamp(x, min=-96.0) * 0.015625
    p = 1.0 + y * (1.0 + y * (0.5 + y * (1.0 / 6.0)))
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    p = p * p
    return p * p


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: hl.constexpr,
    vnni: hl.constexpr,
) -> torch.Tensor:
    """Softmax attention of ``[BH, S, D]`` heads with logits scaled by ``scale``,
    online over key blocks: the running row maxima rescale the sums and the
    output accumulator, only when one of them grows past ``_RESCALE_THRESHOLD``."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    out = torch.empty_like(q)
    for h in hl.grid(heads):
        for tile_m in hl.tile(seq):
            q_tile = q[h, tile_m, :]
            # -FLT_MAX, below any finite score. Finite: Helion's value range
            # analysis of the growth takes -inf - -inf.
            top = hl.full([tile_m], -3.4028234663852886e38, dtype=torch.float32)
            total = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, dim], dtype=torch.float32)
            for tile_n in hl.tile(seq):
                scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                scores = scores + _dot_nt(q_tile, k[h, tile_n, :], vnni)
                maxima = torch.amax(scores, dim=-1) * scale
                # Not "> threshold": a NaN row must open this too, else every
                # other row of the block keeps its stale top and overflows.
                if torch.logical_not(torch.amax(maxima - top) <= _RESCALE_THRESHOLD):
                    new_top = torch.maximum(top, maxima)
                    alpha = torch.exp(top - new_top)
                    total = total * alpha
                    acc = acc * alpha[:, None]
                    top = new_top
                if vnni:
                    p = _exp_bf16(scores * scale - top[:, None])
                else:
                    p = torch.exp(scores * scale - top[:, None])
                total = total + p.sum(-1)
                acc = acc + _dot_nn(p.to(q.dtype), v[h, tile_n, :], vnni)
            # A row with no weight at all (all scores -inf) is 0, not 0 / 0;
            # too small to change a sum of weights, whose largest is near 1.
            out[h, tile_m, :] = (acc / (total[:, None] + 1.0e-37)).to(q.dtype)
    return out


_KERNELS: dict[tuple[int, int], helion.Kernel] = {}


def _kernel(block_m: int, block_n: int) -> helion.Kernel:
    key = (block_m, block_n)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _attention,
            backend="mlir",
            config=helion.Config(block_sizes=[block_m, block_n]),
        )
    return _KERNELS[key]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        b, h, s, d = Q.shape
        heads = b * h
        scale = 1.0 / math.sqrt(d)
        # bf16 runs on AMX, its operands in VNNI layout: pairs along D and along
        # the key block. A sequence of one forms no pair of keys: plain bf16 dots.
        vnni = Q.dtype == torch.bfloat16 and s >= 2
        # The VNNI pairs along D need it even: an odd D gets one zero column,
        # which changes no dot product.
        dim = d + d % 2 if vnni else d
        if dim != d:
            Q, K, V = (F.pad(t, (0, dim - d)) for t in (Q, K, V))
        # Tall query blocks pack each K/V block for many rows; swept on EMR
        # (S2048 D128, ms: 128 rows 13.1, 256 12.3, 512 11.8, 1024 12.1).
        block_m = min(s, 512 if dim <= 128 else 256)
        # VNNI pairs along the key block need it even: a block of at least S is
        # all of S, so an odd S gets one less and a masked tail.
        block_n = min(s - s % 2 if vnni else s, 128 if dim <= 64 else 64)
        out = _kernel(block_m, block_n)(
            Q.reshape(heads, s, dim),
            K.reshape(heads, s, dim),
            V.reshape(heads, s, dim),
            hl.constexpr(scale),
            hl.constexpr(vnni),
        )
        out = out.view(b, h, s, dim)
        return out if dim == d else out[..., :d].contiguous()
