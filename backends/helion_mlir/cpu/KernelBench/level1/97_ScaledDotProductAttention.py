import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

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


def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, vnni: hl.constexpr
) -> torch.Tensor:
    """Softmax attention of ``[BH, S, D]`` heads, online over key blocks: the
    running row maxima rescale the sums and the output accumulator, only when
    one of them grows past ``_RESCALE_THRESHOLD``."""
    heads, seq, dim = q.shape
    hl.specialize(seq)
    hl.specialize(dim)
    scale = 1.0 / math.sqrt(dim)
    out = torch.empty_like(q)
    for h in hl.grid(heads):
        for tile_m in hl.tile(seq):
            q_tile = q[h, tile_m, :]
            # Finite: Helion's value range analysis of the growth takes -inf - -inf.
            top = hl.full([tile_m], -1.0e30, dtype=torch.float32)
            total = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, dim], dtype=torch.float32)
            for tile_n in hl.tile(seq):
                scores = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                scores = scores + _dot_nt(q_tile, k[h, tile_n, :], vnni)
                new_top = torch.maximum(top, torch.amax(scores, dim=-1) * scale)
                if torch.amax(new_top - top) > _RESCALE_THRESHOLD:
                    alpha = torch.exp(top - new_top)
                    total = total * alpha
                    acc = acc * alpha[:, None]
                    top = new_top
                p = torch.exp(scores * scale - top[:, None])
                total = total + p.sum(-1)
                acc = acc + _dot_nn(p.to(q.dtype), v[h, tile_n, :], vnni)
            out[h, tile_m, :] = (acc / total[:, None]).to(q.dtype)
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
        # Tall query blocks pack each K/V block for many rows; swept on EMR.
        block_m = min(s, 512 if d <= 64 else 256)
        block_n = min(s, 128 if d <= 64 else 64)
        out = _kernel(block_m, block_n)(
            Q.reshape(heads, s, d),
            K.reshape(heads, s, d),
            V.reshape(heads, s, d),
            hl.constexpr(Q.dtype == torch.bfloat16),
        )
        return out.view(b, h, s, d)
