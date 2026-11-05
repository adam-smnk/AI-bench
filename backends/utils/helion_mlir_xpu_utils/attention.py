"""Flash attention on Helion's MLIR backend for XPU (KernelBench 97).

The kernel mirrors helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it.

bf16 ``[B, H, S, D]`` inputs (32x32x512x64 in the ``mlir-gpu-bench`` variant),
written fused and tiled as flash attention (online softmax over K/V tiles).
"""

from __future__ import annotations

import math

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor

ATTENTION_BLOCKS = (1, 1, 512, 64)
"""Flash-attention block sizes ``[batch, head, query rows, K/V rows]`` of the bench
shape, picked on the Arc B50 for a head dim of 64, with NaN-ignoring row maxima:

- Batch and head blocks 1: each workgroup works on one (batch, head) matrix
  (unit tile dims are folded; the grid is B x H x S/512 = 32x32x1).
- 512 query rows per workgroup: 32 subgroups of 16 rows (lighthouse's attention
  layout: subgroup tiles of 16 rows, two 8-row DPAS blocks), a whole Xe core's
  threads. Every subgroup reads the whole K and V tiles, so more rows per
  workgroup share each K/V tile through the L1 cache.
- K/V tiles of 64 rows: per subgroup, ``Q @ K^T`` is 16x64 (2x4 DPAS 8x16 blocks, 4 K
  steps of 16 over the head dim) and ``P @ V`` 16x64 over 64 K/V rows (4 K steps): 32
  DPAS each per step.
- Of the query and K/V row counts tried (128 to 512, and 32 to 128), 512 and 64 had the
  lowest device time.
- 2D block loads: Q in 16x16 blocks, V in 32x32 blocks (the VNNI-transformed
  B operand), K transposed (``Q @ K^T`` of K stored ``[seq, head dim]``) in
  16x8 blocks of 32-bit pairs; the output is stored in 8x16 blocks."""
ATTENTION_OTHER_SHAPES_BLOCKS = (1, 1, 128, 64)
"""For sequences that are a multiple of 128 and not of 512: 128 query rows
over K/V tiles of 64 fit them, and leave a K/V loop at a sequence of 128."""


def _attention_form(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``F.scaled_dot_product_attention(q, k, v)`` of ``[B, H, S, D]`` (no mask,
    scale ``1/sqrt(D)``) as flash attention: each workgroup takes a tile of
    query rows of one (batch, head) and runs over the K/V sequence in tiles,
    with an online softmax (running row maxima and sums, the output rescaled
    when the maximum grows). The matmuls take bf16 operands and accumulate in
    f32 (DPAS); the softmax is f32; ``P`` is rounded to V's dtype for ``P @ V``,
    the output to Q's.

    ``row_max`` starts at -FLT_MAX, not -inf, so ``score - row_max`` is never
    ``-inf - -inf``: a first K/V tile of -inf scores adds ``exp(-inf) = 0``, as
    in torch, and a row of only -inf scores ends in ``0 / 0``. Row maxima may
    ignore NaN (``xegpu_nan_ignoring_maxima``): a NaN score gives a NaN row
    through ``exp`` and the sum anyway.

    The tiles are 4D with unit batch and head blocks, so the contractions are
    einsums (``hl.dot`` takes 2D or 3D operands). An einsum of bf16 tiles
    returns bf16; added to an f32 accumulator it accumulates in f32 (the
    backend fuses ``acc + einsum`` into one contraction), so ``Q @ K^T``
    accumulates into zeros."""
    batch, heads, m, d = q.shape
    n = k.size(2)
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    for tb, th, tm in hl.tile([batch, heads, m]):
        q_tile = q[tb, th, tm, :]
        row_max = hl.full([tb, th, tm], -3.4028234663852886e38, dtype=torch.float32)
        row_sum = hl.zeros([tb, th, tm], dtype=torch.float32)
        acc = hl.zeros([tb, th, tm, d], dtype=torch.float32)
        for tn in hl.tile(n):
            scores = hl.zeros([tb, th, tm, tn], dtype=torch.float32)
            scores = scores + torch.einsum("bhmd,bhnd->bhmn", q_tile, k[tb, th, tn, :])
            scores = scores * scale
            new_max = torch.maximum(row_max, scores.amax(-1))
            p = torch.exp(scores - new_max[:, :, :, None])
            alpha = torch.exp(row_max - new_max)
            row_sum = row_sum * alpha + p.sum(-1)
            acc = acc * alpha[:, :, :, None] + torch.einsum(
                "bhmn,bhnd->bhmd", p.to(v.dtype), v[tb, th, tn, :]
            )
            row_max = new_max
        out[tb, th, tm, :] = (acc / row_sum[:, :, :, None]).to(out.dtype)
    return out


_KERNEL = helion.kernel(
    _attention_form,
    backend="mlir",
    static_shapes=True,
    config=helion.Config(
        block_sizes=list(ATTENTION_BLOCKS),
        xegpu_nan_ignoring_maxima=[True],
    ),
)
_OTHER_SHAPES_KERNEL = helion.kernel(
    _attention_form,
    backend="mlir",
    static_shapes=True,
    config=helion.Config(
        block_sizes=list(ATTENTION_OTHER_SHAPES_BLOCKS),
        xegpu_nan_ignoring_maxima=[True],
    ),
)


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Scaled dot-product attention of bf16 ``[B, H, S, D]`` tensors (``D`` 64):
    with ``ATTENTION_BLOCKS`` if ``S`` is a multiple of 512 and the kernel has
    several workgroups, else ``ATTENTION_OTHER_SHAPES_BLOCKS`` (``S`` a multiple of
    128)."""
    batch, heads, seq, _ = q.shape
    fits = seq % 512 == 0 and batch * heads * seq > 512
    return (_KERNEL if fits else _OTHER_SHAPES_KERNEL)(q, k, v)
