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

ATTENTION_BLOCKS = (1, 1, 128, 64)
"""Flash-attention block sizes ``[batch, head, query rows, K/V rows]``, picked
from the hardware (Xe2, large register file) for a head dim of 64:

- Batch and head blocks 1: each workgroup works on one (batch, head) matrix
  (unit tile dims are folded; the grid is B x H x S/128 = 32x32x4).
- 128 query rows per workgroup: 8 subgroups of 16 rows (lighthouse's attention
  layout: subgroup tiles of 16 rows, two 8-row DPAS blocks), 128 threads of
  16 lanes. Every subgroup reads the whole K and V tiles, so more rows per
  workgroup share each K/V tile through the L1 cache; 8 subgroups (of at
  most 32) still leave room for several workgroups per Xe core, and
  32x32x4 = 4096 workgroups fill the GPU.
- K/V tiles of 64 rows: per subgroup, ``Q @ K^T`` is 16x64 (2x4 DPAS 8x16
  blocks, 4 K steps of 16 over the head dim) and ``P @ V`` 16x64 over 64 K/V
  rows (4 K steps): 32 DPAS each per step. The K or V tile (64x64 bf16, 128
  registers of 64 bytes) with Q (16), the scores (32) and the accumulator (32)
  fits the 256-register file; 128 rows (256 registers) would spill. 64 is
  also lighthouse's attention reduction tile, and whole 32x32 V blocks.
- 2D block loads: Q in 16x16 blocks, V in 32x32 blocks (the VNNI-transformed
  B operand), K transposed (``Q @ K^T`` of K stored ``[seq, head dim]``) in
  16x8 blocks of 32-bit pairs; the output is stored in 8x16 blocks.
"""


@helion.kernel(
    backend="mlir",
    static_shapes=True,
    config=helion.Config(block_sizes=list(ATTENTION_BLOCKS)),
)
def _attention_kernel(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``F.scaled_dot_product_attention(q, k, v)`` of ``[B, H, S, D]`` (no mask,
    scale ``1/sqrt(D)``) as flash attention: each workgroup takes a tile of
    query rows of one (batch, head) and runs over the K/V sequence in tiles,
    with an online softmax (running row maxima and sums, the output rescaled
    when the maximum grows). The matmuls take bf16 operands and accumulate in
    f32 (DPAS); the softmax is f32; ``P`` is rounded to V's dtype for ``P @ V``,
    the output to Q's.

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
        row_max = hl.full([tb, th, tm], float("-inf"), dtype=torch.float32)
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


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Scaled dot-product attention of bf16 ``[B, H, S, D]`` tensors (``D`` 64)."""
    return _attention_kernel(q, k, v)
