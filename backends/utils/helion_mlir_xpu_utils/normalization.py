"""Row reductions on Helion's MLIR backend for XPU (KernelBench 23, 24, 38, 39).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it.

f32 ``[M, N]`` inputs (4096x393216 and 32768x65536 in the ``mlir-gpu-bench``
variant). ``[rows, column chunk]`` block sizes: 64 rows per workgroup
(lighthouse's reduction class: 8 subgroups of 8 rows) and 64-column chunks,
multi-pass over the chunks like AI-bench's CPU kernels. The block sizes divide
the input (no masking). The column block is registered first, so it comes first
in the config; both are 64.
"""

from __future__ import annotations

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor

REDUCTION_BLOCKS = (64, 64)
_CONFIG = helion.Config(block_sizes=list(REDUCTION_BLOCKS))


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _softmax_kernel(x: Tensor) -> Tensor:
    """Row softmax in three passes over column chunks: maxima, sums of
    exponentials, then the output."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n] - row_top)
        inv = 1.0 / total.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - row_top) * inv
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _log_softmax_kernel(x: Tensor) -> Tensor:
    """Row log-softmax: ``x - max - log(sum(exp(x - max)))``, in three passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            top = torch.maximum(top, x[tile_m, tile_n])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + torch.exp(x[tile_m, tile_n] - row_top)
        shift = row_top + torch.log(total.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] - shift
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _l1_norm_kernel(x: Tensor) -> Tensor:
    """``x / mean(|x|, dim=1, keepdim=True)`` (KernelBench 38), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        inv = n / acc.sum(-1, keepdim=True)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inv
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _l2_norm_kernel(x: Tensor) -> Tensor:
    """``x / norm(x, p=2, dim=1, keepdim=True)`` (KernelBench 39), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            acc = acc + v * v
        inv = torch.rsqrt(acc.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * inv
    return out


def softmax(x: Tensor) -> Tensor:
    """Softmax over the last dim of ``[M, N]``."""
    return _softmax_kernel(x)


def log_softmax(x: Tensor) -> Tensor:
    """Log-softmax over the last dim of ``[M, N]``."""
    return _log_softmax_kernel(x)


def l1_norm(x: Tensor) -> Tensor:
    """``x`` over the mean of its absolute values along dim 1 of ``[M, N]``."""
    return _l1_norm_kernel(x)


def l2_norm(x: Tensor) -> Tensor:
    """``x`` over its L2 norm along dim 1 of ``[M, N]``."""
    return _l2_norm_kernel(x)
