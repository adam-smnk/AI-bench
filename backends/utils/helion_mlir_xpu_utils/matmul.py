"""Matmuls on Helion's MLIR backend for XPU (KernelBench 1, 2, 6, 7, 9, 13, 16-18).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it.

bf16 operands, f32 accumulator, a bf16 result as PyTorch's matmul of bf16
tensors. Block sizes are ``[M tile, N tile, K tile]``: the workgroup tile and the
K-loop step. The block sizes divide every size they tile (no masking), so a
shape must be chosen with them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

GEMM_BLOCKS = (256, 256, 32)
"""256x256 workgroups of 32 subgroups (lighthouse's inference picks 64x32
subgroup tiles for the shapes in its tuned parameter database, 1, 2, 7, 9 and
13, and 32x64 for the transposed 16-18) with K steps of 32. Other tiles for particular shapes: 128x128 for the
256x256 output of 6_Matmul_with_large_K_dimension_ (4 workgroups), a 16 K step
for K = 32 (9_Tall_skinny_matrix_multiplication_; the K loop must stay, DPAS K
is 16). Transposed operands (A stored ``[K, M]``, B ``[N, K]``) are loaded as
transposed 16x16 tiles."""


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b`` of ``[M, K]`` and ``[K, N]``."""
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_a(a: Tensor, b: Tensor) -> Tensor:
    """``a.T @ b`` of ``a: [K, M]`` and ``b: [K, N]``."""
    k, m = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tk, tn])
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_b(a: Tensor, b: Tensor) -> Tensor:
    """``a @ b.T`` of ``a: [M, K]`` and ``b: [N, K]``."""
    m, k = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tm, tk], b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


def _matmul_transposed_both(a: Tensor, b: Tensor) -> Tensor:
    """``a.T @ b.T`` of ``a: [K, M]`` and ``b: [N, K]``."""
    k, m = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, a[tk, tm].T, b[tn, tk].T)
        out[tm, tn] = acc.to(a.dtype)
    return out


_MATMULS: dict[tuple[bool, bool], Callable[..., Tensor]] = {
    (False, False): _matmul,
    (True, False): _matmul_transposed_a,
    (False, True): _matmul_transposed_b,
    (True, True): _matmul_transposed_both,
}
_KERNELS: dict[tuple[bool, bool, tuple[int, ...]], helion.Kernel] = {}


def _kernel(
    trans_a: bool, trans_b: bool, block_sizes: tuple[int, ...]
) -> helion.Kernel:
    key = (trans_a, trans_b, block_sizes)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _MATMULS[trans_a, trans_b],
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=list(block_sizes)),
        )
    return _KERNELS[key]


def matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    block_sizes: Sequence[int] | None = None,
) -> Tensor:
    """``op(A) @ op(B)``, ``op`` transposing with ``trans_a``/``trans_b``: ``A`` is
    ``[M, K]`` (``[K, M]`` with ``trans_a``) and ``B`` ``[K, N]`` (``[N, K]``
    with ``trans_b``). ``block_sizes`` (M tile, N tile, K tile; default
    :data:`GEMM_BLOCKS`) must divide ``M``, ``N`` and ``K``."""
    blocks = GEMM_BLOCKS if block_sizes is None else tuple(block_sizes)
    return _kernel(trans_a, trans_b, blocks)(a, b)
