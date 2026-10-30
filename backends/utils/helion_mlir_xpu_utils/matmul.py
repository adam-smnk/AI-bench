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
from typing import Any

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

GEMM_BLOCKS = (256, 256, 32)
"""256x256 workgroups of 32 subgroups with K steps of 32, except where measured
otherwise on the B50: 2 forces 32x64 subgroup tiles (the 64x32 lighthouse infers
stalls the memory pipe at K = 8192, :data:`GEMM_STANDARD`), and 1 and 13 use
128x512 workgroups (:data:`GEMM_SQUARE`), so the 256x256 rule no longer holds
for them. Lighthouse's inference picks 64x32 subgroup tiles for the shapes in
its tuned parameter database (1, 2, 7, 9 and 13) on 256x256 workgroups and 32x64
for the transposed 16-18. Other tiles for particular shapes: 128x128 for the
256x256 output of 6_Matmul_with_large_K_dimension_ (4 workgroups), a 16 K step
for K = 32 (9_Tall_skinny_matrix_multiplication_; the K loop must stay, DPAS K
is 16). Transposed operands (A stored ``[K, M]``, B ``[N, K]``) are loaded as
transposed 16x16 tiles."""
GEMM_SQUARE: dict[str, Any] = {
    "block_sizes": (128, 512, 32),
    "xegpu_sg_tiles": ((32, 64),),
    "xegpu_load_tiles": ((32, 32, 16, 32),),
}
"""1_Square and 13_Matmul_for_symmetric_matrices (both 4096^3) on the Arc B50
(16 Xe cores, one 32-subgroup workgroup per core, 18 MB L2, about 220 GB/s
DRAM). Keep in sync with ``GEMM_SQUARE`` in
helion-mlir-backend's ``tests/kernelbench_xpu.py``.

- 128x512 workgroups, not 256x256 (the reason is DRAM traffic per wave).
  Workgroups are dispatched in linear order with the first grid dim (M)
  fastest, so a wave of 16 workgroups (one per Xe core) of 256x256 tiles is a
  full column of M blocks and re-reads all of A: about 544 MB of reads per
  GEMM, 2.4 ms at the measured 220 GB/s, of 3.9 ms. A wave of 128x512 tiles
  covers half of M, which halves A's traffic per wave (about 288 MB per GEMM).
  The price is more L2-to-core traffic: a workgroup reads 128x32 + 32x512 bf16
  (40 KB) per K step for 4.2 Mflop, 102 flop per byte against 128 for 256x256.
  Forcing 32x64 subgroup tiles on 256x256 workgroups does not help (34.3
  TFLOPS): it is the workgroup shape.
- Subgroup tiles 32x64 (4x8 subgroups, 128 of the 256 registers for the
  accumulator). Lighthouse's inference also gives 32x64 on 128x512
  workgroups (only on 256x256 it picks 64x32); the tile is pinned so the
  config does not depend on the selector. Forcing 64x32 costs about 3 % at the
  70 W cap (41.6 against 42.8 TFLOPS device time, at a 5 % lower clock).
- Loads: A 32x32 (a subgroup's whole K step, 64 bytes wide, in one block read),
  B 16x32, as lighthouse's tuned parameters for this shape. A 8x32 / B 32x32
  and A 32x32 / B 32x32 measure the same within noise (42.6, 42.6, 42.2
  TFLOPS). Prefetches are inferred.
- B50-specific: N must be a multiple of 512 and a 16-workgroup wave must cover
  half of M. On a card with 32 Xe cores (B70) a 32-workgroup wave of 128x512
  tiles is again a full M column (32 M blocks); the portable fix is a grouped
  workgroup walk (plan P3).
- Measured 2026-10-30 on the B50 (device time, interleaved; efficiency =
  TFLOPS / (16 Xe cores x 2048 flop/clk x sampled clock), as the card runs at
  its 70 W limit): 42.6 TFLOPS (94 %) against 34.5 (64 %) for the inferred
  256x256 / 64x32; torch 46.0 (95 %). Bit-identical to ``torch.matmul``.

Pass as ``matmul(a, b, **GEMM_SQUARE)``."""
GEMM_STANDARD: dict[str, Any] = {
    "block_sizes": (256, 256, 32),
    "xegpu_sg_tiles": ((32, 64),),
    "xegpu_load_tiles": ((8, 32, 32, 32),),
    "xegpu_prefetch_tiles": ((16, 16, 16, 16),),
    "xegpu_prefetch_depths": ((1, 1),),
}
"""2_Standard (M 2048, N 4096, K 8192) on the Arc B50. Keep in sync with
``GEMM_STANDARD`` in helion-mlir-backend's ``tests/kernelbench_xpu.py``.

- 256x256 workgroups (128 of them: 8 waves of 16 over the Xe cores).
- Subgroup tiles 32x64 (8x4 subgroups) in place of the 64x32 lighthouse
  infers: 64x32 collapses here (23.5 TFLOPS, 51 %) against 42.8 (94 %) with
  32x64. The cause is a memory-pipe stall with 64 strided A rows per subgroup
  at K = 8192 (EU stall sampling: SendStall 41 % against 6 %); it is not shown
  to be a pitch effect alone.
- Loads A 8x32, B 32x32 (lighthouse's tuned parameters for this shape) and
  prefetches of 16x16 at depth 1 (also lighthouse's). In one interleaved run:
  32x64 with inferred loads and prefetches 42.8, with the loads alone 42.6,
  with the loads and the 16x16 prefetch tiles 43.4: the gain is from the
  prefetch tiles. Prefetch depth 2 is equal to depth 1 (43.5 against 43.4).
- B50-specific, like the tiles of ``GEMM_SQUARE``: 128 workgroups are 8 waves
  of 16 on 16 Xe cores.
- Measured 2026-10-30 on the B50 (device time, interleaved; efficiency =
  TFLOPS / (16 Xe cores x 2048 flop/clk x sampled clock), as the card runs at
  its 70 W limit): 43.4-43.6 TFLOPS (95 %); torch 46.6 (98 %). Bit-identical
  to ``torch.matmul``.

Pass as ``matmul(a, b, **GEMM_STANDARD)``."""


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
_KERNELS: dict[tuple[bool, bool, str], helion.Kernel] = {}


def _lists(value: Any) -> Any:
    """``value`` with tuples as lists (the form of ``helion.Config`` values)."""
    if isinstance(value, (tuple, list)):
        return [_lists(item) for item in value]
    return value


def _kernel(trans_a: bool, trans_b: bool, config: dict[str, Any]) -> helion.Kernel:
    key = (trans_a, trans_b, repr(sorted(config.items())))
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            _MATMULS[trans_a, trans_b],
            backend="mlir",
            static_shapes=True,
            config=helion.Config(**{k: _lists(v) for k, v in config.items()}),
        )
    return _KERNELS[key]


def matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    block_sizes: Sequence[int] | None = None,
    **xegpu: Any,
) -> Tensor:
    """``op(A) @ op(B)``, ``op`` transposing with ``trans_a``/``trans_b``: ``A`` is
    ``[M, K]`` (``[K, M]`` with ``trans_a``) and ``B`` ``[K, N]`` (``[N, K]``
    with ``trans_b``). ``block_sizes`` (M tile, N tile, K tile; default
    :data:`GEMM_BLOCKS`) must divide ``M``, ``N`` and ``K``. ``xegpu`` are
    further config keys (``xegpu_sg_tiles``, ...; default: inferred), as in
    :data:`GEMM_SQUARE` and :data:`GEMM_STANDARD`."""
    blocks = GEMM_BLOCKS if block_sizes is None else tuple(block_sizes)
    return _kernel(trans_a, trans_b, {"block_sizes": blocks, **xegpu})(a, b)
