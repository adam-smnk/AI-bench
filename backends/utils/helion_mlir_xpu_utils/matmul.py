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
for the transposed 16-18. Other configs for particular shapes: 7 and 9 (output
store bound) take :data:`GEMM_SMALL_K` (9 with K padded from 32 to 64, the
tiles' two K steps), 6 (K = 524288, a 256x256 output) a split K
(:data:`GEMM_SPLIT_K`). Transposed operands (A stored ``[K, M]``, B ``[N, K]``)
are loaded as transposed 16x16 tiles."""
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

GEMM_SMALL_K: dict[str, Any] = {
    "block_sizes": (64, 128, 32),
    "xegpu_sg_tiles": ((8, 64),),
}
"""7_Matmul_with_small_K_dimension_ (32768x32768x64) and, with K padded to
64, 9_Tall_skinny_matrix_multiplication_ (32768x32768x32), on the Arc B50.
Keep in sync with ``GEMM_SMALL_K`` in helion-mlir-backend's
``tests/kernelbench_xpu.py``.

- These GEMMs are bound by the output store: 2 GiB of bf16 written for 8 MiB
  of operands (which stay in the 18 MB L2), about 0.13 TFLOP of DPAS work.
  The bound is the write bandwidth; torch reaches 117 GB/s (7) and 136 (9).
- The grid runs the column tiles first (the backend's ``columns_first`` rule
  for GEMMs that write more than they read): the workgroups resident at a
  time write neighbouring tiles of the same rows. Rows first, every resident
  workgroup writes rows in other DRAM pages: 49.5 GB/s at 256x256 tiles
  (85.9 columns first), 53.6 at these tiles (140.9 columns first).
- 64x128 workgroups of 16 subgroups of 8x64 (8x2): each subgroup writes 8 rows
  of 128 contiguous bytes (four 8x16 block stores side by side), a workgroup
  64 rows of 256 bytes. 16-bit 2D block stores write 32-byte rows, half a
  cache line; fewer rows per subgroup and per workgroup and wider rows per
  subgroup measured faster (columns first, GB/s): 256x256 / inferred 64x32
  86, 128x512 / inferred 98, 64x512 / 32x32 104, 64x256 / 16x32 117, 64x256 /
  8x64 131, 128x64 / 8x64 139, 64x128 / 8x64 141-143. 64 rows are the
  minimum: lighthouse wants 16 prefetching subgroups (64 rows x 32 K = 16 8x16
  prefetch tiles of A), and at least 16 subgroups per workgroup (64x64 / 8x64
  is rejected).
- The 16-bit store width still costs: the same tiles with an f32 output write
  184 GB/s (4 GiB in 23.4 ms) against 141 GB/s for bf16. A 64-byte-row bf16
  store (IGC has ``intel_sub_group_2d_block_write_16b_8r32x1c``; XeVM takes
  16-bit ``tile_width`` 16 only) needs bf16 pairs packed per work item, which
  the DPAS accumulator layout does not give.
- K steps of 32 (2 per tile), as lighthouse requires a K loop.
- Measured 2026-10-31 on the B50 (e2e, interleaved, L2 flushed): 15.1 ms,
  142.7 GB/s, against torch 18.4 ms (117 GB/s); was 43.5 ms (49.5 GB/s).
  Bit-identical to ``torch.matmul``.

Pass as ``matmul(a, b, **GEMM_SMALL_K)``."""
GEMM_SPLIT_K: dict[str, Any] = {
    "splits": 32,
    "block_sizes": (1, 256, 256, 32),
    "xegpu_sg_tiles": ((32, 64),),
    "xegpu_load_tiles": ((32, 32, 16, 32),),
    "xegpu_prefetch_depths": ((4, 2),),
    "reduce_block_sizes": (8, 256),
}
"""6_Matmul_with_large_K_dimension_ (256x256x524288) on the Arc B50, as
:func:`matmul_split_k`. Keep in sync with ``GEMM_SPLIT_K`` in
helion-mlir-backend's ``tests/kernelbench_xpu.py``.

- The GEMM reads 512 MiB of operands for a 256x256 output: memory-bound
  (floor 2.4 ms at the measured 222 GB/s read bandwidth); torch takes 3.16 ms
  (170 GB/s) with 16 persistent workgroups.
- Without a split, the output is 4 workgroups of 128x128 for 16 Xe cores:
  9.95 ms (54 GB/s). Split over K into 32 chunks of 16384, it is 32
  workgroups of 256x256 (two waves of one per Xe core, 32 subgroups of 32x64,
  A read 32x32 per block read), each reading its chunk of A and B once from
  DRAM, writing f32 partial sums ``[32, 256, 256]`` (8 MiB); a second kernel
  adds them into the bf16 result (8-row tiles, 32 workgroups, 26 us).
- The chunks are contiguous views (``A.view(M, S, K/S)``): a kernel argument
  is a contiguous buffer, so a strided ``[S, M, K/S]`` view would be copied
  whole on every call (2.7 ms for A's 256 MiB: 6.3 ms in all).
- Prefetch 4 K steps ahead for A (its rows are 1 MiB apart: a K step reads
  64 bytes of each of 256 rows), 2 for B: 3.33 ms e2e, against 3.42-3.48 at
  depths 1/1 and 2/2; 3/3 3.61, 4/4 5.19 (too many prefetches in flight).
  16, 32 or 64 splits: 3.37-3.79; 128x256 workgroups 4.29.
- Normalized error against an f32 reference 2.6e-3, as torch's.
- Measured 2026-10-31 on the B50 (e2e, interleaved, L2 flushed).

Pass as ``matmul_split_k(a, b, **GEMM_SPLIT_K)``."""


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


def _matmul_split_k(a: Tensor, b: Tensor) -> Tensor:
    """f32 partial products ``[S, M, N]`` of ``a: [M, S, Kc]`` and
    ``b: [S, Kc, N]``, contiguous views of ``[M, K]`` and ``[K, N]`` operands
    (:func:`matmul_split_k`): chunk ``s`` of a K split into ``S`` chunks of
    ``Kc``. A chunk of ``a`` is the tile ``a[tm, s, tk]``, rows ``K`` apart:
    passing a strided ``[S, M, Kc]`` view instead would copy all of A on every
    call (kernel arguments are contiguous)."""
    m, splits, kc = a.size()
    _, _, n = b.size()
    out = torch.empty([splits, m, n], dtype=torch.float32, device=a.device)
    for ts, tm, tn in hl.tile([splits, m, n]):
        acc = hl.zeros([ts, tm, tn], dtype=torch.float32)
        for tk in hl.tile(kc):
            chunk = a[tm, ts.begin, tk].unsqueeze(0)
            acc = hl.dot(chunk, b[ts, tk, tn], acc=acc)
        out[ts, tm, tn] = acc
    return out


def _sum_splits(parts: Tensor) -> Tensor:
    """``parts.sum(0)`` of f32 ``[S, M, N]`` partial products, rounded to bf16
    (in split order)."""
    splits, m, n = parts.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=parts.device)
    for tm, tn in hl.tile([m, n]):
        acc = parts[0, tm, tn]
        for index in hl.static_range(1, splits):
            acc = acc + parts[index, tm, tn]
        out[tm, tn] = acc.to(torch.bfloat16)
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


def _compiled(fn: Callable[..., Tensor], config: dict[str, Any]) -> helion.Kernel:
    key = (fn.__name__, repr(sorted(config.items())))
    if key not in _OTHER_KERNELS:
        _OTHER_KERNELS[key] = helion.kernel(
            fn,
            backend="mlir",
            static_shapes=True,
            config=helion.Config(**{k: _lists(v) for k, v in config.items()}),
        )
    return _OTHER_KERNELS[key]


_OTHER_KERNELS: dict[tuple[str, str], helion.Kernel] = {}


def matmul_split_k(
    a: Tensor,
    b: Tensor,
    splits: int,
    block_sizes: Sequence[int],
    reduce_block_sizes: Sequence[int],
    **xegpu: Any,
) -> Tensor:
    """``A @ B`` of bf16 ``A: [M, K]`` and ``B: [K, N]`` with K split into
    ``splits`` chunks: a batched GEMM of f32 partial products over views of
    the chunks (``block_sizes``: ``[1, M tile, N tile, K tile]``, ``xegpu``
    its config keys), then a kernel summing them into the bf16 result
    (``reduce_block_sizes``). ``splits`` must divide K, the tiles the sizes
    they tile. See :data:`GEMM_SPLIT_K`."""
    m, k = a.size()
    _, n = b.size()
    if k % splits:
        raise ValueError(f"{splits} splits do not divide K = {k}")
    if len(block_sizes) != 4 or block_sizes[0] != 1:
        raise ValueError(
            "matmul_split_k() block sizes are [1, M tile, N tile, K tile] (one K "
            f"chunk per workgroup), got {list(block_sizes)}"
        )
    chunk = k // splits
    # Contiguous views (a strided [S, M, K/S] view of A would be copied).
    a_chunks = a.view(m, splits, chunk)
    b_chunks = b.view(splits, chunk, n)
    gemm = _compiled(_matmul_split_k, {"block_sizes": tuple(block_sizes), **xegpu})
    reduce = _compiled(_sum_splits, {"block_sizes": tuple(reduce_block_sizes)})
    return reduce(gemm(a_chunks, b_chunks))


def pad_k(a: Tensor, b: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """``a: [M, K]`` and ``b: [K, N]`` with K zero-padded to ``k`` (``a @ b``
    unchanged: the padded products are zeros). For a K too short for the
    tiles of :data:`GEMM_SMALL_K` (two K steps of 32)."""
    pad = k - a.size(1)
    if pad <= 0:
        return a, b
    return torch.nn.functional.pad(a, (0, pad)), torch.nn.functional.pad(
        b, (0, 0, 0, pad)
    )
