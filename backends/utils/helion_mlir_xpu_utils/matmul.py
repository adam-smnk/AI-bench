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

from .device import MANY_XE_CORES
from .device import xe_cores

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

GEMM_BLOCKS = (256, 256, 32)
"""256x256 workgroups of 32 subgroups with K steps of 32, except where measured
otherwise on the B50: 2 forces 32x64 subgroup tiles (the 64x32 lighthouse infers
stalls the memory pipe at K = 8192, :data:`GEMM_STANDARD`), and the plain GEMMs
1, 2, 13, 16, 17 and 18 walk their workgroups along a space-filling curve
(:data:`GEMM_SQUARE`). Lighthouse's inference picks 64x32 subgroup tiles for the shapes in
its tuned parameter database (1, 2, 7, 9 and 13) on 256x256 workgroups and 32x64
for the transposed 16-18. Other configs for particular shapes: 7 and 9 (output
store bound) take :data:`GEMM_SMALL_K` (9 with K padded from 32 to 64, the
tiles' two K steps), 6 (K = 524288, a 256x256 output) a split K
(:data:`GEMM_SPLIT_K`), 16 and 18 their own tiles (:data:`GEMM_TRANSPOSED_A`,
:data:`GEMM_TRANSPOSED_BOTH`). Transposed operands (A stored ``[K, M]``, B ``[N, K]``)
are loaded as transposed 16x16 tiles."""
GEMM_SQUARE: dict[str, Any] = {
    "block_sizes": (256, 256, 32),
    "xegpu_sg_tiles": ((32, 64),),
    "xegpu_load_tiles": ((32, 32, 16, 32),),
    "xegpu_sfc_walk": (True,),
    "xegpu_store_write_back": (True,),
}
"""1_Square and 13_Matmul_for_symmetric_matrices (both 4096^3). Keep in sync with
``GEMM_SQUARE`` in helion-mlir-backend's ``tests/kernelbench_xpu.py``.

- 256x256 workgroups along a space-filling curve (``xegpu_sfc_walk``) and write-back
  stores (``xegpu_store_write_back``), replacing 128x512 workgroups in linear order:
  faster on both cards.
- In linear order 256x256 is the worse tile on the B50. The walkers take workgroups in
  about linear order, the first grid dim fastest, so a wave in that order reads a whole
  column of M blocks. Why the curve helps is inferred: by arithmetic a window of 16
  workgroups reads twice the B50's 18 MiB L2 of operands in linear order and less than it
  along the curve. No L2 counters exist here. The same arithmetic doesn't explain the
  other plain GEMMs, below.
- Other workgroup shapes on the curve don't beat it on the B50, nor do subgroup tiles, load
  tiles or prefetch depths.
- Subgroup tiles 32x64 (8x4 subgroups, 128 of the 256 registers for the accumulator) and
  loads A 32x32, B 16x32, as lighthouse's tuned parameters.

Pass as ``matmul(a, b, **GEMM_SQUARE)``."""
GEMM_STANDARD: dict[str, Any] = {
    "block_sizes": (256, 256, 32),
    "xegpu_sg_tiles": ((32, 64),),
    "xegpu_load_tiles": ((8, 32, 32, 32),),
    "xegpu_prefetch_tiles": ((16, 16, 16, 16),),
    "xegpu_prefetch_depths": ((1, 1),),
    "xegpu_sfc_walk": (True,),
    "xegpu_store_write_back": (True,),
}
"""2_Standard (M 2048, N 4096, K 8192) on the Arc B50. Keep in sync with
``GEMM_STANDARD`` in helion-mlir-backend's ``tests/kernelbench_xpu.py``.

- 256x256 workgroups (128 of them) along a space-filling curve, with write-back stores,
  faster on both cards. By arithmetic a window of 16 workgroups still reads more than the
  B50's L2 along the curve, and it gains anyway: the mechanism isn't established here.
- Subgroup tiles 32x64 (8x4 subgroups) in place of the 64x32 lighthouse infers, which
  collapses here: a memory-pipe stall with 64 strided A rows per subgroup at K = 8192
  (EU stall sampling); it is not shown to be a pitch effect alone.
- Loads A 8x32, B 32x32 (lighthouse's tuned parameters for this shape) and
  prefetches of 16x16 at depth 1 (also lighthouse's): the gain is from the
  prefetch tiles. Prefetch depth 2 is equal to depth 1.
- B50-specific, like the tiles of ``GEMM_SQUARE``: 128 workgroups are 8 waves
  of 16 on 16 Xe cores.
- Bit-identical to ``torch.matmul``.

Pass as ``matmul(a, b, **GEMM_STANDARD)``."""

GEMM_TRANSPOSED_A: dict[str, Any] = {
    "block_sizes": GEMM_BLOCKS,
    "xegpu_load_tiles": ((16, 16, 16, 32),),
    "xegpu_sfc_walk": (True,),
    "xegpu_store_write_back": (True,),
}
"""16_Matmul_with_transposed_A (2048x4096x8192). Pass with ``trans_a=True``.

The transposed A is read in 16x16 tiles anyway, so B is read in the same K
steps: 8 block loads per 32 DPAS instead of 12, the wide moves unchanged. Cycles
stay the same on the B50, which is at its 70 W cap, and its clock rises, so the
wall time falls. The B70 ties. Other load tiles, workgroup shapes and 64x32 subgroup
tiles don't help. A space-filling curve with write-back stores helps on both cards."""

GEMM_TRANSPOSED_B: dict[str, Any] = {
    "block_sizes": GEMM_BLOCKS,
    "xegpu_sfc_walk": (True,),
    "xegpu_store_write_back": (True,),
}
"""17_Matmul_with_transposed_B (2048x4096x8192). Pass with ``trans_b=True``. A
space-filling curve with write-back stores helps on both cards."""

GEMM_TRANSPOSED_BOTH: dict[str, Any] = {
    "block_sizes": GEMM_BLOCKS,
    "xegpu_sg_tiles": ((64, 32),),
    "xegpu_sfc_walk": (True,),
    "xegpu_store_write_back": (True,),
}
"""18_Matmul_with_transposed_both (2048x4096x8192). Pass with ``trans_a=True,
trans_b=True``.

64x32 subgroup tiles, where lighthouse's inference gives 32x64: fewer cycles on both
cards. It's specific to this form: the same tile loses on 17_Matmul_with_transposed_B
and on 16. Load tiles and prefetch depths on top changed nothing. A space-filling curve
with write-back stores helps on both cards."""

GEMM_SMALL_K: dict[str, Any] = {
    "block_sizes": (64, 128, 32),
    "xegpu_sg_tiles": ((8, 64),),
}
"""7_Matmul_with_small_K_dimension_ (32768x32768x64) and, with K padded to
64, 9_Tall_skinny_matrix_multiplication_ (32768x32768x32), on the Arc B50.
Keep in sync with ``GEMM_SMALL_K`` in helion-mlir-backend's
``tests/kernelbench_xpu.py``.

- These GEMMs are bound by the output store: 2 GiB of bf16 written for 8 MiB
  of operands (which stay in the 18 MB L2), little DPAS work. The bound is the
  write bandwidth.
- The grid runs the column tiles first (the backend's ``columns_first`` rule
  for GEMMs that write more than they read): the workgroups resident at a
  time write neighbouring tiles of the same rows. Rows first, every resident
  workgroup writes rows in other DRAM pages, which is far slower.
- 64x128 workgroups of 16 subgroups of 8x64 (8x2): each subgroup writes 8 rows
  of 128 contiguous bytes (four 8x16 block stores side by side), a workgroup
  64 rows of 256 bytes. 16-bit 2D block stores write 32-byte rows, half a
  cache line; fewer rows per subgroup and per workgroup and wider rows per
  subgroup measured faster, columns first. 64 rows are the minimum: lighthouse
  wants 16 prefetching subgroups (64 rows x 32 K = 16 8x16 prefetch tiles of A),
  and at least 16 subgroups per workgroup (64x64 / 8x64 is rejected).
- The 16-bit store width still costs: the same tiles write faster with an f32
  output. A 64-byte-row bf16 store (IGC has
  ``intel_sub_group_2d_block_write_16b_8r32x1c``; XeVM takes 16-bit ``tile_width``
  16 only) needs bf16 pairs packed per work item, which the DPAS accumulator
  layout does not give.
- K steps of 32 (2 per tile), as lighthouse requires a K loop.
- Bit-identical to ``torch.matmul``.

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

- The GEMM reads 512 MiB of operands for a 256x256 output: memory-bound.
- Without a split, the output is 4 workgroups of 128x128 for 16 Xe cores. Split
  over K into 32 chunks of 16384, it is 32 workgroups of 256x256 (two waves of
  one per Xe core, 32 subgroups of 32x64, A read 32x32 per block read), each
  reading its chunk of A and B once from DRAM, writing f32 partial sums
  ``[32, 256, 256]`` (8 MiB); a second, small kernel adds them into the bf16
  result (8-row tiles, 32 workgroups).
- The chunks are contiguous views (``A.view(M, S, K/S)``): a kernel argument
  is a contiguous buffer, so a strided ``[S, M, K/S]`` view would be copied
  whole on every call.
- Prefetch 4 K steps ahead for A (its rows are 1 MiB apart: a K step reads
  64 bytes of each of 256 rows), 2 for B: better than equal depths, and deeper
  puts too many prefetches in flight. Other split counts and 128x256 workgroups
  are slower.
- Normalized error against an f32 reference as torch's.

Pass as ``matmul_split_k(a, b, **gemm_split_k(a.device))``."""


def gemm_split_k(device: torch.device) -> dict[str, Any]:
    """:data:`GEMM_SPLIT_K` for the card of ``device``. 32 Xe cores (B70) drop the
    prefetch depths, which cost there; equal depths measure as none given, and 64
    splits are no gain. The B50 keeps them."""
    if xe_cores(device) < MANY_XE_CORES:
        return GEMM_SPLIT_K
    return {k: v for k, v in GEMM_SPLIT_K.items() if k != "xegpu_prefetch_depths"}


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
