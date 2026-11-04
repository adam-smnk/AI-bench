"""Row reductions on Helion's MLIR backend for XPU (KernelBench 23, 24, 38, 39).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it. Its norms run at the single kernels' block sizes: the one-row
block sizes below are lowered and run by that repository's
``tests/test_xpu_row_reduction*.py``.

f32 ``[M, N]`` inputs (4096x393216 and 32768x65536 in the ``mlir-gpu-bench``
variant), in three kernels (:data:`ROW_SEGMENT_COLS`, :func:`row_segments`):

1. per-segment partials over ``x.view(M * S, N / S)``, contiguous row segments
   of ``N / S`` columns (sums of ``|x|`` or ``x * x``; for softmax the
   segment's maximum and its sum of ``exp(x - max)``, in one pass with a
   running maximum per element);
2. the row results from the ``[M, S]`` partials (``sum / n``, ``rsqrt(sum)``,
   the row maximum and ``1 / sum``, or the row maximum and ``log(sum)``);
3. the output, an elementwise kernel of one-row tiles (as the elementwise
   problems: 1D tiles, the column tiles first in the grid) taking the row
   results as ``[M, 1]`` columns.

The two norms take one kernel of one row per workgroup instead, its columns split
across the subgroups, where the input allows (:data:`ONE_ROW_COLS`,
:func:`one_row_blocks`). Softmax gains nothing from it (99.7 ms at one row against
97.3 here) and keeps its three kernels.

Other inputs (no such segments, or not f32) take the single kernels: 64 rows
per workgroup (lighthouse's reduction class: 8 subgroups of 8 rows) and
64-column chunks, multi-pass over the chunks (:data:`REDUCTION_BLOCKS`). The
block sizes divide the input (no masking). The column block is registered
first, so it comes first in the config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor

from .elementwise import block_sizes as elementwise_block_sizes

if TYPE_CHECKING:
    from collections.abc import Callable

REDUCTION_BLOCKS = (64, 64)
_CONFIG = helion.Config(block_sizes=list(REDUCTION_BLOCKS))

ROW_SEGMENT_COLS = (2048, 1024, 512, 256, 128)
"""Row segment lengths of the three-kernel row reductions, in order of
preference (:func:`row_segments`), measured on the Arc B50 2026-11-01 (f32;
device time). Keep in sync with ``ROW_SEGMENT_COLS`` in helion-mlir-backend's
``tests/kernelbench_xpu.py``.

The kernels are bound by DRAM (B50: 222 GB/s reading, 198 GB/s copying). The
single kernels read each row two or three times and write it once inside one
workgroup of 64 rows, and miss both ceilings:

- The output pass stores 2D blocks of 8 rows x 16 f32 per instruction, rows
  of the spec shapes 256 KiB and 1.5 MiB apart: about 112 GB/s read+write,
  as the 2D elementwise tiles before their one-row tiles. Moved into an
  elementwise kernel of one-row tiles (1D, column tiles first in the grid),
  the same pass copies at 195 GB/s (4096x393216: 65 ms, torch's divide too).
- The reduction pass reads 8-row block tiles of 64 columns of rows that far
  apart: 165-180 GB/s for any tile (8 to 256 rows per workgroup, 64 to 512
  columns, 8 to 32 rows per subgroup). The same rows split into contiguous
  segments of 128-2048 columns (``x.view(M * S, N / S)``, no copy) read at
  200-208 GB/s (8192x65536: 12.2 -> 10.1 ms; torch's norm reduction 9.7).
  2048 columns (2048 measured as 1024 and 512) give the fewest partials.
- Softmax reads the row once for the maximum and the sum (a running
  maximum and sum per element: one ``exp`` per element, see
  :func:`_segment_softmax`) and once for the output: two reads and one write
  where the single kernel read three times (and torch's kernel, one
  workgroup per row, three times too).

4096x393216 softmax/log-softmax 196/205 -> 98 ms (torch 130); 8192x65536 L2
norm 48.5 -> 33.6 ms (torch 31.7; device time 31.8 vs 31.6), L1 norm 48.6
-> 33.8 ms (torch 53.5).
"""
ONE_ROW_COLS = (2048, 1024, 512, 256)
"""K tiles of the norms' one-row kernels (the first dividing the columns), each with its
own layout from the backend (``plan_row_reduction``: 32 subgroups of 64 columns for 2048).

Time on the B50 and the B70 follows the threads per row, whatever the tile (Triton, 8192x65536
on the B50, 32 threads: 23.0 ms at 1 row per workgroup, 27.0 at 2, 32.8 at 4, 37.3 at 32;
and 1 row with 16 threads equals 2 rows with 32). So one row, all subgroups on it.
Inferred, not proven: the output pass finds the row in the cache where the first pass left it.

L1 norm, device time, sustained, B50 at 8192x65536 and B70 at 32768x65536 (ms):

  tile  subgroups      B50    B70
  4096  32 of 128     22.3   34.0
  2048  32 of 64      22.3   33.7
  1024  32 of 32      22.9   34.4
  1024  16 of 64      23.2   36.3
   512  16 of 32      22.9   35.8
   512  32 of 16      68.6  107.2
   256  16 of 16      23.4   36.4

2048 is the best on the B70 and level on the B50. The three kernels: 32.6, 48.0."""
_SURFACE_ROWS = 1 << 24
"""A 2D block load's surface has at most 2^24 rows (the hardware field): the
segment view stays below it."""


def row_segments(shape: torch.Size | tuple[int, ...]) -> int | None:
    """The number of row segments ``S`` the three kernels split each row of a
    ``[M, N]`` input into (the first of :data:`ROW_SEGMENT_COLS` dividing
    ``N`` into a multiple of 16 segments: whole block columns for the
    second kernel), or ``None`` for the single kernels: no such segments,
    ``M`` not a multiple of 8 (one subgroup's rows), fewer than 16 rows (the
    row results kernel would be one workgroup, which fails to lower: "host
    code touches device memory") or too many segment rows for a 2D block
    surface."""
    m, n = shape
    if m % 8 or m < 16:
        return None
    for cols in ROW_SEGMENT_COLS:
        if n % cols == 0 and (n // cols) % 16 == 0 and m * (n // cols) <= _SURFACE_ROWS:
            return n // cols
    return None


def one_row_blocks(x: Tensor) -> tuple[int, int] | None:
    """The block sizes ``(cols, 1)`` of the norms' one-row kernels for ``x``
    (:data:`ONE_ROW_COLS`), or ``None``: not f32 (the backend splits f32 sums only), no
    such tile, or one row (a grid of one workgroup fails to lower: "host code touches
    device memory")."""
    m, n = x.shape
    if x.dtype != torch.float32 or m < 2:
        return None
    return next(((cols, 1) for cols in ONE_ROW_COLS if n % cols == 0), None)


def _rows_block(rows: int) -> int:
    """Workgroup rows of a reduction: the largest of 64 to 8 dividing ``rows``
    (8 subgroups of 8 rows, or fewer) into at least two workgroups (a grid
    of one workgroup fails to lower: "host code touches device memory"), or
    8."""
    return next((b for b in (64, 32, 16, 8) if rows % b == 0 and rows // b >= 2), 8)


# --- The single kernels ---------------------------------------------------------------


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
    """Row log-softmax: ``(x - max) - log(sum(exp(x - max)))``, in three
    passes. ``x - max`` first: ``x - (max + log(sum))`` rounds to the ulp of
    ``max`` (rows offset by 1e4 lost 3e-5 of ``max|out|``)."""
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
        log_total = torch.log(total.sum(-1, keepdim=True))
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - row_top) - log_total
    return out


def _l1_norm_kernel(x: Tensor) -> Tensor:
    """``x / mean(|x|, dim=1, keepdim=True)`` (KernelBench 38), in two passes."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        mean = acc.sum(-1, keepdim=True) / n
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] / mean
    return out


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


# --- The three-kernel form: 1. segment partials ---------------------------------------


def _segment_abs_sums(x: Tensor) -> Tensor:
    """``sum(|x|)`` of each row of ``x`` (a segment), as an f32 column."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + torch.abs(x[tile_m, tile_n])
        out[tile_m, :] = acc.sum(-1, keepdim=True)
    return out


def _segment_square_sums(x: Tensor) -> Tensor:
    """``sum(x * x)`` of each row of ``x`` (a segment), as an f32 column."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            acc = acc + v * v
        out[tile_m, :] = acc.sum(-1, keepdim=True)
    return out


def _segment_softmax(x: Tensor) -> tuple[Tensor, Tensor]:
    """The maximum ``t`` of each row of ``x`` (a segment) and its
    ``sum(exp(x - t))``, as f32 columns, in one pass.

    Each element of the carried tile keeps a running maximum ``top`` and sum
    ``total`` of ``exp(v - top)`` over its column chunks (an online softmax): a
    larger ``v`` rescales the sum by ``e = exp(top - v)`` and adds 1, any other
    adds ``e = exp(v - top)``; ``e = exp(-|v - top|)`` either way, one ``exp``
    per element. ``top`` starts at -FLT_MAX, not -inf, so ``v - top`` is never
    ``-inf - -inf``: -inf elements add ``exp(-inf) = 0``, a row of -inf ends
    with the sum 0 (NaN outputs, as torch's). A NaN makes its sum NaN (the
    comparison is false), and the row's results NaN. The row's maximum
    rescales the per-element sums once; the ``where`` (sums are never
    negative; NaN passes) keeps the product and the sum a reduction rather
    than a contraction."""
    m, n = x.shape
    block_n = hl.register_block_size(n)
    tops = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    sums = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_n], -3.4028234663852886e38, dtype=torch.float32)
        total = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            v = x[tile_m, tile_n]
            e = torch.exp(-torch.abs(v - top))
            total = torch.where(v > top, total * e + 1.0, total + e)
            top = torch.maximum(top, v)
        row_top = torch.amax(top, dim=-1, keepdim=True)
        scaled = torch.where(
            total < 0.0,
            hl.zeros([tile_m, block_n], dtype=torch.float32),
            total * torch.exp(top - row_top),
        )
        tops[tile_m, :] = row_top
        sums[tile_m, :] = torch.sum(scaled, dim=-1, keepdim=True)
    return tops, sums


# --- 2. row results from the [M, S] partials ------------------------------------------


def _means(parts: Tensor, n: hl.constexpr) -> Tensor:
    """``sum / n`` of each row of the partial sums ``parts`` (the output
    divides by it: ``n / sum`` overflows for tiny inputs, e.g. ``1e-40 *
    randn``)."""
    m, s = parts.shape
    block_s = hl.register_block_size(s)
    out = torch.empty([m, 1], dtype=torch.float32, device=parts.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            acc = acc + parts[tile_m, tile_s]
        out[tile_m, :] = acc.sum(-1, keepdim=True) / n
    return out


def _inverse_norms(parts: Tensor) -> Tensor:
    """``rsqrt(sum)`` of each row of the partial sums of squares ``parts``."""
    m, s = parts.shape
    block_s = hl.register_block_size(s)
    out = torch.empty([m, 1], dtype=torch.float32, device=parts.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            acc = acc + parts[tile_m, tile_s]
        out[tile_m, :] = torch.rsqrt(acc.sum(-1, keepdim=True))
    return out


def _softmax_rows(tops: Tensor, sums: Tensor) -> tuple[Tensor, Tensor]:
    """The row maximum ``t`` and ``1 / sum(sums * exp(tops - t))`` of segment
    maxima and sums ``[M, S]`` (see :func:`_segment_softmax`)."""
    m, s = tops.shape
    block_s = hl.register_block_size(s)
    row_tops = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    inverses = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_s], float("-inf"), dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            top = torch.maximum(top, tops[tile_m, tile_s])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            part = sums[tile_m, tile_s]
            total = total + torch.where(
                part < 0.0,
                hl.zeros([tile_m, block_s], dtype=torch.float32),
                part * torch.exp(tops[tile_m, tile_s] - row_top),
            )
        row_tops[tile_m, :] = row_top
        inverses[tile_m, :] = 1.0 / total.sum(-1, keepdim=True)
    return row_tops, inverses


def _log_softmax_rows(tops: Tensor, sums: Tensor) -> tuple[Tensor, Tensor]:
    """The row maximum ``t`` and ``log(sum(sums * exp(tops - t)))`` of
    segment maxima and sums ``[M, S]``, two columns: the output subtracts them
    one at a time (:func:`_log_softmax_out`)."""
    m, s = tops.shape
    block_s = hl.register_block_size(s)
    row_tops = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    log_sums = torch.empty([m, 1], dtype=torch.float32, device=tops.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m, block_s], float("-inf"), dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            top = torch.maximum(top, tops[tile_m, tile_s])
        row_top = torch.amax(top, dim=-1, keepdim=True)
        total = hl.zeros([tile_m, block_s], dtype=torch.float32)
        for tile_s in hl.tile(s, block_size=block_s):
            part = sums[tile_m, tile_s]
            total = total + torch.where(
                part < 0.0,
                hl.zeros([tile_m, block_s], dtype=torch.float32),
                part * torch.exp(tops[tile_m, tile_s] - row_top),
            )
        row_tops[tile_m, :] = row_top
        log_sums[tile_m, :] = torch.log(total.sum(-1, keepdim=True))
    return row_tops, log_sums


# --- 3. the output (elementwise, into a preallocated ``out``) -------------------------


def _scale_rows(x: Tensor, scale: Tensor, out: Tensor) -> Tensor:
    """``out = x * scale`` of a row scale ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] * scale[tile_m, :]
    return out


def _divide_rows(x: Tensor, divisor: Tensor, out: Tensor) -> Tensor:
    """``out = x / divisor`` of a row divisor ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] / divisor[tile_m, :]
    return out


def _log_softmax_out(x: Tensor, tops: Tensor, log_sums: Tensor, out: Tensor) -> Tensor:
    """``out = (x - tops) - log_sums`` of row columns ``[M, 1]``: ``x - tops``
    first, exact near the maximum (``x - (tops + log_sums)`` rounds to the ulp
    of ``tops``)."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = (x[tile_m, tile_n] - tops[tile_m, :]) - log_sums[
            tile_m, :
        ]
    return out


def _softmax_out(x: Tensor, tops: Tensor, inverses: Tensor, out: Tensor) -> Tensor:
    """``out = exp(x - tops) * inverses`` of row columns ``[M, 1]``."""
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = (
            torch.exp(x[tile_m, tile_n] - tops[tile_m, :]) * (inverses[tile_m, :])
        )
    return out


_KERNELS: dict[tuple[Callable[..., object], tuple[int, ...]], helion.Kernel] = {}


def _kernel(fn: Callable[..., object], blocks: tuple[int, ...]) -> helion.Kernel:
    """``fn`` as a Helion kernel with ``blocks`` (compiled once per shape)."""
    kernel = _KERNELS.get((fn, blocks))
    if kernel is None:
        kernel = helion.kernel(
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=list(blocks)),
        )(fn)
        _KERNELS[fn, blocks] = kernel
    return kernel


def _segmented(
    x: Tensor,
    partials: Callable[..., object],
    rows: Callable[..., object],
    apply: Callable[..., object],
    *constants: object,
) -> Tensor | None:
    """The three kernels (module docstring) for ``x``, or ``None`` if it has
    no row segments (:func:`row_segments`)."""
    segments = row_segments(x.shape) if x.dtype == torch.float32 else None
    if segments is None:
        return None
    m, n = x.shape
    x = x.contiguous()
    # The output first: the partials (a few MiB) must not split a free cached
    # block the output would fit in.
    out = torch.empty_like(x)
    view = x.view(m * segments, n // segments)
    parts = _kernel(partials, (64, _rows_block(m * segments)))(view)
    if not isinstance(parts, tuple):
        parts = (parts,)
    parts = tuple(part.view(m, segments) for part in parts)
    block_s = next(block for block in (64, 32, 16) if segments % block == 0)
    results = _kernel(rows, (block_s, _rows_block(m)))(*parts, *constants)
    if not isinstance(results, tuple):
        results = (results,)
    _kernel(apply, elementwise_block_sizes(x.shape))(x, *results, out)
    return out


def softmax(x: Tensor) -> Tensor:
    """Softmax over the last dim of ``[M, N]``."""
    out = _segmented(x, _segment_softmax, _softmax_rows, _softmax_out)
    return _softmax_kernel(x) if out is None else out


def log_softmax(x: Tensor) -> Tensor:
    """Log-softmax over the last dim of ``[M, N]``."""
    out = _segmented(x, _segment_softmax, _log_softmax_rows, _log_softmax_out)
    return _log_softmax_kernel(x) if out is None else out


def l1_norm(x: Tensor) -> Tensor:
    """``x`` over the mean of its absolute values along dim 1 of ``[M, N]``."""
    blocks = one_row_blocks(x)
    if blocks is not None:
        return _kernel(_l1_norm_kernel, blocks)(x)
    out = _segmented(x, _segment_abs_sums, _means, _divide_rows, x.shape[1])
    return _kernel(_l1_norm_kernel, REDUCTION_BLOCKS)(x) if out is None else out


def l2_norm(x: Tensor) -> Tensor:
    """``x`` over its L2 norm along dim 1 of ``[M, N]``."""
    blocks = one_row_blocks(x)
    if blocks is not None:
        return _kernel(_l2_norm_kernel, blocks)(x)
    out = _segmented(x, _segment_square_sums, _inverse_norms, _scale_rows)
    return _kernel(_l2_norm_kernel, REDUCTION_BLOCKS)(x) if out is None else out
