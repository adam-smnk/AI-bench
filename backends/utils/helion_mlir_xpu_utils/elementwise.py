"""Elementwise Helion MLIR XPU kernels (KernelBench 19-22, 25, 27-32).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it.

2D bf16 inputs (4096x393216 in the ``mlir-gpu-bench`` variant), computed in f32
like PyTorch's bf16 kernels, in one-row workgroup tiles
(:data:`ELEMENTWISE_BLOCKS`, :func:`block_sizes`): the XPU target loads and
stores each as a 1D tile of contiguous elements, 8 bf16 (16 bytes) per work
item, and runs the column tiles first in the grid. The tiles divide the input
(no masking). Helion kernels cannot close over a function, so each activation
is its own kernel. Scalar parameters are ``hl.constexpr``: compile-time
constants, since the backend cannot lower a dynamic float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
from helion_mlir_backend._compiler.mlir_transforms.xegpu_elementwise import (
    default_sg_tile,
)
import torch
from torch import Tensor
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

ELEMENTWISE_BLOCKS = (1, 8192)
"""Block sizes ``[rows, cols]`` of the bf16 4096x393216 problems on the Arc B50,
measured 2026-10-31. Keep in sync with ``ELEMENTWISE_BLOCKS`` in
helion-mlir-backend's ``tests/kernelbench_xpu.py`` (the full rationale).

The kernels move 4 bytes per element and are bound by DRAM: the card's
read+write copy ceiling is 198 GB/s, which torch reaches.

- One row per workgroup. A 2D tile stores bf16 in 2D block writes of 16
  elements per row (32 bytes, half a cache line: the 16-bit limit of the
  instruction), and that bounds 2D tiles (columns-first grid: 2 rows 172
  GB/s, 8 rows 149, the former 128x256 tiles 115, and 40 with the old
  rows-first grid). A one-row tile is a 1D tile of contiguous
  elements: gathers and scatters of 8 bf16 (16 bytes) per work item, 256
  contiguous bytes per subgroup instruction. With the column tiles first in
  the grid (the backend does it for elementwise kernels) 196 GB/s (99 %).
- 8192 columns: 8 subgroups of 1024 elements per workgroup, 16 KB in and 16 KB
  out. 1024 to 32768 columns measured the same; 8192 divides the spec's
  393216 columns. Other shapes: :func:`block_sizes`.
- 29_Softplus runs at about 147 GB/s at any tile width, against torch's 191: a
  gap of the backend's math lowering (one scalar f32 ``log1p`` call per
  element), not of the tiles.
"""
_MAX_COLS = ELEMENTWISE_BLOCKS[1]
_MIN_COLS = 16
"""A 1D tile is a multiple of 16 elements (one per work item of a subgroup)."""


def block_sizes(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    """The block sizes for an input of ``shape``: one row (block 1 on every dim
    but the last) and the largest multiple of 16 of at most 8192 columns that
    divides the last dim and has an XPU layout (a one-row tile folds to a 1D
    tile, which must split into at most 32 subgroups of at most 2048
    elements: e.g. 4112 = 16 x 257 has none, so 4112 columns take tiles of
    16). :data:`ELEMENTWISE_BLOCKS` at the spec shape."""
    cols = shape[-1]
    if cols % _MIN_COLS:
        raise ValueError(
            f"last dim {cols} is not a multiple of {_MIN_COLS}: no XPU elementwise tile"
        )
    block = next(
        b
        for b in range(min(_MAX_COLS, cols), 0, -_MIN_COLS)
        if cols % b == 0 and default_sg_tile((b,)) is not None
    )
    return (1,) * (len(shape) - 1) + (block,)


_KERNELS: dict[tuple[Callable[..., Tensor], tuple[int, ...]], helion.Kernel] = {}


def _run(fn: Callable[..., Tensor], x: Tensor, *constants: object) -> Tensor:
    """``fn`` as a Helion kernel with the block sizes of ``x`` (one kernel per
    block sizes, each compiled once per shape), called on ``x``."""
    blocks = block_sizes(x.shape)
    kernel = _KERNELS.get((fn, blocks))
    if kernel is None:
        kernel = helion.kernel(
            backend="mlir",
            static_shapes=True,
            config=helion.Config(block_sizes=list(blocks)),
        )(fn)
        _KERNELS[fn, blocks] = kernel
    return kernel(x, *constants)


def _relu_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.relu(v)).to(x.dtype)
    return out


def _sigmoid_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sigmoid(v)).to(x.dtype)
    return out


def _tanh_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.tanh(v)).to(x.dtype)
    return out


def _swish_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v * torch.sigmoid(v)).to(x.dtype)
    return out


def _selu_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.selu(v)).to(x.dtype)
    return out


def _hardsigmoid_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.hardsigmoid(v)).to(x.dtype)
    return out


def _softplus_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.softplus(v)).to(x.dtype)
    return out


def _softsign_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v / (1 + torch.abs(v))).to(x.dtype)
    return out


def _leaky_relu_kernel(x: Tensor, negative_slope: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.leaky_relu(v, negative_slope=negative_slope).to(x.dtype)
    return out


def _elu_kernel(x: Tensor, alpha: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.elu(v, alpha=alpha).to(x.dtype)
    return out


def _hardtanh_kernel(x: Tensor, min_val: hl.constexpr, max_val: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.hardtanh(v, min_val=min_val, max_val=max_val).to(x.dtype)
    return out


def relu(x: Tensor) -> Tensor:
    """Elementwise ReLU."""
    return _run(_relu_kernel, x)


def sigmoid(x: Tensor) -> Tensor:
    """Elementwise Sigmoid."""
    return _run(_sigmoid_kernel, x)


def tanh(x: Tensor) -> Tensor:
    """Elementwise Tanh."""
    return _run(_tanh_kernel, x)


def swish(x: Tensor) -> Tensor:
    """Elementwise Swish: ``x * sigmoid(x)``."""
    return _run(_swish_kernel, x)


def selu(x: Tensor) -> Tensor:
    """Elementwise SELU."""
    return _run(_selu_kernel, x)


def hardsigmoid(x: Tensor) -> Tensor:
    """Elementwise HardSigmoid."""
    return _run(_hardsigmoid_kernel, x)


def softplus(x: Tensor) -> Tensor:
    """Elementwise Softplus."""
    return _run(_softplus_kernel, x)


def softsign(x: Tensor) -> Tensor:
    """Elementwise Softsign: ``x / (1 + |x|)``."""
    return _run(_softsign_kernel, x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Elementwise LeakyReLU: ``x`` if positive, else ``x * negative_slope``."""
    return _run(_leaky_relu_kernel, x, hl.constexpr(negative_slope))


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Elementwise ELU: ``x`` if positive, else ``alpha * (exp(x) - 1)``."""
    return _run(_elu_kernel, x, hl.constexpr(alpha))


def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    """Elementwise HardTanh: ``x`` clamped to ``[min_val, max_val]``."""
    return _run(_hardtanh_kernel, x, hl.constexpr(min_val), hl.constexpr(max_val))
