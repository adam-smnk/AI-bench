"""``nn.Linear`` with its epilogue fused in, on Helion's MLIR backend for XPU
(KernelBench level 2: 9, 12, 29, 40, 59, 63, 70, 81).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py``
(``LEVEL2_PROBLEMS``) and must be kept in sync with it.

``epilogue(x @ weight.T + bias)`` of bf16 ``x: [M, K]``, ``weight: [N, K]`` (as
``nn.Linear`` stores it: the GEMM reads B transposed) and ``bias: [N]``, in one
kernel per problem (a Helion kernel cannot close over an epilogue): the f32
accumulator plus the bias row (broadcast over the tile's rows), the epilogue in
f32, one rounding to bf16 at the store. Scalar parameters are ``hl.constexpr``,
like the elementwise kernels'. The block sizes divide ``M``, ``N`` and ``K`` (no
masking): ``M`` a multiple of 256 (or 128 below 256), ``N`` of 256, ``K`` of 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

LINEAR_BLOCKS = (256, 256, 32)
"""Level 2 GEMM block sizes ``[M tile, N tile, K tile]``, picked from the hardware
(Xe2: DPAS 8x16x16 bf16 -> f32, 16-lane subgroups, at most 32 subgroups per
workgroup with the large register file, 2D block reads of at most 32 rows x 64
bytes) and lighthouse's GEMM parameters for these shapes (its selector's
workgroup and subgroup tiles; the K step and A loads of AI-bench's tuned
``kb_params_level2-*.json`` for the lighthouse path):

- 256x256 workgroups of 32 subgroups (inferred ``[8, 4]`` subgroups of 32x64:
  4x4 DPAS blocks). Per subgroup the f32 accumulator takes 128 of the 256
  64-byte registers, a K step's A (32x32) and B (32x64) bf16 tiles 32 + 64:
  the large register file is needed, and 32 subgroups of it fill one Xe core
  (8 vector engines x 4 threads). Per K step a workgroup reads 2x256x32 bf16
  for 256x256x32 MACs: 128 flops per byte. The grid of 1024x8192 outputs is
  4x32 (128 workgroups: 4 waves on B70's 32 Xe cores), of 16384x4096 64x16.
- K steps of 32: two DPAS K steps per load, and an A tile of 32x32 bf16 (64
  bytes wide) per subgroup, read with one 2D block read (``xegpu_load_tiles``
  ``[32, 32, ...]``, as lighthouse's tuned parameters; inferred would be eight
  8x16 reads). B (the weight, stored ``[N, K]``) is read transposed in 16x16
  tiles (32-bit pairs: the transposed block read takes 32-bit elements, 8
  wide), lighthouse's choice.
- The epilogue runs once on the subgroup's 32x64 f32 accumulator in registers
  (small next to the 8192/32 = 256 K steps, but its temporaries share the 32
  registers the K loop leaves); the bias row is a gather of the subgroup's 64
  columns (4 elements per lane), once per workgroup.
"""
LINEAR_SKINNY_BLOCKS = (128, 256, 32)
"""59's ``M = 128`` (128 x 32768 x 32768): one 128-row workgroup tile, so the
weight (2 GiB of bf16, the dominant traffic at 128 flops per byte) is read
exactly once over the 1x128 grid (128 workgroups). 32 subgroups of 16x64
(inferred ``[8, 4]``, as lighthouse's selector and tuned parameters), A tiles
16x32 in one 2D block read (the tuned parameters read 8x16)."""
_A_LOAD = (32, 32, 16, 16)
_SKINNY_A_LOAD = (16, 32, 16, 16)
"""``xegpu_load_tiles`` ``[a_m, a_k, b_k, b_n]``: A in one block read per
subgroup tile (32x32, or 16x32 for the 128-row tiles), B transposed in 16x16."""


def _linear_subtract_multiply_relu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    subtract_value: hl.constexpr,
    multiply_value: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 9: ``relu((linear(x) - subtract) * multiply)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = torch.relu((y - subtract_value) * multiply_value).to(x.dtype)
    return out


def _linear_multiply_leaky_relu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    multiplier: hl.constexpr,
    negative_slope: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 12: ``leaky_relu(linear(x) * multiplier)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = (acc + bias[tn]) * multiplier
        out[tm, tn] = F.leaky_relu(y, negative_slope=negative_slope).to(x.dtype)
    return out


def _linear_mish_mish(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """KernelBench level 2 29: ``mish(mish(linear(x)))``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = F.mish(F.mish(y)).to(x.dtype)
    return out


def _linear_scaling_residual_add(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    scaling_factor: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 40: ``y * scaling_factor + y`` of ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (y * scaling_factor + y).to(x.dtype)
    return out


def _linear_swish_scaling(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    scaling_factor: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 59: ``y * sigmoid(y) * scaling_factor`` of
    ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (y * torch.sigmoid(y) * scaling_factor).to(x.dtype)
    return out


def _linear_relu_divide(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    divisor: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 63: ``relu(linear(x)) / divisor``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (torch.relu(y) / divisor).to(x.dtype)
    return out


def _linear_sigmoid_scaling_residual_add(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    scaling_factor: hl.constexpr,
) -> Tensor:
    """KernelBench level 2 70: ``sigmoid(y) * scaling_factor + y`` of
    ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        out[tm, tn] = (torch.sigmoid(y) * scaling_factor + y).to(x.dtype)
    return out


def _linear_swish_divide_clamp_tanh_clamp(
    x: Tensor, weight: Tensor, bias: Tensor
) -> Tensor:
    """KernelBench level 2 81: ``clamp(tanh(clamp(swish(y) / 2, -1, 1)), -1, 1)``
    of ``y = linear(x)``."""
    m, k = x.size()
    n, _ = weight.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], weight[tn, tk].T)
        y = acc + bias[tn]
        y = torch.clamp(y * torch.sigmoid(y) / 2.0, min=-1.0, max=1.0)
        out[tm, tn] = torch.clamp(torch.tanh(y), min=-1.0, max=1.0).to(x.dtype)
    return out


_KERNELS: dict[tuple[Callable[..., Tensor], bool], helion.Kernel] = {}


def _run(
    fn: Callable[..., Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *scalars: float,
) -> Tensor:
    """``fn`` with the block sizes for ``x``'s rows (:data:`LINEAR_SKINNY_BLOCKS`
    below 256 rows, else :data:`LINEAR_BLOCKS`), the scalars as ``hl.constexpr``.
    ``weight`` and ``bias`` are taken in ``x``'s dtype; no bias adds zeros."""
    skinny = x.size(0) < LINEAR_BLOCKS[0]
    key = (fn, skinny)
    if key not in _KERNELS:
        blocks, a_load = (
            (LINEAR_SKINNY_BLOCKS, _SKINNY_A_LOAD)
            if skinny
            else (LINEAR_BLOCKS, _A_LOAD)
        )
        _KERNELS[key] = helion.kernel(
            fn,
            backend="mlir",
            static_shapes=True,
            config=helion.Config(
                block_sizes=list(blocks), xegpu_load_tiles=[list(a_load)]
            ),
        )
    weight = weight.detach().to(x.dtype)
    if bias is None:
        bias = torch.zeros(weight.size(0), dtype=x.dtype, device=x.device)
    bias = bias.detach().to(x.dtype)
    return _KERNELS[key](x, weight, bias, *(hl.constexpr(value) for value in scalars))


def linear_subtract_multiply_relu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    subtract_value: float,
    multiply_value: float,
) -> Tensor:
    """KernelBench level 2 9: ``relu((linear(x) - subtract_value) * multiply_value)``."""
    return _run(
        _linear_subtract_multiply_relu, x, weight, bias, subtract_value, multiply_value
    )


def linear_multiply_leaky_relu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    multiplier: float,
    negative_slope: float,
) -> Tensor:
    """KernelBench level 2 12: ``leaky_relu(linear(x) * multiplier, negative_slope)``."""
    return _run(
        _linear_multiply_leaky_relu, x, weight, bias, multiplier, negative_slope
    )


def linear_mish_mish(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """KernelBench level 2 29: ``mish(mish(linear(x)))``."""
    return _run(_linear_mish_mish, x, weight, bias)


def linear_scaling_residual_add(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    scaling_factor: float,
) -> Tensor:
    """KernelBench level 2 40: ``y * scaling_factor + y`` of ``y = linear(x)``."""
    return _run(_linear_scaling_residual_add, x, weight, bias, scaling_factor)


def linear_swish_scaling(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    scaling_factor: float,
) -> Tensor:
    """KernelBench level 2 59: ``y * sigmoid(y) * scaling_factor`` of ``y = linear(x)``."""
    return _run(_linear_swish_scaling, x, weight, bias, scaling_factor)


def linear_relu_divide(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    divisor: float,
) -> Tensor:
    """KernelBench level 2 63: ``relu(linear(x)) / divisor``."""
    return _run(_linear_relu_divide, x, weight, bias, divisor)


def linear_sigmoid_scaling_residual_add(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    scaling_factor: float,
) -> Tensor:
    """KernelBench level 2 70: ``sigmoid(y) * scaling_factor + y`` of ``y = linear(x)``."""
    return _run(_linear_sigmoid_scaling_residual_add, x, weight, bias, scaling_factor)


def linear_swish_divide_clamp_tanh_clamp(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """KernelBench level 2 81: ``clamp(tanh(clamp(swish(y) / 2, -1, 1)), -1, 1)`` of ``y = linear(x)``."""
    return _run(_linear_swish_divide_clamp_tanh_clamp, x, weight, bias)
