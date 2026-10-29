"""Elementwise Helion MLIR XPU kernels (KernelBench 19-22, 25, 27-32).

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it.

2D bf16 inputs (4096x393216 in the ``mlir-gpu-bench`` variant), computed in f32
like PyTorch's bf16 kernels. ``[rows, cols]`` workgroup tiles of 128x256
(lighthouse kernel_bench's elementwise default); the XPU target infers subgroup
tiles of 32x32 (32 subgroups, 512 threads), 32x32 bf16 block loads and 8x16
block stores. The tiles divide the input (no masking). Helion kernels cannot
close over a function, so each activation is its own kernel. Scalar parameters
are ``hl.constexpr``: compile-time constants, since the backend cannot lower a
dynamic float.
"""

from __future__ import annotations

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor
import torch.nn.functional as F

ELEMENTWISE_BLOCKS = (128, 256)
_CONFIG = helion.Config(block_sizes=list(ELEMENTWISE_BLOCKS))


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _relu_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.relu(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _sigmoid_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.sigmoid(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _tanh_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.tanh(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _swish_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v * torch.sigmoid(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _selu_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (torch.selu(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _hardsigmoid_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.hardsigmoid(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _softplus_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (F.softplus(v)).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _softsign_kernel(x: Tensor) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = (v / (1 + torch.abs(v))).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _leaky_relu_kernel(x: Tensor, negative_slope: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.leaky_relu(v, negative_slope=negative_slope).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _elu_kernel(x: Tensor, alpha: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.elu(v, alpha=alpha).to(x.dtype)
    return out


@helion.kernel(backend="mlir", static_shapes=True, config=_CONFIG)
def _hardtanh_kernel(x: Tensor, min_val: hl.constexpr, max_val: hl.constexpr) -> Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        v = x[tile].to(torch.float32)
        out[tile] = F.hardtanh(v, min_val=min_val, max_val=max_val).to(x.dtype)
    return out


def relu(x: Tensor) -> Tensor:
    """Elementwise ReLU."""
    return _relu_kernel(x)


def sigmoid(x: Tensor) -> Tensor:
    """Elementwise Sigmoid."""
    return _sigmoid_kernel(x)


def tanh(x: Tensor) -> Tensor:
    """Elementwise Tanh."""
    return _tanh_kernel(x)


def swish(x: Tensor) -> Tensor:
    """Elementwise Swish: ``x * sigmoid(x)``."""
    return _swish_kernel(x)


def selu(x: Tensor) -> Tensor:
    """Elementwise SELU."""
    return _selu_kernel(x)


def hardsigmoid(x: Tensor) -> Tensor:
    """Elementwise HardSigmoid."""
    return _hardsigmoid_kernel(x)


def softplus(x: Tensor) -> Tensor:
    """Elementwise Softplus."""
    return _softplus_kernel(x)


def softsign(x: Tensor) -> Tensor:
    """Elementwise Softsign: ``x / (1 + |x|)``."""
    return _softsign_kernel(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Elementwise LeakyReLU: ``x`` if positive, else ``x * negative_slope``."""
    return _leaky_relu_kernel(x, hl.constexpr(negative_slope))


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Elementwise ELU: ``x`` if positive, else ``alpha * (exp(x) - 1)``."""
    return _elu_kernel(x, hl.constexpr(alpha))


def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    """Elementwise HardTanh: ``x`` clamped to ``[min_val, max_val]``."""
    return _hardtanh_kernel(x, hl.constexpr(min_val), hl.constexpr(max_val))
