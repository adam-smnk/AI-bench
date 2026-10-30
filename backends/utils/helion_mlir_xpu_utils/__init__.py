"""Helper utilities for AI-bench Helion MLIR XPU kernels.

The kernels mirror helion-mlir-backend's ``tests/kernelbench_xpu.py`` and must be
kept in sync with it. Their block sizes are hand-picked from the hardware (see
each module); the XPU target lowers them through XeGPU to XeVM.
"""

from __future__ import annotations

from .attention import ATTENTION_BLOCKS
from .attention import attention
from .elementwise import ELEMENTWISE_BLOCKS
from .elementwise import elu
from .elementwise import hardsigmoid
from .elementwise import hardtanh
from .elementwise import leaky_relu
from .elementwise import relu
from .elementwise import selu
from .elementwise import sigmoid
from .elementwise import softplus
from .elementwise import softsign
from .elementwise import swish
from .elementwise import tanh
from .linear import LINEAR_BLOCKS
from .linear import LINEAR_SKINNY_BLOCKS
from .linear import linear_mish_mish
from .linear import linear_multiply_leaky_relu
from .linear import linear_relu_divide
from .linear import linear_scaling_residual_add
from .linear import linear_sigmoid_scaling_residual_add
from .linear import linear_subtract_multiply_relu
from .linear import linear_swish_divide_clamp_tanh_clamp
from .linear import linear_swish_scaling
from .matmul import GEMM_BLOCKS
from .matmul import GEMM_SQUARE
from .matmul import GEMM_STANDARD
from .matmul import matmul
from .normalization import REDUCTION_BLOCKS
from .normalization import l1_norm
from .normalization import l2_norm
from .normalization import log_softmax
from .normalization import softmax

__all__ = [
    "ATTENTION_BLOCKS",
    "ELEMENTWISE_BLOCKS",
    "GEMM_BLOCKS",
    "GEMM_SQUARE",
    "GEMM_STANDARD",
    "LINEAR_BLOCKS",
    "LINEAR_SKINNY_BLOCKS",
    "REDUCTION_BLOCKS",
    "attention",
    "elu",
    "hardsigmoid",
    "hardtanh",
    "l1_norm",
    "l2_norm",
    "leaky_relu",
    "linear_mish_mish",
    "linear_multiply_leaky_relu",
    "linear_relu_divide",
    "linear_scaling_residual_add",
    "linear_sigmoid_scaling_residual_add",
    "linear_subtract_multiply_relu",
    "linear_swish_divide_clamp_tanh_clamp",
    "linear_swish_scaling",
    "log_softmax",
    "matmul",
    "relu",
    "selu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]
