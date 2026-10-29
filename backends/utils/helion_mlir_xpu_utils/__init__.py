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
from .matmul import GEMM_BLOCKS
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
    "REDUCTION_BLOCKS",
    "attention",
    "elu",
    "hardsigmoid",
    "hardtanh",
    "l1_norm",
    "l2_norm",
    "leaky_relu",
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
