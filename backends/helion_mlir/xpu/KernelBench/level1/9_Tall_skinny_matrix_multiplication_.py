import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import matmul

# K = 32: a K step of 16 (DPAS K) keeps the K loop.
_BLOCK_SIZES = (256, 256, 16)


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul(A, B, block_sizes=_BLOCK_SIZES)
