import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import matmul

# The 256x256 output is 4 workgroups of 128x128 (the default 256x256 tile would be one).
_BLOCK_SIZES = (128, 128, 32)


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul(A, B, block_sizes=_BLOCK_SIZES)
