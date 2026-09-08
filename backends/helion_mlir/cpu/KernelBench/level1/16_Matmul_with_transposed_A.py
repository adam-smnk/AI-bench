import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul(A, B, trans_a=True)
