import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


class Model(nn.Module):
    """``einsum("bijl,lk->bijk")`` as one ``[b*i*j, l] @ [l, k]`` GEMM."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, i, j, inner = A.shape
        return matmul(A.reshape(b * i * j, inner), B).view(b, i, j, B.shape[1])
