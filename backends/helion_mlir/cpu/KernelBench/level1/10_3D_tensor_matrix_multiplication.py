import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul


class Model(nn.Module):
    """``[N, M, K] @ [K, L]`` as one ``[N*M, K] @ [K, L]`` GEMM."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        n, m, k = A.shape
        return matmul(A.reshape(n * m, k), B).view(n, m, B.shape[1])
