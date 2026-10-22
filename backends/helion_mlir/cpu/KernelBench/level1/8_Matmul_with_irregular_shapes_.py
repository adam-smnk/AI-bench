import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul

# Tiles (rows, columns, K chunk) of the bf16 kernel tuned per (M, N, K) on a
# 64-thread Xeon 8592+; other shapes use matmul()'s heuristic.
_BLOCK_SIZES = {(8205, 5921, 2949): (544, 256, 736)}


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        block_sizes = _BLOCK_SIZES.get((A.shape[0], B.shape[1], A.shape[1]))
        return matmul(A, B, block_sizes=block_sizes)
