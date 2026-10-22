import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import matmul

# Tiles (rows, columns, K chunk) of the bf16 kernel tuned per (M, N, K) on a
# 64-thread Xeon 8592+; other shapes use matmul()'s heuristic.
_BLOCK_SIZES = {(2048, 4096, 8192): (512, 256, 512)}


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        k, m = A.shape
        block_sizes = _BLOCK_SIZES.get((m, B.shape[0], k))
        return matmul(A, B, trans_a=True, trans_b=True, block_sizes=block_sizes)
