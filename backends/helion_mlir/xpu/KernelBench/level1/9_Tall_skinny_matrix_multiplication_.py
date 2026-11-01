import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import GEMM_SMALL_K
from helion_mlir_xpu_utils import matmul
from helion_mlir_xpu_utils import pad_k

# K = 32 is padded to 64 (zeros: the same products) for GEMM_SMALL_K's two K
# steps of 32; with K = 32 the tiles need a K step of 16, and lighthouse's 16
# prefetching subgroups then need 128x128 workgroups (B50: 19.0 ms against
# 15.4 ms padded, which includes the two small pad kernels; torch 15.9 ms).
_K = 64


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul(*pad_k(A, B, _K), **GEMM_SMALL_K)
