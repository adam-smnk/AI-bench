import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_swish_divide_clamp_tanh_clamp


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_swish_divide_clamp_tanh_clamp(x, self.gemm.weight, self.gemm.bias)
