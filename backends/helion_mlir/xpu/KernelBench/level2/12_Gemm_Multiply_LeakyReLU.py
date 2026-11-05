import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_multiply_leaky_relu


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_multiply_leaky_relu(
            x, self.gemm.weight, self.gemm.bias, self.multiplier, self.negative_slope
        )
