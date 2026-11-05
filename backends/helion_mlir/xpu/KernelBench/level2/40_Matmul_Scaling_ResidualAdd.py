import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_scaling_residual_add


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_scaling_residual_add(
            x, self.matmul.weight, self.matmul.bias, self.scaling_factor
        )
