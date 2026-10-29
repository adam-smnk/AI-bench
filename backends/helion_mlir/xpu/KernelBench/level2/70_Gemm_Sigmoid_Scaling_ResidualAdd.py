import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_sigmoid_scaling_residual_add


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_sigmoid_scaling_residual_add(
            x, self.gemm.weight, self.gemm.bias, self.scaling_factor
        )
