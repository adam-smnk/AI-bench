import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_subtract_multiply_relu


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_subtract_multiply_relu(
            x,
            self.linear.weight,
            self.linear.bias,
            self.subtract_value,
            self.multiply_value,
        )
