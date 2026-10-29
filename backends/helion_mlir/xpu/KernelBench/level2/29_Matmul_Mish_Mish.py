import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import linear_mish_mish


class Model(nn.Module):
    """KernelBench-compatible wrapper: the GEMM and its epilogue in one kernel."""

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_mish_mish(x, self.linear.weight, self.linear.bias)
