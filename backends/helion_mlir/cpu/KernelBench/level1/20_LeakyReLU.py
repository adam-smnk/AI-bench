import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import leaky_relu


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, negative_slope: float = 0.01, *args, **kwargs):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_relu(x, self.negative_slope)
