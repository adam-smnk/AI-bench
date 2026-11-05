import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import elu


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return elu(x, self.alpha)
