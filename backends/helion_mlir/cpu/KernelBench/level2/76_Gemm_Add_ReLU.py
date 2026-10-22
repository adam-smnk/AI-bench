import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(
            x,
            self.gemm,
            torch.relu,
            self._linear_cache,
            biases=(self.bias,),
        )
        return x
