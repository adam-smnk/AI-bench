import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear


def _epilogue(x):
    x = torch.sigmoid(x) * x
    x = torch.tanh(x)
    x = x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
    return torch.clamp(x, -1.0, 1.0)


class Model(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(
            x,
            self.matmul,
            _epilogue,
            self._linear_cache,
            biases=(self.matmul.bias, self.add_value),
        )
        return x
