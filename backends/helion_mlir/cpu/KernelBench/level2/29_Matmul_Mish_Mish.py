import helion_mlir_backend  # noqa: F401
import torch.nn as nn
import torch.nn.functional as functional

from helion_mlir_cpu_utils import linear


def _mish_twice(x):
    return functional.mish(functional.mish(x))


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(x, self.linear, _mish_twice, self._linear_cache)
        return x
