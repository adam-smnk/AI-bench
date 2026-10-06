import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_sum


def _post_logsumexp(x):
    x = torch.log(x)
    x = torch.where(x >= 0, x, x * 0.01)
    x = torch.where(x >= 0, x, x * 0.01)
    x = x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._linear_cache = None

    def forward(self, x):
        x, self._linear_cache = linear(x, self.linear, torch.exp, self._linear_cache)
        return row_sum(x, _post_logsumexp)
