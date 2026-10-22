import helion_mlir_backend  # noqa: F401
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_sum


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._linear_cache = None

    def forward(self, x):
        y, self._linear_cache = linear(x, self.linear, cache=self._linear_cache)
        # The max, mean and both logsumexps after the sum reduce a singleton dim.
        return row_sum(y)
