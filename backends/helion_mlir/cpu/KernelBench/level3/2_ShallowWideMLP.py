import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import identity_epilogue
from helion_mlir_cpu_utils import linear


class Model(nn.Module):
    """Linear + ReLU layers, each one fused kernel (``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=1``
    reuses packed weights across calls, else every call packs them)."""

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        sizes = [input_size, *hidden_layer_sizes, output_size]
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(sizes, sizes[1:]))
        self._caches = [None] * len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            epilogue = identity_epilogue if i == last else torch.relu
            x, self._caches[i] = linear(x, layer, epilogue, cache=self._caches[i])
        return x
