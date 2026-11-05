import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_xpu_utils import mlp


class Model(nn.Module):
    """Linear + ReLU layers, 17 of them in one kernel from one entry on the weights as
    stored (no copy, so ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS`` doesn't apply)."""

    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)  # the reference's parameter names
        self._caches = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mlp(x, self.network, self._caches)
