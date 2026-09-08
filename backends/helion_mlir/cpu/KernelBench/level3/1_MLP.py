import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import identity_epilogue
from helion_mlir_cpu_utils import matmul


def _linear(x: torch.Tensor, layer: nn.Linear, epilogue) -> torch.Tensor:
    return matmul(x, layer.weight, trans_b=True, bias=layer.bias, epilogue=epilogue)


class Model(nn.Module):
    """KernelBench-compatible wrapper"""

    def __init__(self, input_size, layer_sizes, output_size, *args, **kwargs):
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight packing and bias+ReLU happen inside matmul() on every call
        # (no pre-packing, no separate elementwise pass).
        for layer in self.layers[:-1]:
            x = _linear(x, layer, torch.relu)
        return _linear(x, self.layers[-1], identity_epilogue)
