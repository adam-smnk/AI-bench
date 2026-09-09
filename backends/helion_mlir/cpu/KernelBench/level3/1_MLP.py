import os

import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import identity_epilogue
from helion_mlir_cpu_utils import matmul
from helion_mlir_cpu_utils import matmul_prepacked_b
from helion_mlir_cpu_utils import pack_b_blocked_t

_CACHE_PREPACKED_WEIGHTS_ENV = "HELION_MLIR_CACHE_PREPACKED_WEIGHTS"


def _linear(x: torch.Tensor, layer: nn.Linear, epilogue) -> torch.Tensor:
    return matmul(x, layer.weight, trans_b=True, bias=layer.bias, epilogue=epilogue)


def _parameter_key(parameter: torch.Tensor) -> tuple:
    return (
        parameter.data_ptr(),
        parameter._version,
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
    )


class Model(nn.Module):
    """KernelBench-compatible MLP.

    Set ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=1`` to model deployment with
    constant weights: the first forward packs each weight and later forwards
    reuse it. By default every forward retains the full runtime packing cost.
    """

    def __init__(self, input_size, layer_sizes, output_size, *args, **kwargs):
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.layers = nn.ModuleList(layers)
        self._prepacked_layers = None

    def _get_prepacked_layers(self, x: torch.Tensor):
        keys = tuple(
            (_parameter_key(layer.weight), _parameter_key(layer.bias))
            for layer in self.layers
        )
        cache_key = (keys, x.dtype, x.device)
        if self._prepacked_layers is None or self._prepacked_layers[0] != cache_key:
            packed_layers = []
            for layer in self.layers:
                weight = layer.weight.detach().to(dtype=x.dtype, device=x.device)
                bias = layer.bias.detach().to(dtype=x.dtype, device=x.device)
                packed_layers.append(
                    (pack_b_blocked_t(weight), bias, int(layer.out_features))
                )
            self._prepacked_layers = (cache_key, packed_layers)
        return self._prepacked_layers[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if os.environ.get(_CACHE_PREPACKED_WEIGHTS_ENV, "").strip() == "1":
            packed_layers = self._get_prepacked_layers(x)
            for packed_weight, bias, out_features in packed_layers[:-1]:
                x = matmul_prepacked_b(
                    x,
                    packed_weight,
                    n=out_features,
                    bias=bias,
                    epilogue=torch.relu,
                )
            packed_weight, bias, out_features = packed_layers[-1]
            return matmul_prepacked_b(
                x,
                packed_weight,
                n=out_features,
                bias=bias,
                epilogue=identity_epilogue,
            )

        # Default comparison path: pack weights on every timed call.
        for layer in self.layers[:-1]:
            x = _linear(x, layer, torch.relu)
        return _linear(x, self.layers[-1], identity_epilogue)
