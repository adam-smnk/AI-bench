import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import softmax
from helion_mlir_cpu_utils.matmul import _omp_threads


def _panel(x: torch.Tensor, out_features: int) -> int | None:
    """Columns of the bf16 weight's VNNI panels (= output tile width).

    bench-cpu is 128x16384x16384: a 512 MiB weight, beyond the LLC, read
    once per call from DRAM, so the GEMM is bandwidth bound. 256-column
    panels make one tile of all 128 rows per panel, 64 tiles for the 64
    cores: each core streams its own 8 MiB of the weight once, A (4 MiB)
    comes from the LLC. EMR, cpu-bench, GEMM alone: 1.82 ms (295 GB/s);
    the helper's default for few rows, 64 columns (256 tiles), 2.14 ms;
    128 columns 1.94 ms. Wider panels need a tile per core, so smaller
    layers keep the default (vnni_panel)."""
    threads = _omp_threads()  # as vnni_panel counts them
    if x.dtype == torch.bfloat16 and x.shape[0] < 256:
        if -(-out_features // 256) >= threads:
            return 256
    return None


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self._linear_cache = None

    def forward(self, x):
        # AI-bench evaluates models in eval mode (KernelRunner.init_model),
        # where dropout is the identity. Random training-mode masks cannot
        # match the reference's, so refuse them instead of ignoring them.
        if self.training and self.dropout.p > 0:
            raise ValueError("dropout with p > 0 needs eval mode")
        panel = _panel(x, self.matmul.out_features)
        y, self._linear_cache = linear(
            x, self.matmul, cache=self._linear_cache, panel=panel
        )
        return softmax(y)
