from types import SimpleNamespace

import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
import torch.nn as nn

from helion_mlir_cpu_utils import column_sum
from helion_mlir_cpu_utils import linear
from helion_mlir_cpu_utils import row_dot
from helion_mlir_cpu_utils import row_sum

_KERNEL_DTYPES = (torch.bfloat16, torch.float32)


class Model(nn.Module):
    """``sum_n(x Wᵀ / 2) * scale`` per row, as ``x · sum_n(W) / 2 * scale``.

    The row sum of a GEMM is the product of its input and the summed weight
    rows, so the GEMM and the row sum become one read of ``W`` and one of
    ``x``. The sum runs on every call: a change through ``.data`` leaves no
    trace in the parameter's ``_version``, and a cached sum would go stale.

    Fast-math semantics (allowed by the user): the sums are reordered, in f32,
    rounded once at the end. Where they stay finite, bf16 results are closer to
    the fp64 result than the reference's, which rounds every GEMM element; a
    cancellation can also leave a finite, correct result where the reference
    overflows. A result that is not finite may come from the reordering alone
    (x = 1e-5 with W = 1e37 overflows the summed weight, x = +-1e20 with W = 1e17
    the partial dot; one infinite x gives +-inf where the reference sums +inf and
    -inf into NaN), so it is recomputed by the GEMM and the row sum.

    A weight of another kernel dtype than the input takes the GEMM and the row
    sum. Inputs no kernel handles (not 2-D, an empty dim, other dtypes) run the
    reference's eager ops, with the reference's results and errors.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self._scaling_factor = scaling_factor
        divide = hl.constexpr(2.0)
        scaling_factor = hl.constexpr(scaling_factor)

        def divide_epilogue(x):
            return x / divide

        def scale_epilogue(x):
            return x * scaling_factor

        def divide_scale_epilogue(x):
            return x / divide * scaling_factor

        self._divide_epilogue = divide_epilogue
        self._scale_epilogue = scale_epilogue
        self._divide_scale_epilogue = divide_scale_epilogue

    def _gemm_sum(self, x):
        """The GEMM and the row sum, repacking the weight on every call. Off the bench
        paths: a weight of another dtype at 128x4096x4096 costs 1.07 ms, not 0.095 with a
        cached packed weight, and the recomputation of a result that is not finite 0.75
        ms, not 0.098."""
        weight = SimpleNamespace(
            weight=self.weight,
            bias=None,
            out_features=self.weight.shape[0],
        )
        # No cached packed weight: this path must see a change through .data too.
        x, _ = linear(x, weight, self._divide_epilogue, None)
        return row_sum(x, self._scale_epilogue)

    def forward(self, x):
        weight = self.weight
        if (
            x.dim() != 2
            or x.dtype not in _KERNEL_DTYPES
            or weight.dtype not in _KERNEL_DTYPES
            or 0 in x.shape
            or 0 in weight.shape
        ):
            out = torch.matmul(x, weight.T) / 2
            return torch.sum(out, dim=1, keepdim=True) * self._scaling_factor
        if weight.dtype != x.dtype:
            return self._gemm_sum(x)
        out = row_dot(x, column_sum(weight.detach()), self._divide_scale_epilogue)
        return out if torch.isfinite(out).all() else self._gemm_sum(x)
