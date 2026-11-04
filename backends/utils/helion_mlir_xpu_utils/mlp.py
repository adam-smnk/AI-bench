"""KernelBench level 3 MLPs (1, 2, 3) on Helion's MLIR backend for XPU.

Every layer but the last is ``relu(x @ weight.T + bias)`` of a GEMM fused with its
activation: the f32 accumulator plus the bias row, the ReLU in f32, one rounding
to bf16 at the store, read by the next layer. Torch's rounding inside a layer wasn't
traced; its error and these kernels' against fp64 are in the same range.

**Weights, 1 and 2.** At ``M = 128`` the weights are most of the traffic (128 flops
per byte), so the kernels are bound by reading them. There are two forms, chosen by
:func:`~.linear.cache_transposed_weights`:

- **Cached, the default:** a transposed copy, ``[K, N]``, made on the first call and
  reused while the parameter looks unchanged (:func:`transposed_weight`). The K steps
  read a few long runs instead of one short run per output column (the reason is
  inferred, the gain measured). It costs a second copy (L3/2: 8.0 against 4.0 GiB).
  **The gain is the layout's, not these kernels':** torch gains about the same from
  ``x @ wt``.
- **Stored,** ``HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0``: the parameters as
  ``nn.Linear`` holds them, ``[N, K]``, no copy.

**The copy assumes constant weights,** as the CPU models do. It follows what changes
a parameter's pointer, shape or ``_version``, and nothing else. Measured on torch
2.13+xpu, these optimizer steps change weights without bumping ``_version``, so the
copy **silently keeps the old weights**: ``Adam``, ``AdamW``, ``NAdam`` and ``Adamax``
with defaults, ``Adam`` with ``foreach=True``, and ``fused=True`` of ``Adam``, ``AdamW``
and ``SGD`` (fused on CPU too). ``SGD`` with ``foreach=True``, the ``foreach=False``
forms of ``Adam`` and ``AdamW``, and ``SGD``, ``RMSprop``, ``Adagrad``, ``RAdam`` and
``Adadelta`` with defaults bump it. The tests pin this list; the torch-only reproducer
is in the wiki's ``upstream-repro/xpu-foreach-version``. In-place writes through
``.data`` aren't seen either. Train with the stored form.

**Depth, 3.** A launch per layer costs about 25 us of host time on the B50, 17 times
per call; :func:`_deep_mlp` runs the 17 layers from one entry. Each intermediate
of an entry is a new buffer and they stay cold, so the hidden layers take turns
writing two buffers: wall on the B50 (one process, before the final model), 1688 us
per layer, 1638 in one entry with 16 buffers, 1429 with 2; the final model, 1469-1479,
1500 through AI-bench's CLI. Weights stay as stored: a transposed copy moved wall by
-2 % on the B50 and +4.2 % on the B70 (cause not measured), so the form above doesn't
apply to it and nothing can go stale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import helion
import helion.language as hl
import helion_mlir_backend  # noqa: F401
import torch
from torch import Tensor
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

from .device import MANY_XE_CORES
from .device import xe_cores
from .linear import cache_transposed_weights
from .linear import linear_plain
from .linear import linear_relu

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

SKINNY_BLOCKS = (128, 128, 32)
SKINNY_SG_TILE = (64, 32)
SKINNY_DEPTHS = (1, 0)
"""1 and 2 on transposed weights, ``M = 128``, one config on both cards. Device
reads are bound by the weights, so the choices follow what reading them costs
(sustained wall, us, one process each, 100 reps; L3/1, L3/2):

========================= ============== ==============
tiles, depths, subgroup   B50            B70
========================= ============== ==============
128x128, (1, 0), 64x32    6513, 21444    2542, 7760
128x512, (1, 0), 64x32    6696, 21680    2527, 7797
128x512, (1, 0), inferred 6729, 21955    2593, 7857
128x512, (0, 0), inferred 7179, 22043    2873, 8007
128x512, inferred         6887, 38143    2965, 15952
========================= ============== ==============

- No prefetch of the weight, ``(1, 0)``: inferred depths cost 1.7x on the B50 and 2x
  on the B70 at the widest layers. A prefetch of A alone gains 0.4 to 10 % over
  none: A is small and read every K step.
- 128x128 over 128x512: the B50's lead decides, 3 % on L3/1; on the B70 they are
  within 0.5 %.
- Subgroups of 64x32, 8 per workgroup: the accumulator takes the same 128 registers
  as 32x64; the gain over the inferred tile is measured, its cause isn't.
- 128 rows in one workgroup against two of 64: mixed at 512 columns (64x512 is 2 %
  ahead on L3/1 and 13 % behind on L3/2, B50, no prefetch), so the wider layers
  decide. K tiles of 64 are 2 to 8 % slower.
"""
DEEP_LAYERS = 17
"""3: 8192 -> 1024, 15 x 1024 -> 1024, 1024 -> 8192."""
DEEP_BLOCKS = (256, 256, 32)
"""3 on the B50: 16 workgroups fill its 16 Xe cores in the 1024x1024 layers, and
their 2 MiB weights make the prefetches pay (no prefetch: +2.3 % wall in one
entry). Sustained wall per layer against 256x256: hidden layers 128x256 +4 %, 256x128
+6 %, 128x128 +11 %; first layer 128x256 +7 %; last layer 128x512 +0.5 %. In one entry,
K tiles of 64 on the hidden layers change wall under 1 %."""
DEEP_MANY_CORES_BLOCKS = (128, 256, 32)
"""3 on the B70: 256x256 leaves half of its 32 Xe cores idle in the 1024x1024 layers
(16 workgroups); 128x256 fills them. Sustained wall, us: 932 against 758."""


def _skinny(rows: int, columns: int) -> bool:
    return rows == SKINNY_BLOCKS[0] and columns % SKINNY_BLOCKS[1] == 0


def _cached_kwargs(rows: int, columns: int) -> dict:
    """Keys of ``helion.Config`` of a layer on a transposed weight. Shapes outside the
    spec's M=128 take :data:`DEEP_BLOCKS` with inferred layouts: not tuned for this
    form, only known to lower and be right at M=256 and 1024; tiles that don't divide
    the shape are refused by the backend, as masking isn't used here."""
    if _skinny(rows, columns):
        return {
            "block_sizes": list(SKINNY_BLOCKS),
            "xegpu_sg_tiles": [list(SKINNY_SG_TILE)],
            "xegpu_prefetch_depths": [list(SKINNY_DEPTHS)],
        }
    return {"block_sizes": list(DEEP_BLOCKS)}


def _deep_block_sizes(cores: int) -> list[int]:
    """Three block sizes per layer of :func:`_deep_mlp`."""
    blocks = DEEP_MANY_CORES_BLOCKS if cores >= MANY_XE_CORES else DEEP_BLOCKS
    return list(blocks) * DEEP_LAYERS


def linear_layers(network: nn.Sequential) -> list[nn.Linear]:
    """The ``nn.Linear`` layers of a level 3 model, read from its modules on every call
    so that replacing one takes effect. The kernels build in the ReLUs, so anything but
    ``Linear, ReLU, ..., Linear`` raises rather than being ignored."""
    layers = []
    for index, module in enumerate(network):
        if not isinstance(module, nn.ReLU if index % 2 else nn.Linear):
            raise TypeError(
                f"module {index} of a level 3 XPU model is a {type(module).__name__}: "
                "the kernels fuse Linear, ReLU, ..., Linear"
            )
        if not index % 2:
            layers.append(module)
    if not len(network) % 2:
        raise TypeError(
            "the kernels fuse Linear, ReLU, ..., Linear: a model ends in a Linear"
        )
    return layers


def _check_shapes(x: Tensor, layers: Sequence[nn.Linear]) -> None:
    """The kernels loop over ``x``'s K and would read part of a wider weight, or past
    a narrower one, so a mismatch raises, as ``nn.Linear`` does."""
    k = x.size(1)
    for layer in layers:
        n, weight_k = layer.weight.size()
        if weight_k != k:
            raise RuntimeError(
                f"mat1 and mat2 shapes cannot be multiplied ({x.size(0)}x{k} and "
                f"{n}x{weight_k})"
            )
        k = n


class TransposedWeight(NamedTuple):
    key: tuple
    source: Tensor
    """The parameter's storage, held so that its pointer can't come back for another
    tensor while this entry lives."""
    weight: Tensor


def _pointer(parameter: Tensor) -> int:
    return id(parameter) if isinstance(parameter, FakeTensor) else parameter.data_ptr()


def _weight_key(parameter: Tensor, dtype: torch.dtype) -> tuple:
    """A parameter's identity for :func:`transposed_weight`: data pointer (object
    identity for a fake tensor, which has none), ``_version`` and shape."""
    return (
        _pointer(parameter),
        parameter._version,
        tuple(parameter.shape),
        dtype,
        parameter.device,
    )


def transposed_weight(
    parameter: Tensor, dtype: torch.dtype, cached: TransposedWeight | None
) -> TransposedWeight:
    """``parameter.T`` in ``dtype``, reusing ``cached`` while it is current. A new
    parameter or storage, in-place ops that bump ``_version`` and ``load_state_dict``
    are seen, and a parametrized weight is a new tensor on each call. In-place writes
    through ``.data`` (``p.data.mul_(2)``) and the optimizer steps in this module's
    docstring leave ``_version`` as it was and are **not** seen: the copy silently
    goes stale, as in the CPU models' ``LinearCache``."""
    key = _weight_key(parameter, dtype)
    if cached is not None and cached.key == key:
        return cached
    source = parameter.detach()
    weight = source.to(dtype).T
    if isinstance(weight, FakeTensor):  # IR-only tools: copying one needs a device
        return TransposedWeight(key, source, weight.new_empty(weight.shape))
    return TransposedWeight(key, source, weight.contiguous())


def _linear_relu_t(x: Tensor, wt: Tensor, bias: Tensor) -> Tensor:
    """Hidden layer on a transposed weight: ``relu(x @ wt + bias)``."""
    m, k = x.size()
    _, n = wt.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], wt[tk, tn])
        out[tm, tn] = torch.relu(acc + bias[tn]).to(x.dtype)
    return out


def _linear_plain_t(x: Tensor, wt: Tensor, bias: Tensor) -> Tensor:
    """Last layer on a transposed weight: ``x @ wt + bias``."""
    m, k = x.size()
    _, n = wt.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = torch.addmm(acc, x[tm, tk], wt[tk, tn])
        out[tm, tn] = (acc + bias[tn]).to(x.dtype)
    return out


def _deep_mlp(
    x: Tensor,
    w0: Tensor,
    b0: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
    w4: Tensor,
    b4: Tensor,
    w5: Tensor,
    b5: Tensor,
    w6: Tensor,
    b6: Tensor,
    w7: Tensor,
    b7: Tensor,
    w8: Tensor,
    b8: Tensor,
    w9: Tensor,
    b9: Tensor,
    w10: Tensor,
    b10: Tensor,
    w11: Tensor,
    b11: Tensor,
    w12: Tensor,
    b12: Tensor,
    w13: Tensor,
    b13: Tensor,
    w14: Tensor,
    b14: Tensor,
    w15: Tensor,
    b15: Tensor,
    w16: Tensor,
    b16: Tensor,
) -> Tensor:
    """Level 3 3: 16 times ``relu(linear(x))``, then ``linear(x)``, the hidden
    layers writing ``y0`` and ``y1`` in turn; see :data:`DEEP_BLOCKS`."""
    m, _ = x.size()
    n0, k0 = w0.size()
    n1, k1 = w1.size()
    n2, k2 = w2.size()
    n3, k3 = w3.size()
    n4, k4 = w4.size()
    n5, k5 = w5.size()
    n6, k6 = w6.size()
    n7, k7 = w7.size()
    n8, k8 = w8.size()
    n9, k9 = w9.size()
    n10, k10 = w10.size()
    n11, k11 = w11.size()
    n12, k12 = w12.size()
    n13, k13 = w13.size()
    n14, k14 = w14.size()
    n15, k15 = w15.size()
    n16, k16 = w16.size()
    y0 = torch.empty([m, n0], dtype=x.dtype, device=x.device)
    y1 = torch.empty([m, n0], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n16], dtype=x.dtype, device=x.device)
    for tm, tn in hl.tile([m, n0]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k0):
            acc = torch.addmm(acc, x[tm, tk], w0[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b0[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n1]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k1):
            acc = torch.addmm(acc, y0[tm, tk], w1[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b1[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n2]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k2):
            acc = torch.addmm(acc, y1[tm, tk], w2[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b2[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n3]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k3):
            acc = torch.addmm(acc, y0[tm, tk], w3[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b3[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n4]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k4):
            acc = torch.addmm(acc, y1[tm, tk], w4[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b4[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n5]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k5):
            acc = torch.addmm(acc, y0[tm, tk], w5[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b5[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n6]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k6):
            acc = torch.addmm(acc, y1[tm, tk], w6[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b6[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n7]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k7):
            acc = torch.addmm(acc, y0[tm, tk], w7[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b7[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n8]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k8):
            acc = torch.addmm(acc, y1[tm, tk], w8[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b8[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n9]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k9):
            acc = torch.addmm(acc, y0[tm, tk], w9[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b9[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n10]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k10):
            acc = torch.addmm(acc, y1[tm, tk], w10[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b10[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n11]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k11):
            acc = torch.addmm(acc, y0[tm, tk], w11[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b11[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n12]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k12):
            acc = torch.addmm(acc, y1[tm, tk], w12[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b12[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n13]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k13):
            acc = torch.addmm(acc, y0[tm, tk], w13[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b13[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n14]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k14):
            acc = torch.addmm(acc, y1[tm, tk], w14[tn, tk].T)
        y0[tm, tn] = torch.relu(acc + b14[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n15]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k15):
            acc = torch.addmm(acc, y0[tm, tk], w15[tn, tk].T)
        y1[tm, tn] = torch.relu(acc + b15[tn]).to(x.dtype)
    hl.barrier()
    for tm, tn in hl.tile([m, n16]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k16):
            acc = torch.addmm(acc, y1[tm, tk], w16[tn, tk].T)
        out[tm, tn] = (acc + b16[tn]).to(x.dtype)
    return out


_KERNELS: dict[tuple, helion.Kernel] = {}


def _kernel(fn: Callable[..., Tensor], key: tuple, config: dict) -> helion.Kernel:
    key = (fn, *key)
    if key not in _KERNELS:
        _KERNELS[key] = helion.kernel(
            fn, backend="mlir", static_shapes=True, config=helion.Config(**config)
        )
    return _KERNELS[key]


class DeepCall(NamedTuple):
    """:func:`_deep_mlp`'s kernel and arguments after ``x``: preparing 35 tensors
    per call cost 43 us wall on the B50 and 91 us on the B70."""

    key: tuple
    kernel: helion.Kernel
    args: list[Tensor]
    reusable: bool
    """Whether ``args`` are views of the parameters. Then every write reaches the kernel
    through their storage. Parameters in another dtype than ``x`` are converted again
    on every call, as a conversion can't see the writes ``_version`` misses."""


def _is_deep(rows: int, layers: Sequence[nn.Linear]) -> bool:
    """The one-entry kernel's shape: 17 layers with biases, 16 hidden layers of one
    size."""
    return (
        len(layers) == DEEP_LAYERS
        and all(layer.bias is not None for layer in layers)
        and rows % DEEP_BLOCKS[0] == 0
        and len({layer.out_features for layer in layers[:-1]}) == 1
    )


def _deep_key(x: Tensor, layers: Sequence[nn.Linear]) -> tuple:
    """Rows, dtype and device of ``x``, and pointer and shape of every parameter: the
    arguments hold the storage, so a pointer can't come back for another tensor."""
    return (
        x.size(0),
        x.dtype,
        x.device,
        tuple(
            (_pointer(parameter), parameter.shape)
            for layer in layers
            for parameter in (layer.weight, layer.bias)
        ),
    )


def _deep(x: Tensor, layers: Sequence[nn.Linear], caches: dict) -> Tensor:
    key = _deep_key(x, layers)
    call = caches.get("deep")
    if call is None or call.key != key or not call.reusable:
        _check_shapes(x, layers)
        cores = xe_cores(x.device)
        config = {"block_sizes": _deep_block_sizes(cores)}
        args = []
        for layer in layers:
            args += [layer.weight.detach().to(x.dtype), _bias(x, layer)]
        reusable = all(
            parameter.dtype == x.dtype
            for layer in layers
            for parameter in (layer.weight, layer.bias)
        )
        kernel = _kernel(_deep_mlp, (x.size(0), cores), config)
        call = DeepCall(key, kernel, args, reusable)
        caches["deep"] = call
    elif x.size(1) != call.args[0].size(1):
        _check_shapes(x, layers)
    return call.kernel(x, *call.args)


def _bias(x: Tensor, layer: nn.Linear) -> Tensor:
    if layer.bias is None:
        return torch.zeros(layer.out_features, dtype=x.dtype, device=x.device)
    return layer.bias.detach().to(x.dtype)


def mlp(x: Tensor, network: nn.Sequential, caches: dict) -> Tensor:
    """``relu(linear(x))`` over the ``Linear, ReLU, ..., Linear`` of ``network``, the
    last without the ReLU. ``caches`` (empty at the start) holds the transposed weights
    or the prepared call: the model owns it, so they live as long as the model does."""
    layers = linear_layers(network)
    if _is_deep(x.size(0), layers):
        return _deep(x, layers, caches)
    _check_shapes(x, layers)
    if not cache_transposed_weights():
        for layer in layers[:-1]:
            x = linear_relu(x, layer.weight, layer.bias)
        return linear_plain(x, layers[-1].weight, layers[-1].bias)
    rows = x.size(0)
    last = len(layers) - 1
    for index, layer in enumerate(layers):
        cached = transposed_weight(layer.weight, x.dtype, caches.get(index))
        caches[index] = cached
        columns = cached.weight.size(1)
        fn = _linear_plain_t if index == last else _linear_relu_t
        kernel = _kernel(fn, (rows, columns), _cached_kwargs(rows, columns))
        x = kernel(x, cached.weight, _bias(x, layer))
    return x
