"""Level 3 MLPs on Helion MLIR for XPU: the transposed-weight cache, the switch, shape
and module checks, the configs of the spec's shapes, and (on an XPU) the outputs
against fp64 torch, with weights scaled so that every layer and ``x`` reach them.

Shapes on the device are the smallest the tiles divide, not the spec's.
"""

import importlib
import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

pytest.importorskip("helion_mlir_backend")

from helion_mlir_xpu_utils.linear import CACHE_WEIGHTS_ENV
from helion_mlir_xpu_utils.linear import cache_transposed_weights

mlp_utils = importlib.import_module("helion_mlir_xpu_utils.mlp")
linear_utils = importlib.import_module("helion_mlir_xpu_utils.linear")

XPU = torch.xpu.is_available()
needs_xpu = pytest.mark.skipif(not XPU, reason="needs an XPU")
KERNELS = (
    Path(__file__).resolve().parents[1]
    / "backends"
    / "helion_mlir"
    / "xpu"
    / "KernelBench"
    / "level3"
)
BF16 = torch.bfloat16
TOL = 2e-2
SHALLOW = ("1_MLP", 128, [1024, 1024, 1024, 512])
DEEP = ("3_DeepNarrowMLP", 256, [512, *[256] * 16, 512])
GAIN = 2.5
"""The default init shrinks the signal per deep layer until ``x`` and the early layers
barely reach the output (2.6e-5): a kernel that drops them still matches."""

OPTIMIZERS_THAT_MISS_VERSION = [
    ("Adam", {}),
    ("Adam", {"foreach": True}),
    ("Adam", {"fused": True}),
    ("AdamW", {}),
    ("AdamW", {"fused": True}),
    ("NAdam", {}),
    ("Adamax", {}),
    ("SGD", {"fused": True}),
]
OPTIMIZERS_THAT_BUMP_VERSION = [
    ("Adam", {"foreach": False}),
    ("AdamW", {"foreach": False}),
    ("SGD", {}),
    ("SGD", {"foreach": True}),
    ("RMSprop", {}),
    ("Adagrad", {}),
    ("RAdam", {}),
    ("Adadelta", {}),
]


def _load_model(name: str) -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location(
        f"xpu_l3_{name}", KERNELS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def _make(case, gain=None, dtype=BF16) -> tuple[nn.Module, torch.Tensor]:
    name, rows, sizes = case
    torch.manual_seed(5)
    model = _load_model(name)(sizes[0], sizes[1:-1], sizes[-1])
    model = model.to(device="xpu", dtype=dtype)
    if gain is not None:
        with torch.no_grad():
            for layer in _layers(model):
                layer.weight.mul_(gain)
    return model, _input(rows, sizes[0])


def _input(rows: int, k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, k, dtype=torch.float32).to(BF16).to("xpu")


def _layers(model: nn.Module) -> list[nn.Linear]:
    return mlp_utils.linear_layers(model.network)


def _reference(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    y = x.cpu().double()
    layers = _layers(model)
    for index, layer in enumerate(layers):
        y = y @ layer.weight.detach().cpu().double().T + layer.bias.cpu().double()
        if index < len(layers) - 1:
            y = torch.relu(y)
    return y


def _error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out = out.cpu().double()
    assert out.shape == ref.shape
    assert out.isnan().sum() == 0
    return ((out - ref).abs().max() / ref.abs().max()).item()


def _check(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The output, after checking it against fp64."""
    with torch.no_grad():
        out = model(x)
    assert _error(out, _reference(x, model)) < TOL
    return out.cpu()


def _new_values(parameter: torch.Tensor, scale: float = GAIN) -> torch.Tensor:
    values = torch.randn(parameter.shape, dtype=torch.float32) * scale / 15.0
    return values.to(parameter.dtype).to(parameter.device)


@pytest.fixture(params=["cached", "stored"])
def form(request, monkeypatch) -> str:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, "1" if request.param == "cached" else "0")
    return request.param


# --- The cache ---------------------------------------------------------------


def test_transposed_weight_is_reused_while_unchanged() -> None:
    weight = nn.Parameter(torch.arange(6.0).reshape(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    assert torch.equal(first.weight, weight.detach().T)
    assert mlp_utils.transposed_weight(weight, torch.float32, first) is first


def test_transposed_weight_refreshes_on_in_place_op() -> None:
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    with torch.no_grad():
        weight.mul_(2.0)
    second = mlp_utils.transposed_weight(weight, torch.float32, first)
    assert second is not first
    assert torch.equal(second.weight, weight.detach().T)


def test_transposed_weight_refreshes_on_load_state_dict() -> None:
    layer = nn.Linear(3, 2)
    first = mlp_utils.transposed_weight(layer.weight, torch.float32, None)
    layer.load_state_dict({"weight": torch.full((2, 3), 5.0), "bias": torch.zeros(2)})
    second = mlp_utils.transposed_weight(layer.weight, torch.float32, first)
    assert torch.equal(second.weight, torch.full((3, 2), 5.0))


def test_transposed_weight_refreshes_on_rebind() -> None:
    layer = nn.Linear(3, 2)
    first = mlp_utils.transposed_weight(layer.weight, torch.float32, None)
    layer.weight = nn.Parameter(torch.full((2, 3), 7.0))
    second = mlp_utils.transposed_weight(layer.weight, torch.float32, first)
    assert torch.equal(second.weight, torch.full((3, 2), 7.0))


def test_transposed_weight_refreshes_on_new_storage_alone() -> None:
    """``.data = new`` leaves ``_version``: only the pointer shows it."""
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    version = weight._version
    weight.data = torch.full((2, 3), 3.0)
    assert weight._version == version
    second = mlp_utils.transposed_weight(weight, torch.float32, first)
    assert torch.equal(second.weight, torch.full((3, 2), 3.0))


def test_transposed_weight_holds_the_storage_of_its_key() -> None:
    """Without it a freed block's pointer and version come back for new weights."""
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    assert first.source.data_ptr() == first.key[0]
    assert first.source.data_ptr() == weight.data_ptr()
    weight.data = torch.full((2, 3), 3.0)
    assert first.source.data_ptr() != weight.data_ptr()


def test_transposed_weight_stays_stale_after_data_write() -> None:
    """Documented limit, as the CPU models' cache: ``.data`` doesn't bump
    ``_version``. If this starts to fail the docstring is out of date."""
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    weight.data.mul_(2.0)
    assert mlp_utils.transposed_weight(weight, torch.float32, first) is first
    assert torch.equal(first.weight, torch.ones(3, 2))


def test_transposed_weight_is_a_copy() -> None:
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    assert first.weight.data_ptr() != weight.data_ptr()


def test_transposed_weight_follows_dtype() -> None:
    weight = nn.Parameter(torch.ones(2, 3))
    first = mlp_utils.transposed_weight(weight, torch.float32, None)
    second = mlp_utils.transposed_weight(weight, BF16, first)
    assert second.weight.dtype == BF16


class _Doubled(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * 2.0


def test_transposed_weight_of_a_parametrized_weight_follows_writes() -> None:
    layer = nn.Linear(3, 2)
    parametrize.register_parametrization(layer, "weight", _Doubled())
    first = mlp_utils.transposed_weight(layer.weight, torch.float32, None)
    layer.parametrizations.weight.original.data.mul_(3.0)
    second = mlp_utils.transposed_weight(layer.weight, torch.float32, first)
    assert torch.equal(second.weight, layer.weight.detach().T)


# --- The switch --------------------------------------------------------------


def test_cached_is_the_default(monkeypatch) -> None:
    """The user's decision (2026-11-04), consistent with the CPU models."""
    monkeypatch.delenv(CACHE_WEIGHTS_ENV, raising=False)
    assert cache_transposed_weights() is True


@pytest.mark.parametrize("default", [False, True])
def test_switch_default_is_one_constant(default, monkeypatch) -> None:
    monkeypatch.delenv(CACHE_WEIGHTS_ENV, raising=False)
    monkeypatch.setattr(linear_utils, "CACHE_WEIGHTS_DEFAULT", default)
    assert cache_transposed_weights() is default


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize(("value", "cached"), [("0", False), ("1", True)])
def test_switch_overrides_either_default(default, value, cached, monkeypatch) -> None:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, value)
    monkeypatch.setattr(linear_utils, "CACHE_WEIGHTS_DEFAULT", default)
    assert cache_transposed_weights() is cached


def test_stored_form_makes_no_copy() -> None:
    with (
        mock.patch.dict("os.environ", {CACHE_WEIGHTS_ENV: "0"}),
        mock.patch.object(mlp_utils, "transposed_weight") as transposed,
        mock.patch.object(mlp_utils, "linear_relu", lambda x, w, b: x),
        mock.patch.object(mlp_utils, "linear_plain", lambda x, w, b: x),
    ):
        network = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        mlp_utils.mlp(torch.ones(1, 4), network, {})
    transposed.assert_not_called()


# --- Modules and shapes, no device ---------------------------------------------


def test_linear_layers_reads_the_live_modules() -> None:
    network = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    new = nn.Linear(4, 4)
    network[2] = new
    assert mlp_utils.linear_layers(network)[1] is new


@pytest.mark.parametrize(
    "network",
    [
        nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4)),
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        nn.Sequential(nn.ReLU(), nn.Linear(4, 4)),
    ],
    ids=["other_activation", "ends_in_relu", "starts_with_relu"],
)
def test_other_module_structures_raise(network) -> None:
    with pytest.raises(TypeError, match="Linear, ReLU"):
        mlp_utils.linear_layers(network)


@pytest.mark.parametrize("case", [SHALLOW, DEEP], ids=["shallow", "deep"])
@pytest.mark.parametrize("k", [256, 2048], ids=["narrow", "wide"])
def test_wrong_input_width_raises_as_torch(case, k) -> None:
    name, rows, sizes = case
    model = _load_model(name)(sizes[0], sizes[1:-1], sizes[-1])
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        model(torch.zeros(rows, k))


def test_wrong_inner_width_raises() -> None:
    network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        mlp_utils.mlp(torch.zeros(1, 4), network, {})


# --- Configs of the spec's shapes --------------------------------------------


@pytest.mark.parametrize(
    "rows,sizes",
    [
        (128, [16384, 16384, 16384, 8192]),
        (128, [16384, 32768, 32768, 16384]),
    ],
)
def test_cached_tiles_divide_the_spec_shapes(rows, sizes) -> None:
    for k, n in zip(sizes, sizes[1:]):
        config = mlp_utils._cached_kwargs(rows, n)
        tile_m, tile_n, tile_k = config["block_sizes"]
        assert rows % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0


@pytest.mark.parametrize("cores", [16, 32])
def test_deep_block_sizes_have_three_per_layer(cores) -> None:
    sizes = mlp_utils._deep_block_sizes(cores)
    assert len(sizes) == 3 * mlp_utils.DEEP_LAYERS
    for index, (k, n) in enumerate(zip([8192] + [1024] * 16, [1024] * 16 + [8192])):
        tile_m, tile_n, tile_k = sizes[3 * index : 3 * index + 3]
        assert 1024 % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0


# --- Optimizers and _version: torch's behaviour this module's docs rest on -----


@pytest.mark.parametrize(("name", "kwargs"), [("Adam", {"fused": True})])
def test_fused_optimizer_misses_version_on_cpu(name, kwargs) -> None:
    _step_and_check_version(name, kwargs, "cpu", bumps=False)


@needs_xpu
@pytest.mark.parametrize(("name", "kwargs"), OPTIMIZERS_THAT_MISS_VERSION)
def test_optimizer_misses_version_on_xpu(name, kwargs) -> None:
    _step_and_check_version(name, kwargs, "xpu", bumps=False)


@needs_xpu
@pytest.mark.parametrize(("name", "kwargs"), OPTIMIZERS_THAT_BUMP_VERSION)
def test_optimizer_bumps_version_on_xpu(name, kwargs) -> None:
    _step_and_check_version(name, kwargs, "xpu", bumps=True)


def _step_and_check_version(name, kwargs, device, bumps) -> None:
    parameter = nn.Parameter(torch.ones(32, 32, dtype=torch.float32, device=device))
    parameter.grad = torch.ones(32, 32, dtype=torch.float32, device=device)
    before = parameter.detach().clone()
    version = parameter._version
    getattr(torch.optim, name)([parameter], lr=0.1, **kwargs).step()
    assert not torch.equal(before.cpu(), parameter.detach().cpu())
    assert (parameter._version != version) is bumps


def _adam_step(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    torch.optim.Adam(model.parameters(), lr=0.01).step()


@needs_xpu
def test_adam_leaves_the_default_form_stale(monkeypatch) -> None:
    """Pinned because it's silently wrong: the reason to train with the stored form.
    If it starts to fail, the docstring changes."""
    monkeypatch.delenv(CACHE_WEIGHTS_ENV, raising=False)
    model, x = _make(SHALLOW)
    before = _check(model, x)
    _adam_step(model)
    with torch.no_grad():
        out = model(x)
    ref = _reference(x, model)
    # Bias reaches the kernel live: only the weights' part stays behind.
    assert _error(out, ref) > TOL
    assert not torch.equal(before, ref.to(BF16))


@needs_xpu
def test_adam_reaches_the_stored_form(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, "0")
    model, x = _make(SHALLOW)
    before = _check(model, x)
    _adam_step(model)
    assert not torch.equal(_check(model, x), before)


@needs_xpu
def test_adam_reaches_the_deep_model() -> None:
    model, x = _make(DEEP, gain=GAIN)
    before = _check(model, x)
    _adam_step(model)
    assert not torch.equal(_check(model, x), before)


# --- Outputs on the device ---------------------------------------------------


@needs_xpu
@pytest.mark.parametrize("name", ["1_MLP", "2_ShallowWideMLP"])
def test_shallow_models_match_fp64(name, form) -> None:
    model, _ = _make((name, *SHALLOW[1:]), gain=GAIN)
    outputs = [_check(model, _input(128, 1024, seed)) for seed in range(2)]
    assert not torch.equal(outputs[0], outputs[1])  # x reaches the output


@needs_xpu
def test_cached_form_off_the_spec_rows_matches_fp64(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, "1")
    model, _ = _make(("1_MLP", 256, [512, 512, 512, 256]), gain=GAIN)
    _check(model, _input(256, 512))


@needs_xpu
def test_deep_model_matches_fp64() -> None:
    """17 layers in one entry, two hidden buffers in turn."""
    model, _ = _make(DEEP, gain=GAIN)
    rows, k = DEEP[1], DEEP[2][0]
    refs = [_reference(_input(rows, k, seed), model) for seed in range(2)]
    moved = ((refs[0] - refs[1]).abs().max() / refs[0].abs().max()).item()
    assert moved > 0.05, "the signal must reach the output for this test to see layers"
    outputs = [_check(model, _input(rows, k, seed)) for seed in range(2)]
    assert not torch.equal(outputs[0], outputs[1])


@needs_xpu
def test_every_deep_layer_reaches_the_output() -> None:
    """Negating a layer's weights moves fp64's answer; the kernel must follow. Catches
    a kernel that drops a layer or reads an older buffer in the early layers."""
    model, x = _make(DEEP, gain=GAIN)
    base = _check(model, x)
    for layer in _layers(model):
        with torch.no_grad():
            layer.weight.mul_(-1.0)
        moved = _check(model, x)
        assert not torch.equal(moved, base)
        with torch.no_grad():
            layer.weight.mul_(-1.0)


@needs_xpu
@pytest.mark.parametrize("index", [0, 3, 9, 16], ids=lambda i: f"layer{i}")
def test_deep_model_follows_new_storage_alone(index) -> None:
    """``.data = new``: only the pointer changes (``d_no_pointer``, ``d_never_refresh``
    of the review pass without this)."""
    model, x = _make(DEEP, gain=GAIN)
    before = _check(model, x)
    layer = _layers(model)[index]
    version = layer.weight._version
    layer.weight.data = _new_values(layer.weight)
    assert layer.weight._version == version
    assert not torch.equal(_check(model, x), before)


@needs_xpu
@pytest.mark.parametrize("index", [0, 3, 9, 16], ids=lambda i: f"layer{i}")
def test_deep_model_follows_rebinds_twice(index) -> None:
    model, x = _make(DEEP, gain=GAIN)
    before = _check(model, x)
    layer = _layers(model)[index]
    for _ in range(2):
        weight = _new_values(layer.weight)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        after = _check(model, x)
        assert not torch.equal(after, before)
        before = after


@needs_xpu
def test_deep_model_checks_widths_after_a_good_call() -> None:
    """A prepared call must not skip the checks: a new input width, and a weight that
    changed shape at the same pointer (only the shape shows it)."""
    model, x = _make(DEEP, gain=GAIN)
    _check(model, x)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        model(_input(DEEP[1], 1024))
    _check(model, x)
    weight = _layers(model)[0].weight
    pointer = weight.data_ptr()
    weight.data = weight.data.reshape(weight.size(1), weight.size(0))
    assert weight.data_ptr() == pointer
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        model(x)


@needs_xpu
@pytest.mark.parametrize("index", [3, 9], ids=lambda i: f"layer{i}")
def test_deep_model_follows_writes_through_data(index) -> None:
    """Weights as stored: no copy to go stale."""
    model, x = _make(DEEP, gain=GAIN)
    before = _check(model, x)
    _layers(model)[index].weight.data.mul_(-1.0)
    assert not torch.equal(_check(model, x), before)


@needs_xpu
def test_deep_model_with_f32_parameters_follows_every_write(monkeypatch) -> None:
    """A conversion to ``x``'s dtype isn't a view: it's made again on every call."""
    model, x = _make(DEEP, gain=GAIN, dtype=torch.float32)
    layer = _layers(model)[9]
    before = _check(model, x)
    layer.weight.data.mul_(-1.0)
    after = _check(model, x)
    assert not torch.equal(after, before)
    _adam_step(model)
    assert not torch.equal(_check(model, x), after)
    for _ in range(2):
        layer.weight.data = _new_values(layer.weight)
        _check(model, x)


@needs_xpu
def test_shallow_cached_refreshes_on_each_update(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, "1")
    model, x = _make(SHALLOW, gain=GAIN)
    layer = _layers(model)[0]
    before = _check(model, x)
    with torch.no_grad():
        layer.weight.mul_(-1.0)
    after = _check(model, x)
    assert not torch.equal(after, before)
    # New storage alone, twice: the second would reuse a freed block's pointer.
    for _ in range(2):
        layer.weight.data = _new_values(layer.weight)
        moved = _check(model, x)
        assert not torch.equal(moved, after)
        after = moved
    for _ in range(2):
        layer.weight = nn.Parameter(_new_values(layer.weight), requires_grad=False)
        moved = _check(model, x)
        assert not torch.equal(moved, after)
        after = moved
    # A write through .data is not seen: the output follows the old copy.
    layer.weight.data.mul_(-1.0)
    with torch.no_grad():
        stale = model(x).cpu()
    assert torch.equal(stale, after)
    assert _error(stale, _reference(x, model)) > TOL


@needs_xpu
def test_parametrized_weight_follows_writes(monkeypatch) -> None:
    monkeypatch.setenv(CACHE_WEIGHTS_ENV, "1")
    model, x = _make(SHALLOW, gain=GAIN)
    layer = _layers(model)[0]
    parametrize.register_parametrization(layer, "weight", _Doubled())
    before = _check(model, x)
    layer.parametrizations.weight.original.data.mul_(-1.0)
    assert not torch.equal(_check(model, x), before)


@needs_xpu
@pytest.mark.parametrize("case", [SHALLOW, DEEP], ids=["shallow", "deep"])
def test_replaced_module_takes_effect(case, form) -> None:
    model, x = _make(case, gain=GAIN)
    before = _check(model, x)
    old = model.network[-1]
    new = nn.Linear(old.in_features, old.out_features).to(device="xpu", dtype=BF16)
    with torch.no_grad():
        new.weight.mul_(GAIN)
    model.network[-1] = new
    assert not torch.equal(_check(model, x), before)
