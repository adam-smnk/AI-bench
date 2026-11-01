"""Seed-only fidelity of the helion-mlir CPU KernelBench models.

The harness copies the reference weights into a backend model before it
checks correctness, but tools that compare backends in separate processes
(e.g. eager vs helion-mlir timing runs) rely on ``set_all_seeds`` alone. A
backend model must therefore draw its parameters exactly like the
KernelBench reference: same distributions, in the same order, and it must
leave the RNG where the reference does (inputs are drawn after the model).
The XPU models are checked too, built on the CPU.

``load_model`` imports every kernel as ``sys.modules["kernel_model"]``, so
build a model right after loading it: loading a second model replaces the
module the first one's globals refer to.
"""

import pytest
import torch

from ai_bench.harness import core as ai_hc
from ai_bench.harness.runner import KernelBenchRunner
from ai_bench.harness.runner.benchmark_compare import copy_model_weights
from ai_bench.harness.runner.benchmark_compare import set_all_seeds

pytest.importorskip("helion_mlir_backend")

CPU = torch.device("cpu")


def _runners():
    ref = KernelBenchRunner(device=CPU, backend=ai_hc.Backend.PYTORCH)
    mlir = KernelBenchRunner(device=CPU, backend=ai_hc.Backend.HELION_MLIR)
    return ref, mlir


def _cases():
    _, mlir = _runners()
    trees = {
        "cpu": mlir.kernels,
        "xpu": mlir.kernels.parent.parent / "xpu" / "KernelBench",
    }
    for target, tree in trees.items():
        for level in ("level1", "level2", "level3"):
            for kernel in sorted((tree / level).glob("*.py")):
                if (mlir.specs / level / f"{kernel.stem}.yaml").exists():
                    yield pytest.param(
                        target,
                        f"{level}/{kernel.stem}",
                        id=f"{target}-{level}/{kernel.stem}",
                    )


def _build(key, spec_type, target="cpu"):
    """Seed-only reference and helion-mlir models plus inputs for ``key``.

    Also returns the RNG states right after building each model."""
    ref_runner, mlir_runner = _runners()
    level, name = key.split("/")
    spec = ref_runner.load_spec(ref_runner.specs / level / f"{name}.yaml")
    if spec_type not in spec:
        pytest.skip(f"no {spec_type} variant")
    variant = ai_hc.expand_variants(spec[spec_type])[0]
    inits = ref_runner.get_spec_inits(spec)
    set_all_seeds(123)
    ref = ref_runner.init_model(
        ref_runner.load_model(ref_runner.kernels / level / f"{name}.py"),
        variant,
        inits,
    )
    ref_rng = torch.get_rng_state()
    args = ai_hc.get_inputs(variant, ref_runner.get_spec_inputs(spec), device=CPU)
    tree = mlir_runner.kernels
    if target == "xpu":
        tree = tree.parent.parent / "xpu" / "KernelBench"
    set_all_seeds(123)
    try:
        model = mlir_runner.init_model(
            mlir_runner.load_model(tree / level / f"{name}.py"), variant, inits
        )
    except Exception as error:  # noqa: BLE001
        if target != "xpu":
            raise
        pytest.skip(f"XPU model not buildable on CPU: {type(error).__name__}")
    model_rng = torch.get_rng_state()
    return ref, model, args, ref_rng, model_rng


@pytest.mark.parametrize("target,key", list(_cases()))
def test_seed_only_parameters_match_reference(target, key):
    """Every parameter, in order, equals the reference's under one seed, and
    building the model leaves the RNG in the reference's state."""
    ref, model, _, ref_rng, model_rng = _build(key, "ci", target)
    assert torch.equal(model_rng, ref_rng)
    ref_params = [p for _, p in ref.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    assert len(params) == len(ref_params)
    for index, (p, r) in enumerate(zip(params, ref_params)):
        assert p.shape == r.shape, index
        assert torch.equal(p, r), index


@pytest.mark.parametrize("target,key", list(_cases()))
def test_harness_weight_copy_succeeds(target, key):
    """``copy_model_weights`` takes the reference state; derived constants
    kept as buffers must be non-persistent (97's ``_bn_scale`` made the copy
    fail, so the harness fell back to seed-only weights)."""
    ref, model, *_ = _build(key, "ci", target)
    assert copy_model_weights(ref, model)


@pytest.mark.parametrize(
    "key", ["level2/33_Gemm_Scale_BatchNorm", "level2/39_Gemm_Scale_BatchNorm"]
)
def test_gemm_scale_batchnorm_seed_only_output(key):
    """The bf16 mlir-cpu-bench case (128x4096x4096) without a weight copy:
    the models drew ``scale`` from torch.rand where the reference uses
    torch.randn, so the outputs differed by 1.3 x max|ref|."""
    ref, model, args, *_ = _build(key, "mlir-cpu-bench")
    with torch.no_grad():
        expected = ref(*args).float()
        out = model(*args).float()
    error = ((out - expected).abs().max() / expected.abs().max()).item()
    assert error < 2e-2


def _pair(key, inits, dtype):
    """Reference and helion-mlir CPU models of ``key`` built from ``inits``, in
    eval mode and ``dtype``, with the reference weights copied in. Built right
    after loading each (see the module docstring)."""
    ref_runner, mlir_runner = _runners()
    level, name = key.split("/")
    set_all_seeds(123)
    ref = ref_runner.load_model(ref_runner.kernels / level / f"{name}.py")(*inits)
    ref = ref.to(dtype).eval()
    model = mlir_runner.load_model(mlir_runner.kernels / level / f"{name}.py")(*inits)
    model = model.to(dtype).eval()
    assert copy_model_weights(ref, model)
    return ref, model


def _error(out, ref):
    out, ref = out.double(), ref.double()
    return ((out - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize(
    "key",
    [
        "level2/66_Matmul_Dropout_Softmax",
        "level2/84_Gemm_BatchNorm_Scaling_Softmax",
        "level2/99_Matmul_GELU_Softmax",
    ],
)
def test_softmax_over_partial_last_chunk(key):
    """130x1001x515 f32, where 66 first failed: the softmax over 515 columns
    runs in 128-column chunks, and the zero padding of the partial last one
    entered the row maxima and sums (0.18 x max|ref| off)."""
    inits = (1001, 515, 0.2) if key.startswith("level2/66_") else (1001, 515)
    ref, model = _pair(key, inits, torch.float32)
    torch.manual_seed(0)
    x = torch.rand(130, 1001)
    with torch.no_grad():
        assert _error(model(x), ref(x)) < 1e-4


@pytest.mark.parametrize(
    "key",
    [
        "level2/66_Matmul_Dropout_Softmax",
        "level2/84_Gemm_BatchNorm_Scaling_Softmax",
        "level2/99_Matmul_GELU_Softmax",
    ],
)
def test_softmax_over_one_column(key):
    """OUT_FEATURES = 1: the chunked softmax kernels did not compile
    (ControlFlowTensorMismatch); a softmax over one column is 1."""
    inits = (32, 1, 0.2) if key.startswith("level2/66_") else (32, 1)
    ref, model = _pair(key, inits, torch.float32)
    x = torch.rand(5, 32)
    with torch.no_grad():
        assert torch.equal(model(x), ref(x))


def test_matmul_dropout_softmax_needs_eval_mode():
    """AI-bench runs models in eval mode, where dropout is the identity;
    random training masks cannot match the reference, so they are refused."""
    _, model = _pair("level2/66_Matmul_Dropout_Softmax", (32, 32, 0.2), torch.float32)
    with pytest.raises(ValueError, match="eval mode"):
        model.train()(torch.rand(2, 32))


@pytest.mark.parametrize(
    ("m", "k", "n", "pool", "scale", "dtype"),
    [
        (130, 1001, 515, 16, 2.0, torch.float32),  # partial last window dropped
        (130, 1001, 515, 16, 2.0, torch.bfloat16),  # odd K: blocked bf16 GEMM
        (100, 512, 301, 3, -1.5, torch.float32),  # window of 3, negative scale
        (64, 512, 512, 16, 2.0, torch.bfloat16),  # AMX VNNI path
    ],
)
def test_matmul_avgpool_pooled_weights(m, k, n, pool, scale, dtype):
    """98 folds the average pool into the weights (window means of W's rows
    and of b, W' rounded once to the input dtype): checked against the
    reference and against its fp64 result."""
    ref, model = _pair(
        "level2/98_Matmul_AvgPool_GELU_Scale_Max", (k, n, pool, scale), dtype
    )
    torch.manual_seed(0)
    x = torch.rand(m, k).to(dtype)
    with torch.no_grad():
        out = model(x)
        expected = ref(x)
        exact = ref.double()(x.double())
    if dtype is torch.float32:
        assert _error(out, expected) < 1e-5
    else:
        # bf16: no further from the fp64 result than the reference is.
        assert _error(out, exact) <= max(_error(expected, exact), 4e-3)


def test_matmul_avgpool_pooled_weights_follow_updates():
    """The pooled weights are rebuilt when the parameters change, in place or
    through ``load_state_dict``."""
    key = "level2/98_Matmul_AvgPool_GELU_Scale_Max"
    ref, model = _pair(key, (64, 64, 8, 2.0), torch.float32)
    x = torch.rand(4, 64)
    with torch.no_grad():
        model(x)
        model.matmul.weight.mul_(-0.5)
        ref.matmul.weight.mul_(-0.5)
        assert _error(model(x), ref(x)) < 1e-5
        ref.matmul.bias.add_(0.25)
        model.load_state_dict(ref.state_dict())
        assert _error(model(x), ref(x)) < 1e-5
