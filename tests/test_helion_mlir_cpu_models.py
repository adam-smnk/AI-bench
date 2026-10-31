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
