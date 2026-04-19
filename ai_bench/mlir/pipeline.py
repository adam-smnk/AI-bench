from lighthouse import dialects as lh_dialects
from lighthouse import schedule as lh_schedule
from lighthouse import transform as lh_transform
import lighthouse.schedule.x86 as lh_schedule_x86
import lighthouse.transform.x86 as lh_transform_x86
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tensor
from mlir.passmanager import PassManager


def cpu_pipeline(module: ir.Module) -> ir.Module:
    """
    The default lowering pipeline for CPU.
    Lowers MLIR ops within the module to MLIR LLVM IR dialect.

    The pipeline focuses on enabling end-to-end lowering for various
    generic kernel modules.

    Performance is currently secondary and not representative.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """

    # Use standard C interface wrappers for functions.
    pm = PassManager("builtin.module", module.context)
    pm.add("func.func(llvm-request-c-wrappers)")

    # Bufferize.
    pm.add("eliminate-empty-tensors")
    pm.add(
        "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}"
    )
    pm.add("drop-equivalent-buffer-results")
    pm.add("buffer-deallocation-pipeline")
    pm.add("convert-bufferization-to-memref")
    pm.add("cse")
    pm.add("canonicalize")

    # Lower to LLVM.
    pm.add("convert-linalg-to-loops")
    pm.add("math-expand-ops")
    pm.add("expand-strided-metadata")
    pm.add("canonicalize")

    pm.add("convert-vector-to-scf")
    pm.add("lower-affine")
    pm.add("convert-scf-to-cf")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-math-to-libm")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    return module


def cpu_vectorizer(module: ir.Module) -> ir.Module:
    """
    The vectorization pipeline for CPU.
    Lowers MLIR ops within the module to MLIR LLVM IR dialect.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """

    # Use standard C interface wrappers for functions.
    pm = PassManager("builtin.module", module.context)
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.run(module.operation)

    with module.context, ir.Location.unknown():
        lh_dialects.register_and_load(reload=True)

    tile_size = 32
    with module.context, ir.Location.unknown():
        sched = lh_schedule.block_pack_matmuls(
            block_factors=[tile_size, tile_size, tile_size],
            rhs_transpose_outer_block=True,
            rhs_transpose_inner_block=False,
        )
        sched.body.operations[0].apply(module)
        sched = lh_schedule_x86.lower_packs_unpacks(tile_size)
        sched.body.operations[0].apply(module)
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
            transform.apply_registered_pass(
                transform.any_op_t(),
                ops,
                "linalg-morph-ops",
                options={
                    "named-to-category": True,
                    "generic-to-category": True,
                },
            )
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        sched.body.operations[0].apply(module)

    # GEMM cache tiling.
    gemm_op = "linalg.contract"
    with module.context, ir.Location.unknown():
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            ops = lh_transform.match_op(named_seq.bodyTarget, gemm_op)
            with lh_transform.foreach(ops) as op:
                lh_transform_x86.matmul_cache_tiling(
                    op, num_tiles=6, tile_size=tile_size, fuse_producers=True
                )
                transform.yield_()
            transform.yield_()
        sched.body.operations[0].apply(module)
        sched = lh_schedule.linalg_contract_fold_unit_dims()
        sched.body.operations[0].apply(module)

        # GEMM register tiling.
        reg_tile_batch = 1
        reg_tile_m = 8
        reg_tile_n = 32
        reg_tile_k = 1
        reg_peel_loops = []
        assert tile_size % reg_tile_k == 0, "Invalid K dim register iling"
        if tile_size % reg_tile_n != 0:
            reg_peel_loops.append(1)
        if tile_size % reg_tile_m != 0:
            reg_peel_loops.append(0)
        sched = lh_schedule.tile_ops(
            gemm_op,
            tile_sizes=[reg_tile_batch, reg_tile_m, reg_tile_n, reg_tile_k],
            tile_interchange=[1, 2, 0, 3],
            peel_loops=reg_peel_loops,
        )
        sched.body.operations[0].apply(module)

        # GEMM register unroll.
        reg_unroll_m = 1
        reg_unroll_n = 16
        # When VNNI can be used, tuples of 32-bit elements are needed.
        reg_unroll_k = 1  # TODO: enable VNNI
        reg_unroll_factors = [
            reg_tile_m // reg_unroll_m,
            reg_tile_n // reg_unroll_n,
            reg_tile_k // reg_unroll_k,
        ]
        sched = lh_schedule.tile_ops(
            gemm_op,
            tile_sizes=[0, reg_unroll_m, reg_unroll_n, reg_unroll_k],
            unroll_factors=reg_unroll_factors,
        )
        sched.body.operations[0].apply(module)

        # sched = lh_schedule.tile_ops("linalg.fill", tile_sizes=[1, 1, 1])
        # sched.body.operations[0].apply(module)
        # sched = lh_schedule.tile_ops("linalg.generic", tile_sizes=[1, 8])
        # sched.body.operations[0].apply(module)

        # Vectorization.
        sched = lh_schedule.vectorize_linalg()
        sched.body.operations[0].apply(module)
        sched = lh_schedule.hoist_loops()
        sched.body.operations[0].apply(module)
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
                transform.apply_patterns_canonicalization()
            transform.yield_()
        sched.body.operations[0].apply(module)
        sched = lh_schedule.x86_vectorization()
        sched.body.operations[0].apply(module)

    # Bufferize.
    pm = PassManager("builtin.module", module.context)
    pm.add("eliminate-empty-tensors")
    pm.add(
        "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}"
    )
    pm.add("drop-equivalent-buffer-results")
    pm.add("buffer-deallocation-pipeline")
    pm.add("convert-bufferization-to-memref")
    pm.add("cse")
    pm.add("canonicalize")
    pm.run(module.operation)

    with module.context, ir.Location.unknown():
        sched = lh_schedule.x86_vectorization()
        sched.body.operations[0].apply(module)
        sched = lh_schedule.vectorize_all()
        sched.body.operations[0].apply(module)

        # Cleanup vector ops.
        with lh_schedule.schedule_boilerplate() as (sched, named_seq):
            lh_transform.flatten_vector_ops(named_seq.bodyTarget)
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()
        sched.body.operations[0].apply(module)

    # Lower to LLVM.
    pm = PassManager("builtin.module", module.context)
    pm.add("convert-linalg-to-loops")
    pm.add("math-expand-ops")
    pm.add("expand-strided-metadata")
    pm.add("canonicalize")

    pm.add("func.func(lower-vector-multi-reduction)")
    pm.add("convert-vector-to-scf")
    pm.add("lower-affine")
    pm.add("convert-scf-to-cf")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-math-to-libm")
    pm.add("convert-to-llvm")
    # pm.add("print-ir")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    return module
