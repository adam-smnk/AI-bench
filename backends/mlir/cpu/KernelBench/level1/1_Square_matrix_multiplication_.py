from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import gpu
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector
from mlir.dialects.transform import tensor
from mlir.dialects.transform import x86vector
from mlir.passmanager import PassManager
import torch
import torch.nn as nn

import ai_bench.mlir

TILE_SIZE = 64


def cleanup(target):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_cse(func)
    with ir.InsertionPoint(transform.ApplyPatternsOp(func).patterns):
        transform.apply_patterns_canonicalization()


def tile_and_vector_gemm(ctx: ir.Context) -> ir.Module:
    """
    Specialized schedule for Linalg operations.

    Tiling and vectorization is progressively applied to
    achieve SIMD code generation.

    Args:
        ctx: MLIR context.
    Returns:
        MLIR transform module.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create a transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.any_op_t()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            anytype = transform.any_op_t()

            # GEMM tiling.
            gemm_name = "linalg.generic"
            mm = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, [gemm_name]
            ).result
            structured.FuseOp(mm, tile_sizes=[1, 1], apply_cleanup=True).results[0]

            # Tile buffer initialization for better vectorization.
            tiled_fill = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.fill"]
            ).result
            reg_fill = structured.TileUsingForOp(tiled_fill, sizes=[1, 1]).results[0]

            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                # structured.apply_patterns_linalg_fold_unit_extent_dims_via_reshapes()
                structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
                structured.apply_patterns_linalg_fold_pack_unpack_into_empty()
            cleanup(named_seq.bodyTarget)

            # Register tiling.
            brgemm = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, [gemm_name]
            ).result
            reg_mm = structured.TileUsingForOp(brgemm, sizes=[1, 8, 32, 1]).results[0]

            # Vectorize operations.
            structured.structured_vectorize(reg_mm, [], create_named_contraction=True)
            structured.structured_vectorize(reg_fill, [])
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                vector.apply_patterns_vector_reduction_to_contract()
                vector.apply_patterns_vector_transfer_permutation_patterns()
            cleanup(named_seq.bodyTarget)

            # Loop hoisting.
            all_loops = structured.MatchOp(
                anytype,
                named_seq.bodyTarget,
                interface=structured.MatchInterfaceEnum.LoopLikeInterface,
            ).results
            transform.apply_licm(all_loops)
            loop.loop_hoist_loop_invariant_subsets(all_loops)

            # Unroll GEMM.
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                gpu.apply_patterns_gpu_unroll_vectors_subgroup_mma(m=1, n=32, k=1)
                vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
                transform.apply_patterns_canonicalization()

            # # Lower to broadcast+FMA instructions.
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                x86vector.apply_patterns_x86vector_vector_contract_to_fma()
                x86vector.apply_patterns_x86vector_sink_vector_producer_ops()
                vector.apply_patterns_vector_flatten_vector_transfer_ops()

            # Cleanup.
            transform.apply_cse(named_seq.bodyTarget)
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                transform.apply_patterns_canonicalization()

            transform.yield_()
    return schedule


def pack_gemm(ctx: ir.Context) -> ir.Module:
    with ctx, ir.Location.unknown(context=ctx):
        # Create a transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.any_op_t()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            anytype = transform.any_op_t()

            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            ).result
            transform.apply_registered_pass(
                anytype,
                func,
                "linalg-block-pack-matmul",
                options={
                    "block-factors": (TILE_SIZE, TILE_SIZE, TILE_SIZE),
                    "rhs-transpose-outer-blocks": True,
                    "rhs-transpose-inner-blocks": False,
                },
            )

            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                structured.apply_patterns_linalg_fold_pack_unpack_into_empty()
                structured.apply_patterns_tensor_fold_into_pack_and_unpack()
            cleanup(named_seq.bodyTarget)

            packs = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.pack"]
            )
            foreach_pack = transform.ForeachOp([], (packs,))
            with ir.InsertionPoint(foreach_pack.body):
                pack_op = foreach_pack.bodyTargets[0]
                tiled_pack = structured.FuseOp(
                    pack_op, tile_sizes=[1, 1], apply_cleanup=True
                ).results[0]
                _, _, transpose = structured.structured_lower_pack(
                    anytype,
                    anytype,
                    anytype,
                    tiled_pack,
                    lower_pad_like_with_insert_slice=False,
                )
                structured.TileUsingForOp(
                    transpose, sizes=[1, 1, 1]
                )
                transform.yield_()
            cleanup(named_seq.bodyTarget)
            # transform.print_()

            unpacks = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["linalg.unpack"]
            )
            foreach_unpack = transform.ForeachOp([], (unpacks,))
            with ir.InsertionPoint(foreach_unpack.body):
                unpack_op = foreach_unpack.bodyTargets[0]
                tiled_unpack = structured.FuseOp(
                    unpack_op, tile_sizes=[TILE_SIZE, TILE_SIZE], apply_cleanup=True
                ).results[0]
                structured.structured_lower_unpack(
                    anytype, anytype, anytype, anytype, tiled_unpack
                )
                transform.yield_()

            # Cleanup.
            with ir.InsertionPoint(
                transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
            ):
                tensor.apply_patterns_tensor_merge_consecutive_insert_extract_slice()
            cleanup(named_seq.bodyTarget)
            # transform.print_()

            transform.yield_()
    return schedule


def vector_copy(ctx: ir.Context) -> ir.Module:
    """
    Specialized schedule for Linalg operations.

    Tiling and vectorization is progressively applied to
    achieve SIMD code generation.

    Args:
        ctx: MLIR context.
    Returns:
        MLIR transform module.
    """
    with ctx, ir.Location.unknown(context=ctx):
        # Create a transform module.
        schedule = ir.Module.create()
        schedule.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule.body):
            named_seq = transform.NamedSequenceOp(
                "__transform_main",
                [transform.any_op_t()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )

        # Create the schedule.
        with ir.InsertionPoint(named_seq.body):
            anytype = transform.any_op_t()
            # transform.print_()

            func = structured.MatchOp.match_op_names(
                named_seq.bodyTarget, ["func.func"]
            ).result
            structured.structured_vectorize_children_and_apply_patterns(anytype, func)
            cleanup(named_seq.bodyTarget)

            transform.yield_()

        return schedule


def lower_to_llvm(module: ir.Module) -> ir.Module:
    """
    Lower MLIR ops within the module to MLIR LLVM IR dialect.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        MLIR module with lowered IR.
    """
    pack_sched = pack_gemm(module.context)
    pack_sched.body.operations[0].apply(module)
    # print(module)

    # Apply initial transformations using schedule.
    sched = tile_and_vector_gemm(module.context)
    sched.body.operations[0].apply(module)
    # print(module)

    # # Build pipeline.
    pm = PassManager("builtin.module", module.context)

    # Preprocess.
    # Use standard C interface wrappers for functions.
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

    pm.run(module.operation)
    # print(module)

    sched = vector_copy(module.context)
    sched.body.operations[0].apply(module)
    # print(module)

    # Lower to LLVM.
    pm = PassManager("builtin.module", module.context)
    pm.add("convert-linalg-to-loops")
    pm.add("expand-strided-metadata")
    pm.add("canonicalize")

    pm.add("convert-vector-to-scf")
    pm.add("lower-affine")
    pm.add("convert-scf-to-cf")
    pm.add("convert-vector-to-llvm")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    # Return the same module which now holds LLVM IR dialect ops.
    return module


@torch.compile(dynamic=False, backend=ai_bench.mlir.cpu_backend(lower_to_llvm))
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert all(dim % TILE_SIZE == 0 for dim in A.shape), (
            f"A shape must be multiple of {TILE_SIZE}"
        )
        assert all(dim % TILE_SIZE == 0 for dim in B.shape), (
            f"B shape must be multiple of {TILE_SIZE}"
        )

        return torch.matmul(A, B)
