from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass

from lighthouse import utils as lh_utils
from lighthouse.ingress.torch import import_from_model
from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.dialects import func
from mlir.dialects import bufferization

import torch
import torch.nn as nn
from torch_mlir.fx import OutputType


@dataclass
class ModelResult:
    shape: list[int]
    dtype: torch.dtype
    device: torch.device


class JITModel:
    def __init__(
        self,
        module: ir.Module,
        results: list[ModelResult],
        shared_libs: Sequence[str] = [],
        entry_func: str = "main",
    ):
        """
        Initialize the JITModel object.
        Typically called from the `@lighthouse.runtime.torch.jit` decorator.

        Args:
            fn_compile_mlir: Function to lower imported MLIR to LLVM IR dialect.
            model: PyTorch model to be compiled through MLIR.
            dialect: The target dialect for MLIR IR imported from PyTorch model.
            ir_context: An optional MLIR context to use for compilation.
                If not provided, a new default context is created.
            shared_libs: Paths to external runtime libraries used to execute
                compiled MLIR function.
        """
        self.eng = ExecutionEngine(module, opt_level=3, shared_libs=shared_libs)
        self.eng.initialize()
        self.fn = self.eng.lookup(entry_func)
        self.results = results

    def __call__(
        self,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Jit the PyTorch model and call the MLIR function.

        Args:
            args: The positional arguments to pass the MLIR function.
                If all arguments are PyTorch tensors, then they are converted
                to packed C-type arguments before passing to the MLIR function.
                Otherwise, `args` are passed directly as is.
            model_args: The optional positional arguments to the Pytorch model
                required to jit into MLIR.
                If not provided, `args` are used instead.
            kwargs: The keyword arguments to the PyTorch model required to jit
                into MLIR.

        Returns:
            Any: The result of the MLIR function call.
        """

        outs = [
            torch.empty(res.shape, dtype=res.dtype, device=res.device)
            for res in self.results
        ]

        mlir_args = list(args)
        mlir_args.append(outs)
        mlir_args = lh_utils.torch.to_packed_args(mlir_args)
        self.fn(mlir_args)

        return outs


class MLIRBackend:
    def __init__(
        self,
        device: torch.device,
        fn_compile: Callable[[ir.Module], ir.Module],
        dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
    ):
        """
        Initialize the JITModel object.
        Typically called from the `@lighthouse.runtime.torch.jit` decorator.

        Args:
            fn_compile: Function to lower imported MLIR to LLVM IR dialect.
            model: PyTorch model to be compiled through MLIR.
            dialect: The target dialect for MLIR IR imported from PyTorch model.
            ir_context: An optional MLIR context to use for compilation.
                If not provided, a new default context is created.
            shared_libs: Paths to external runtime libraries used to execute
                compiled MLIR function.
        """
        self.device = device
        self.fn_compile = fn_compile
        self.dialect = OutputType.get(dialect)
        self.ctx = ir_context if ir_context is not None else ir.Context()
        self.shared_libs = shared_libs
        self.entry_func = "main"

    def _get_entry_func(self, module: ir.Module) -> func.FuncOp | None:
        assert len(module.operation.regions) == 1, "Expected module with one region"
        assert len(module.operation.regions[0].blocks) == 1, (
            "Expected module with one block"
        )

        for op in module.operation.regions[0].blocks[0].operations:
            if isinstance(op.opview, func.FuncOp) and op.opview.name.value == self.entry_func:
                return op.opview
        return None

    def _get_results(self, func_op: func.FuncOp) -> list[ModelResult]:
        results = []
        for res in func_op.type.results:
            assert isinstance(res, ir.RankedTensorType), "Expected ranked tensor output"
            res_dtype = lh_utils.torch.dtype_from_mlir_type(res.element_type)
            results.append(
                ModelResult(shape=res.shape, dtype=res_dtype, device=self.device)
            )
        return results

    def _move_results_to_args(self, func_op: func.FuncOp):
        results = func_op.type.results
        if len(results) == 0:
            return

        with func_op.context, func_op.location as loc:
            # Append results to function args and its block args
            new_func_type = ir.FunctionType.get(inputs=[*func_op.type.inputs, *results], results=results)
            func_op.function_type = ir.TypeAttr.get(new_func_type)
            for res in results:
                func_op.entry_block.add_argument(res, loc)
            # TODO: Transfer the result attributes to arg attributes

            # Ensure outputs are written to the new result arguments
            return_op: func.ReturnOp = func_op.entry_block.operations[-1]
            with ir.InsertionPoint.at_block_terminator(func_op.entry_block):
                new_returns = []
                for idx, arg in enumerate(func_op.arguments[-len(results):]):
                    buf_op = bufferization.materialize_in_destination(arg.type, return_op.operands[idx], arg)
                    new_returns.append(buf_op)
                func.return_(new_returns)
                return_op.erase()

    def __call__(
        self, model: torch.fx.GraphModule, inputs: list[torch.Tensor]
    ) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        """
        Jit the PyTorch model and call the MLIR function.

        Args:
            args: The positional arguments to pass the MLIR function.
                If all arguments are PyTorch tensors, then they are converted
                to packed C-type arguments before passing to the MLIR function.
                Otherwise, `args` are passed directly as is.
            model_args: The optional positional arguments to the Pytorch model
                required to jit into MLIR.
                If not provided, `args` are used instead.
            kwargs: The keyword arguments to the PyTorch model required to jit
                into MLIR.

        Returns:
            Any: The result of the MLIR function call.
        """
        # Convert into MLIR IR.
        mlir_mod = import_from_model(
            model,
            sample_args=inputs,
            dialect=self.dialect,
            ir_context=self.ctx,
        )

        # Preprocess entry function.
        func_op: func.FuncOp = self._get_entry_func(mlir_mod)
        if func_op is None:
            raise ValueError(f"Failed to find MLIR entry: {self.entry_func}")

        # Metadata about function returns is stored for later
        # output buffer allocation.
        results = self._get_results(func_op)
        # Add extra arguments to store results in external buffers.
        self._move_results_to_args(func_op)

        # Transform MLIR module.
        mlir_mod = self.fn_compile(mlir_mod)

        return JITModel(
            mlir_mod, results, shared_libs=self.shared_libs, entry_func=self.entry_func
        )


def cpu_backend(
    fn_compile: Callable[[ir.Module], ir.Module],
    dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
    """
    Decorator for JIT-compiling a PyTorch model using MLIR.

    When a PyTorch model class is passed, the model must be initialized first.
    Only calls to instantiated model objects are JIT-compiled.

    When a jitted MLIR function is called, input arguments are implicitly converted
    into packed C-type arguments if all inputs are PyTorch tensors.
    Otherwise, the input arguments are directly passed to the jitted function.

    The jitted function signature depends on the provided MLIR compilation function
    `fn_compile` and may differ from the original PyTorch model call signature.
    See 'JITModel' for further calling convention details.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        model: PyTorch model to be compiled with MLIR.
            If a class or None, a decorator is returned.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        kwargs: The keyword arguments for the PyTorch model constructor.

    Returns:
        object: A PyTorch model or a partially bound decorator.
    """
    # TODO: Convert into proper torch.compile backend
    return MLIRBackend(
        torch.device("cpu"),
        fn_compile,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
    )
