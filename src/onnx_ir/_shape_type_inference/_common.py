"""Symbolic shape inference for ONNX IR."""

from __future__ import annotations

import abc
import dataclasses
import functools
from collections.abc import Collection, Sequence
from typing import Any, Callable

import sympy

import onnx_ir as ir


MAX_SUPPORTED_OPSET = 23


def get_expr(shape: ir.Shape, index: int) -> sympy.Expr:
    """Get the expression or value at a specific index in the shape.

    Args:
        shape: The shape to get the expression from.
        index: The index of the dimension to get.

    Returns:
        The expression or value at the specified index.
    """
    dim = shape[index]
    if isinstance(dim, ir.SymbolicDim):
        if dim.expr is not None:
            return dim.expr
        if dim.value is None:
            return sympy.Symbol("__unknown__")
        return sympy.Symbol(dim.value)
    return sympy.Integer(dim)


@dataclasses.dataclass
class InferenceResult:
    values: Sequence[ir.Value] | None = None
    failure: str | None = None


class NodeInferrer(abc.ABC):
    """Base class for node inferrers.

    This class provides a common interface for all node inferrers.
    """

    def __init__(self, op_type: str, opsets: Collection[int], domain: str = "") -> None:
        """Initialize the node inferrer.

        Args:
            op_type: The type of the operation.
            opsets: A collection of ONNX opset versions supported by this inferrer.
            domain: The domain of the operation, default is an empty string.
        """
        self.op_type = op_type
        self.opsets = opsets
        self.domain = domain

    def __repr__(self) -> str:
        """Return a string representation of the node inferrer."""
        return f"{self.__class__.__name__}(op_type={self.op_type}, opsets={self.opsets}, domain={self.domain})"

    @abc.abstractmethod
    def infer(self, node: ir.Node) -> InferenceResult:
        """Infer the shape for the node.

        Args:
            node: The ONNX node to infer the type and shape for.

        Returns:
            A sequence of ONNX values containing the inferred shapes.
        """
        raise NotImplementedError


def requires_non_none_inputs(
    count: int, /
) -> Callable[
    [Callable[[Any, ir.Node], InferenceResult]], Callable[[Any, ir.Node], InferenceResult]
]:
    """Ensure that the node has a specific number of non-None inputs.

    Args:
        count: The exact number of non-None inputs required for the node.

    Returns:
        A decorator that checks the number of inputs and their non-None status.
    """

    def decorator(
        func: Callable[[Any, ir.Node], InferenceResult],
    ) -> Callable[[Any, ir.Node], InferenceResult]:
        @functools.wraps(func)
        def wrapper(self, node: ir.Node) -> InferenceResult:
            if len(node.inputs) != count:
                return InferenceResult(
                    failure=f"[{node.op_type} must have {count} inputs, got {len(node.inputs)}."
                )
            for i, inp in enumerate(node.inputs):
                if inp is None:
                    return InferenceResult(failure=f"{node.op_type} input {i} cannot be None.")
            return func(self, node)

        return wrapper

    return decorator


def requires_outputs(
    count: int, /
) -> Callable[
    [Callable[[Any, ir.Node], InferenceResult]], Callable[[Any, ir.Node], InferenceResult]
]:
    """Ensure that the node has a specific number of outputs.

    Args:
        count: The exact number of outputs required for the node.

    Returns:
        A decorator that checks the number of outputs.
    """

    def decorator(
        func: Callable[[Any, ir.Node], InferenceResult],
    ) -> Callable[[Any, ir.Node], InferenceResult]:
        @functools.wraps(func)
        def wrapper(self, node: ir.Node) -> InferenceResult:
            if len(node.outputs) != count:
                return InferenceResult(
                    failure=f"[{node.op_type} must have {count} outputs, got {len(node.outputs)}."
                )
            return func(self, node)

        return wrapper

    return decorator


def inclusive_range(start_or_end: int = 0, end: int | None = None) -> range:
    """Create an inclusive range from start to end with a given step.

    Args:
        start_or_end: The starting value of the range.
        end: The ending value of the range (inclusive).

    Returns:
        A range object that includes both start and end.
    """
    if end is None:
        end = start_or_end
        start = 0
    else:
        start = start_or_end

    return range(start, end + 1)
