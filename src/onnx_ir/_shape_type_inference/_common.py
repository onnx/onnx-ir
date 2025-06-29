"""Symbolic shape inference for ONNX IR."""

import abc
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np

import onnx_ir as ir

if TYPE_CHECKING:
    import sympy


def get_expr(shape: ir.Shape, index: int) -> sympy.Expr:
    """Get the expression or value at a specific index in the shape.

    Args:
        shape: The shape to get the expression from.
        index: The index of the dimension to get.

    Returns:
        The expression or value at the specified index.
    """
    import sympy

    dim = shape[index]
    if isinstance(dim, ir.SymbolicDim):
        if dim.expr is not None:
            return dim.expr
        return sympy.Symbol(dim.value)
    return sympy.Integer(dim)


def set_expr(shape: ir.Shape, index: int, expr: sympy.Expr | int) -> None:
    """Set the expression or value at a specific index in the shape.

    Args:
        shape: The shape to set the expression in.
        index: The index of the dimension to set.
        expr: The expression or value to set at the specified index.
    """
    from sympy.utilities.misc import as_int
    if isinstance(expr, (int, np.integer)):
        shape[index] = int(expr)
        return
    assert isinstance(expr, sympy.Expr), f"Expected sympy.Expr or int, got {type(expr)}"
    expr = sympy.sympify(expr)
    if expr.is_integer:
        shape[index] = as_int(expr)
        return
    shape[index] = ir.SymbolicDim(str(expr), expr=expr)


class NodeInferencer(abc.ABC):
    """Base class for node inferencers.

    This class provides a common interface for all node inferencers.
    """

    def __init__(self, op_type: str, opsets: Collection[int], domain: str = "") -> None:
        """Initialize the node inferencer.

        Args:
            op_type: The type of the operation.
            opsets: A collection of ONNX opset versions supported by this inferencer.
            domain: The domain of the operation, default is an empty string.
        """
        self.op_type = op_type
        self.opsets = opsets
        self.domain = domain

    @abc.abstractmethod
    def check(self, node: ir.Node) -> None:
        """Check if the node is valid for this inferencer."""
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, node: ir.Node) -> Sequence[ir.Value]:
        """Infer the shape for the node.

        Args:
            node: The ONNX node to infer the type and shape for.

        Returns:
            A sequence of ONNX values containing the inferred shapes.
        """
        raise NotImplementedError
