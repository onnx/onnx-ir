"""Standard Inferrers for ONNX IR nodes."""

from __future__ import annotations

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ElementwiseInferrer(_common.NodeInferrer):
    """Base class for elementwise operation inferrers."""

    def __init__(self, op_type: str, opsets: Collection[int] | None = None) -> None:
        """Initialize the elementwise inferrer with the operation type."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__(op_type, opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for elementwise operations."""
        if len(node.inputs) != 1:
            return _common.InferenceResult(
                failure=f"Elementwise operation must have exactly one input, got {len(node.inputs)}."
            )
        if node.inputs[0] is None:
            return _common.InferenceResult(
                failure="Elementwise operation input cannot be None."
            )
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Elementwise operation must have exactly one output, got {len(node.outputs)}."
            )

        return _common.InferenceResult(
            (ir.Value(shape=node.inputs[0].shape, type=node.inputs[0].type),)
        )


def broadcast_shapes_bidirectional(shape1: ir.Shape, shape2: ir.Shape) -> ir.Shape:
    """Broadcast two shapes bidirectionally.

    Args:
        shape1: The first shape to broadcast.
        shape2: The second shape to broadcast.

    Returns:
        A new shape that is the result of broadcasting both shapes.
    """
    # TODO: Use _common.get_expr and use sympy for broadcasting logic


class BinaryInferrer(_common.NodeInferrer):
    """Base class for binary operation inferrers."""

    def __init__(self, op_type: str, opsets: Collection[int] | None = None) -> None:
        """Initialize the binary inferrer with the operation type."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__(op_type, opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for binary operations."""
        if len(node.inputs) != 2:
            return _common.InferenceResult(
                failure=f"Binary operation must have exactly two inputs, got {len(node.inputs)}."
            )
        if node.inputs[0] is None or node.inputs[1] is None:
            return _common.InferenceResult(failure="Binary operation inputs cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Binary operation must have exactly one output, got {len(node.outputs)}."
            )
        first_type = node.inputs[0].type
        second_type = node.inputs[1].type
        if first_type is not None and second_type is not None and first_type != second_type:
            return _common.InferenceResult(
                failure=f"Input types do not match: {first_type} vs {second_type}."
            )
