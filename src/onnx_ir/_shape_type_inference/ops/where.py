"""Where operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common
from onnx_ir._shape_type_inference.ops.standard_ops import broadcast_shapes_bidirectional


class WhereInferrer(_common.NodeInferrer):
    """Inferrer for Where operations."""

    def __init__(self) -> None:
        """Initialize the Where inferrer."""
        super().__init__("Where", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(3)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Where operations."""
        assert node.inputs[0] is not None  # condition
        assert node.inputs[1] is not None  # x
        assert node.inputs[2] is not None  # y

        condition_shape = node.inputs[0].shape
        x_shape = node.inputs[1].shape
        y_shape = node.inputs[2].shape

        if condition_shape is None:
            return _common.InferenceResult(failure="Where condition input shape is not known.")
        if x_shape is None:
            return _common.InferenceResult(failure="Where x input shape is not known.")
        if y_shape is None:
            return _common.InferenceResult(failure="Where y input shape is not known.")

        # Check that x and y have compatible types
        x_type = node.inputs[1].type
        y_type = node.inputs[2].type
        if x_type is not None and y_type is not None and x_type != y_type:
            return _common.InferenceResult(
                failure=f"Where x and y input types do not match: {x_type} vs {y_type}."
            )

        # Broadcast all three inputs to determine output shape
        # First broadcast x and y
        xy_shape = broadcast_shapes_bidirectional(x_shape, y_shape)
        # Then broadcast result with condition
        output_shape = broadcast_shapes_bidirectional(condition_shape, xy_shape)

        output_type = x_type if x_type is not None else y_type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
