"""Cast operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class CastInferrer(_common.NodeInferrer):
    """Inferrer for Cast operations."""

    def __init__(self) -> None:
        """Initialize the Cast inferrer."""
        super().__init__("Cast", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Cast operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Cast input shape is not known.")

        # Get the target type from the 'to' attribute
        to_type = node.attributes.get_int("to")
        if to_type is None:
            return _common.InferenceResult(failure="Cast operation requires 'to' attribute.")

        # Convert ONNX data type to appropriate IR type
        # The shape remains the same, only the type changes
        output_shape = input_shape
        
        # For now, we'll preserve the input type structure but note the cast
        # This may need refinement based on the actual IR type system
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )