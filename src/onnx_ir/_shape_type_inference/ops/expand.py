"""Expand operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common
from onnx_ir._shape_type_inference.ops.standard_ops import broadcast_shapes_bidirectional


class ExpandInferrer(_common.NodeInferrer):
    """Inferrer for Expand operations."""

    def __init__(self) -> None:
        """Initialize the Expand inferrer."""
        super().__init__("Expand", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Expand operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Expand input shape is not known.")

        # Get the target shape from the second input
        shape_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if shape_tensor is not None:
            target_shape_values = shape_tensor.numpy().tolist()
            target_shape = ir.Shape(target_shape_values)
            
            # Use broadcasting logic to compute output shape
            output_shape = broadcast_shapes_bidirectional(input_shape, target_shape)
        else:
            # Handle case where target shape is not constant
            shape_input_shape = node.inputs[1].shape
            if shape_input_shape is None or len(shape_input_shape) != 1:
                return _common.InferenceResult(
                    failure="Expand shape input must be a 1D tensor with known shape."
                )
            
            target_rank = shape_input_shape[0]
            if not isinstance(target_rank, int):
                return _common.InferenceResult(
                    failure="Expand target rank must be statically known."
                )
            
            # Create output shape with unknown dimensions
            output_shape = ir.Shape([None] * target_rank)

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )