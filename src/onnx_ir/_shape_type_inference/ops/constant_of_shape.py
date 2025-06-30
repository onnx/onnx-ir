"""ConstantOfShape operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ConstantOfShapeInferrer(_common.NodeInferrer):
    """Inferrer for ConstantOfShape operations."""

    def __init__(self) -> None:
        """Initialize the ConstantOfShape inferrer."""
        super().__init__("ConstantOfShape", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ConstantOfShape operations."""
        assert node.inputs[0] is not None

        shape_input = node.inputs[0]
        shape_input_shape = shape_input.shape

        if shape_input_shape is None:
            return _common.InferenceResult(failure="ConstantOfShape input shape is not known.")

        # Input should be a 1D tensor containing the target shape
        if len(shape_input_shape) != 1:
            return _common.InferenceResult(
                failure="ConstantOfShape input must be a 1D tensor."
            )

        # Try to get the shape values if they're constant
        shape_tensor = ir.convenience.get_const_tensor(shape_input)
        if shape_tensor is not None:
            target_shape_values = shape_tensor.numpy().tolist()
            if not isinstance(target_shape_values, list):
                target_shape_values = [target_shape_values]
            output_shape = ir.Shape(target_shape_values)
        else:
            # Shape is not constant, determine rank from input shape
            rank = shape_input_shape[0]
            if isinstance(rank, int):
                output_shape = ir.Shape([None] * rank)
            else:
                # Even the rank is unknown
                return _common.InferenceResult(
                    failure="ConstantOfShape requires statically known rank."
                )

        # Get value attribute to determine output type
        value_attr = node.attributes.get_tensor("value")
        if value_attr is not None:
            output_type = value_attr.dtype
        else:
            # Default value is 0.0 (float32)
            output_type = ir.TensorType.FLOAT32

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
