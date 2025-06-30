"""Compress operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _handle_negative_axis(axis: int, rank: int) -> int:
    """Handle negative axis by converting to positive."""
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Axis {axis} is out of bounds for rank {rank}")
    return axis


class CompressInferrer(_common.NodeInferrer):
    """Inferrer for Compress operations."""

    def __init__(self) -> None:
        """Initialize the Compress inferrer."""
        super().__init__("Compress", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Compress operations."""
        assert node.inputs[0] is not None  # input
        assert node.inputs[1] is not None  # condition
        
        input_shape = node.inputs[0].shape
        condition_shape = node.inputs[1].shape
        
        if input_shape is None:
            return _common.InferenceResult(failure="Compress input shape is not known.")
        if condition_shape is None:
            return _common.InferenceResult(failure="Compress condition shape is not known.")

        input_rank = len(input_shape)
        
        if input_rank == 0:
            return _common.InferenceResult(failure="Compress input cannot be a scalar.")

        # Condition should be 1D
        if len(condition_shape) != 1:
            return _common.InferenceResult(failure="Compress condition must be 1D.")

        # Get axis attribute (default is None, meaning flatten first)
        axis = node.attributes.get_int("axis")
        
        if axis is not None:
            try:
                axis = _handle_negative_axis(axis, input_rank)
            except ValueError as e:
                return _common.InferenceResult(failure=str(e))
            
            # The condition length should match the size of the axis dimension
            condition_length = condition_shape.dims[0]
            axis_length = input_shape.dims[axis]
            # For symbolic inference, we assume they match
            
            # Output shape: same as input but compressed dimension has unknown size
            output_dims = list(input_shape.dims)
            
            # Try to compute compressed size if condition is constant
            condition_tensor = ir.convenience.get_const_tensor(node.inputs[1])
            if condition_tensor is not None:
                condition_values = condition_tensor.numpy()
                compressed_size = int(condition_values.sum())
                output_dims[axis] = compressed_size
            else:
                # Condition is not constant, compressed size is unknown
                output_dims[axis] = None
            
            output_shape = ir.Shape(output_dims)
        else:
            # axis is None - flatten input first, then compress
            # Output is 1D with unknown size
            condition_tensor = ir.convenience.get_const_tensor(node.inputs[1])
            if condition_tensor is not None:
                condition_values = condition_tensor.numpy()
                compressed_size = int(condition_values.sum())
                output_shape = ir.Shape([compressed_size])
            else:
                output_shape = ir.Shape([None])

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )