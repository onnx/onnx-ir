"""Slice operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _handle_negative_axis(axis: int, rank: int) -> int:
    """Handle negative axis by converting to positive."""
    if axis < 0:
        axis += rank
    return max(0, min(axis, rank - 1))


class SliceInferrer(_common.NodeInferrer):
    """Inferrer for Slice operations."""

    def __init__(self) -> None:
        """Initialize the Slice inferrer."""
        super().__init__("Slice", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least data input
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Slice operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Slice input shape is not known.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Slice input cannot be a scalar.")

        # For symbolic inference, we'll create an output shape with the same rank
        # but potentially different dimensions (unknown sizes due to slicing)
        output_dims = []
        
        # Try to get slice parameters if they're constant
        starts_tensor = None
        ends_tensor = None
        axes_tensor = None
        steps_tensor = None
        
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            starts_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if len(node.inputs) >= 3 and node.inputs[2] is not None:
            ends_tensor = ir.convenience.get_const_tensor(node.inputs[2])
        if len(node.inputs) >= 4 and node.inputs[3] is not None:
            axes_tensor = ir.convenience.get_const_tensor(node.inputs[3])
        if len(node.inputs) >= 5 and node.inputs[4] is not None:
            steps_tensor = ir.convenience.get_const_tensor(node.inputs[4])

        if (starts_tensor is not None and ends_tensor is not None and 
            axes_tensor is not None):
            # We have constant slice parameters
            starts = starts_tensor.numpy().tolist()
            ends = ends_tensor.numpy().tolist()
            axes = axes_tensor.numpy().tolist()
            steps = steps_tensor.numpy().tolist() if steps_tensor is not None else [1] * len(axes)
            
            if not isinstance(starts, list):
                starts = [starts]
            if not isinstance(ends, list):
                ends = [ends]
            if not isinstance(axes, list):
                axes = [axes]
            if not isinstance(steps, list):
                steps = [steps]
            
            # Start with input dimensions
            output_dims = list(input_shape.dims)
            
            # Apply slicing to specified axes
            for i, axis in enumerate(axes):
                axis = _handle_negative_axis(axis, rank)
                if axis < rank:
                    # For symbolic inference, we can't compute exact slice sizes
                    # but we know the dimension exists and may be smaller
                    output_dims[axis] = None  # Unknown size due to slicing
        else:
            # Parameters are not constant, output shape is unknown
            output_dims = [None] * rank

        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )