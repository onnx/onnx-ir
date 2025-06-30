"""Split operation inferrer for ONNX IR nodes."""

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


class SplitInferrer(_common.NodeInferrer):
    """Inferrer for Split operations."""

    def __init__(self) -> None:
        """Initialize the Split inferrer."""
        super().__init__("Split", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least input data
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Split operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Split input shape is not known.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Split input cannot be a scalar.")

        # Get axis attribute (default is 0)
        axis = node.attributes.get_int("axis", 0)
        
        try:
            axis = _handle_negative_axis(axis, rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Get split sizes
        split_attr = node.attributes.get_ints("split")
        split_tensor = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            split_tensor = ir.convenience.get_const_tensor(node.inputs[1])

        num_outputs = len(node.outputs)
        if num_outputs == 0:
            return _common.InferenceResult(failure="Split operation must have at least one output.")

        output_values = []
        
        if split_attr is not None:
            # Split sizes provided as attribute
            if len(split_attr) != num_outputs:
                return _common.InferenceResult(
                    failure=f"Split attribute length {len(split_attr)} does not match output count {num_outputs}."
                )
            
            for split_size in split_attr:
                output_dims = list(input_shape.dims)
                output_dims[axis] = split_size
                output_shape = ir.Shape(output_dims)
                output_values.append(ir.Value(shape=output_shape, type=node.inputs[0].type))
                
        elif split_tensor is not None:
            # Split sizes provided as input tensor
            split_sizes = split_tensor.numpy().tolist()
            if not isinstance(split_sizes, list):
                split_sizes = [split_sizes]
            
            if len(split_sizes) != num_outputs:
                return _common.InferenceResult(
                    failure=f"Split sizes length {len(split_sizes)} does not match output count {num_outputs}."
                )
            
            for split_size in split_sizes:
                output_dims = list(input_shape.dims)
                output_dims[axis] = split_size
                output_shape = ir.Shape(output_dims)
                output_values.append(ir.Value(shape=output_shape, type=node.inputs[0].type))
                
        else:
            # Equal split (default behavior)
            input_dim = input_shape.dims[axis]
            if isinstance(input_dim, int):
                if input_dim % num_outputs != 0:
                    return _common.InferenceResult(
                        failure=f"Cannot equally split dimension {input_dim} into {num_outputs} parts."
                    )
                split_size = input_dim // num_outputs
            else:
                # Symbolic dimension - assume equal split is possible
                split_size = None
            
            for _ in range(num_outputs):
                output_dims = list(input_shape.dims)
                output_dims[axis] = split_size
                output_shape = ir.Shape(output_dims)
                output_values.append(ir.Value(shape=output_shape, type=node.inputs[0].type))

        return _common.InferenceResult(values=output_values)