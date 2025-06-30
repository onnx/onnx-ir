"""OneHot operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _handle_negative_axis(axis: int, rank: int) -> int:
    """Handle negative axis by converting to positive."""
    if axis < 0:
        axis += rank
    if axis < 0 or axis > rank:  # Note: axis can equal rank for insertion
        raise ValueError(f"Axis {axis} is out of bounds for rank {rank}")
    return axis


class OneHotInferrer(_common.NodeInferrer):
    """Inferrer for OneHot operations."""

    def __init__(self) -> None:
        """Initialize the OneHot inferrer."""
        super().__init__("OneHot", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(3)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for OneHot operations."""
        assert node.inputs[0] is not None  # indices
        assert node.inputs[1] is not None  # depth
        assert node.inputs[2] is not None  # values
        
        indices_shape = node.inputs[0].shape
        depth_shape = node.inputs[1].shape
        values_shape = node.inputs[2].shape
        
        if indices_shape is None:
            return _common.InferenceResult(failure="OneHot indices input shape is not known.")
        if depth_shape is None:
            return _common.InferenceResult(failure="OneHot depth input shape is not known.")
        if values_shape is None:
            return _common.InferenceResult(failure="OneHot values input shape is not known.")

        # Depth should be a scalar
        if len(depth_shape) != 0:
            return _common.InferenceResult(failure="OneHot depth input must be a scalar.")
        
        # Values should be a 1D tensor with 2 elements [off_value, on_value]
        if len(values_shape) != 1 or values_shape[0] != 2:
            return _common.InferenceResult(failure="OneHot values input must be a 1D tensor with 2 elements.")

        # Get axis attribute (default is -1)
        axis = node.attributes.get_int("axis", -1)
        
        input_rank = len(indices_shape)
        output_rank = input_rank + 1
        
        try:
            axis = _handle_negative_axis(axis, output_rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Get depth value if it's constant
        depth_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if depth_tensor is not None:
            depth_value = int(depth_tensor.numpy().item())
        else:
            # Depth is not constant, use symbolic dimension
            depth_value = None

        # Construct output shape by inserting depth dimension at specified axis
        output_dims = list(indices_shape.dims)
        output_dims.insert(axis, depth_value)
        
        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[2].type  # Output type matches values type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )