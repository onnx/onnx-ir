"""TopK operation inferrer for ONNX IR nodes."""

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


class TopKInferrer(_common.NodeInferrer):
    """Inferrer for TopK operations."""

    def __init__(self) -> None:
        """Initialize the TopK inferrer."""
        super().__init__("TopK", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(2)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for TopK operations."""
        assert node.inputs[0] is not None  # X
        assert node.inputs[1] is not None  # K
        
        input_shape = node.inputs[0].shape
        k_shape = node.inputs[1].shape
        
        if input_shape is None:
            return _common.InferenceResult(failure="TopK input shape is not known.")
        if k_shape is None:
            return _common.InferenceResult(failure="TopK K input shape is not known.")

        # K should be a scalar
        if len(k_shape) != 0:
            return _common.InferenceResult(failure="TopK K input must be a scalar.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="TopK input cannot be a scalar.")

        # Get axis attribute (default is -1)
        axis = node.attributes.get_int("axis", -1)
        
        try:
            axis = _handle_negative_axis(axis, rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Try to get K value if it's constant
        k_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if k_tensor is not None:
            k_value = int(k_tensor.numpy().item())
        else:
            # K is not constant
            k_value = None

        # Output shape: same as input but with K elements along the specified axis
        output_dims = list(input_shape.dims)
        output_dims[axis] = k_value

        output_shape = ir.Shape(output_dims)
        
        # TopK has two outputs: values and indices
        values_output = ir.Value(shape=output_shape, type=node.inputs[0].type)
        indices_output = ir.Value(shape=output_shape, type=ir.TensorType.INT64)

        return _common.InferenceResult(
            values=(values_output, indices_output)
        )