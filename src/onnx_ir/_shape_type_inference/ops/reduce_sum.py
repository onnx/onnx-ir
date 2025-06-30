"""ReduceSum operation inferrer for ONNX IR nodes."""

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


class ReduceSumInferrer(_common.NodeInferrer):
    """Inferrer for ReduceSum operations."""

    def __init__(self) -> None:
        """Initialize the ReduceSum inferrer."""
        super().__init__("ReduceSum", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least input data
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ReduceSum operations."""
        assert node.inputs[0] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="ReduceSum input shape is not known.")

        rank = len(input_shape)

        # Get axes to reduce
        axes_attr = node.attributes.get_ints("axes")
        axes_tensor = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            axes_tensor = ir.convenience.get_const_tensor(node.inputs[1])

        if axes_attr is not None:
            axes = axes_attr
        elif axes_tensor is not None:
            axes = axes_tensor.numpy().tolist()
            if not isinstance(axes, list):
                axes = [axes]
        else:
            # No axes specified - reduce all dimensions
            axes = list(range(rank))

        # Normalize axes
        try:
            normalized_axes = set()
            for axis in axes:
                normalized_axis = _handle_negative_axis(axis, rank)
                normalized_axes.add(normalized_axis)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Get keepdims attribute (default is True for newer opsets, False for older)
        keepdims = node.attributes.get_int("keepdims", 1) != 0

        # Calculate output shape
        if keepdims:
            # Keep reduced dimensions as size 1
            output_dims = []
            for i in range(rank):
                if i in normalized_axes:
                    output_dims.append(1)
                else:
                    output_dims.append(input_shape.dims[i])
        else:
            # Remove reduced dimensions
            output_dims = []
            for i in range(rank):
                if i not in normalized_axes:
                    output_dims.append(input_shape.dims[i])

        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
