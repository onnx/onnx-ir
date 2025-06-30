"""Gather operation inferrer for ONNX IR nodes."""

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


class GatherInferrer(_common.NodeInferrer):
    """Inferrer for Gather operations."""

    def __init__(self) -> None:
        """Initialize the Gather inferrer."""
        super().__init__("Gather", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Gather operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None
        
        data_shape = node.inputs[0].shape
        indices_shape = node.inputs[1].shape
        
        if data_shape is None:
            return _common.InferenceResult(failure="Gather data input shape is not known.")
        if indices_shape is None:
            return _common.InferenceResult(failure="Gather indices input shape is not known.")

        rank = len(data_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Gather data input cannot be a scalar.")

        # Get axis attribute (default is 0)
        axis = node.attributes.get_int("axis", 0)
        
        try:
            axis = _handle_negative_axis(axis, rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Output shape: data_shape[:axis] + indices_shape + data_shape[axis+1:]
        output_dims = (
            list(data_shape.dims[:axis]) +
            list(indices_shape.dims) +
            list(data_shape.dims[axis + 1:])
        )
        
        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )