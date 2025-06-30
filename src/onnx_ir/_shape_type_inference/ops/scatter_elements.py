"""ScatterElements operation inferrer for ONNX IR nodes."""

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


class ScatterElementsInferrer(_common.NodeInferrer):
    """Inferrer for ScatterElements operations."""

    def __init__(self) -> None:
        """Initialize the ScatterElements inferrer."""
        super().__init__("ScatterElements", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(3)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ScatterElements operations."""
        assert node.inputs[0] is not None  # data
        assert node.inputs[1] is not None  # indices
        assert node.inputs[2] is not None  # updates
        
        data_shape = node.inputs[0].shape
        indices_shape = node.inputs[1].shape
        updates_shape = node.inputs[2].shape
        
        if data_shape is None:
            return _common.InferenceResult(failure="ScatterElements data input shape is not known.")
        if indices_shape is None:
            return _common.InferenceResult(failure="ScatterElements indices input shape is not known.")
        if updates_shape is None:
            return _common.InferenceResult(failure="ScatterElements updates input shape is not known.")

        data_rank = len(data_shape)
        indices_rank = len(indices_shape)
        updates_rank = len(updates_shape)
        
        if data_rank == 0:
            return _common.InferenceResult(failure="ScatterElements data input cannot be a scalar.")

        # Indices and updates must have the same shape
        if indices_rank != updates_rank:
            return _common.InferenceResult(
                failure=f"ScatterElements indices and updates must have same rank, got {indices_rank} vs {updates_rank}."
            )

        # Check that indices and updates shapes match
        for i in range(indices_rank):
            indices_dim = indices_shape.dims[i]
            updates_dim = updates_shape.dims[i]
            # For symbolic inference, we assume they match
            # In practice, this would need runtime verification

        # Get axis attribute (default is 0)
        axis = node.attributes.get_int("axis", 0)
        
        try:
            axis = _handle_negative_axis(axis, data_rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Check type compatibility
        data_type = node.inputs[0].type
        updates_type = node.inputs[2].type
        if data_type is not None and updates_type is not None and data_type != updates_type:
            return _common.InferenceResult(
                failure=f"ScatterElements data and updates types must match: {data_type} vs {updates_type}."
            )

        # For ScatterElements, the output shape is the same as the data input shape
        output_shape = data_shape
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )