"""GatherElements operation inferrer for ONNX IR nodes."""

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


class GatherElementsInferrer(_common.NodeInferrer):
    """Inferrer for GatherElements operations."""

    def __init__(self) -> None:
        """Initialize the GatherElements inferrer."""
        super().__init__("GatherElements", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for GatherElements operations."""
        assert node.inputs[0] is not None  # data
        assert node.inputs[1] is not None  # indices

        data_shape = node.inputs[0].shape
        indices_shape = node.inputs[1].shape

        if data_shape is None:
            return _common.InferenceResult(
                failure="GatherElements data input shape is not known."
            )
        if indices_shape is None:
            return _common.InferenceResult(
                failure="GatherElements indices input shape is not known."
            )

        data_rank = len(data_shape)
        indices_rank = len(indices_shape)

        if data_rank == 0:
            return _common.InferenceResult(
                failure="GatherElements data input cannot be a scalar."
            )
        if indices_rank == 0:
            return _common.InferenceResult(
                failure="GatherElements indices input cannot be a scalar."
            )

        # Data and indices must have the same rank
        if data_rank != indices_rank:
            return _common.InferenceResult(
                failure=f"GatherElements data and indices must have same rank, got {data_rank} vs {indices_rank}."
            )

        # Get axis attribute (default is 0)
        axis = node.attributes.get_int("axis", 0)

        try:
            axis = _handle_negative_axis(axis, data_rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # For GatherElements, the output shape is the same as the indices shape
        # All dimensions must be compatible between data and indices except possibly the axis dimension
        for i in range(data_rank):
            if i != axis:
                data_dim = data_shape.dims[i]
                indices_dim = indices_shape.dims[i]
                # For symbolic inference, we assume they're compatible
                # In practice, this would need runtime verification

        output_shape = indices_shape
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
