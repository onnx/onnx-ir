"""GatherND operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class GatherNDInferrer(_common.NodeInferrer):
    """Inferrer for GatherND operations."""

    def __init__(self) -> None:
        """Initialize the GatherND inferrer."""
        super().__init__("GatherND", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for GatherND operations."""
        assert node.inputs[0] is not None  # data
        assert node.inputs[1] is not None  # indices

        data_shape = node.inputs[0].shape
        indices_shape = node.inputs[1].shape

        if data_shape is None:
            return _common.InferenceResult(failure="GatherND data input shape is not known.")
        if indices_shape is None:
            return _common.InferenceResult(
                failure="GatherND indices input shape is not known."
            )

        data_rank = len(data_shape)
        indices_rank = len(indices_shape)

        if data_rank == 0:
            return _common.InferenceResult(failure="GatherND data input cannot be a scalar.")
        if indices_rank == 0:
            return _common.InferenceResult(
                failure="GatherND indices input cannot be a scalar."
            )

        # Get batch_dims attribute (default is 0)
        batch_dims = node.attributes.get_int("batch_dims", 0)

        if batch_dims < 0:
            return _common.InferenceResult(failure="GatherND batch_dims must be non-negative.")
        if batch_dims >= min(data_rank, indices_rank):
            return _common.InferenceResult(
                failure=f"GatherND batch_dims {batch_dims} must be less than min(data_rank, indices_rank)."
            )

        # The last dimension of indices contains the coordinates
        if indices_rank == 0:
            return _common.InferenceResult(failure="GatherND indices cannot be a scalar.")

        indices_last_dim = indices_shape.dims[-1]

        # For symbolic inference, we'll construct the output shape
        # Output shape: indices.shape[:-1] + data.shape[batch_dims + indices.shape[-1]:]

        # Try to get the coordinate dimension size if it's constant
        if isinstance(indices_last_dim, int):
            coord_size = indices_last_dim
        else:
            # Indices last dimension is symbolic, we can't determine exact output shape
            coord_size = None

        if coord_size is not None:
            if batch_dims + coord_size > data_rank:
                return _common.InferenceResult(
                    failure=f"GatherND coordinate size {coord_size} with batch_dims {batch_dims} "
                    f"exceeds data rank {data_rank}."
                )

            # Build output shape
            output_dims = (
                list(indices_shape.dims[:-1])  # All but last dim of indices
                + list(data_shape.dims[batch_dims + coord_size :])  # Remaining data dims
            )
        else:
            # Coordinate size is unknown, output shape is partially unknown
            # We know the batch dimensions from indices, but not the data dimensions
            output_dims = list(indices_shape.dims[:-1]) + [None]  # Unknown trailing dimensions

        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
