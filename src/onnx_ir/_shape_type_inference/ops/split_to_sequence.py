"""SplitToSequence operation inferrer for ONNX IR nodes."""

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


class SplitToSequenceInferrer(_common.NodeInferrer):
    """Inferrer for SplitToSequence operations."""

    def __init__(self) -> None:
        """Initialize the SplitToSequence inferrer."""
        super().__init__("SplitToSequence", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least input data
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for SplitToSequence operations."""
        assert node.inputs[0] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="SplitToSequence input shape is not known.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="SplitToSequence input cannot be a scalar.")

        # Get axis attribute (default is 0)
        axis = node.attributes.get_int("axis", 0)

        try:
            axis = _handle_negative_axis(axis, rank)
        except ValueError as e:
            return _common.InferenceResult(failure=str(e))

        # Get keepdims attribute (default is 1)
        keepdims = node.attributes.get_int("keepdims", 1) != 0

        # Check if we have a split input (second input)
        split_tensor = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            split_tensor = ir.convenience.get_const_tensor(node.inputs[1])

        # For SplitToSequence, the output is a sequence type
        # The shape of the sequence depends on how many splits are made

        if split_tensor is not None:
            # Split sizes are specified
            split_sizes = split_tensor.numpy().tolist()
            if not isinstance(split_sizes, list):
                split_sizes = [split_sizes]

            # Number of sequence elements = number of split sizes
            sequence_length = len(split_sizes)
        else:
            # Equal splits - need to determine how many
            # Without knowing the split size, we can't determine the sequence length
            sequence_length = None

        # The output is a sequence, so we need to represent its shape
        # For symbolic inference, we'll represent this as a 1D shape with the sequence length
        if sequence_length is not None:
            output_shape = ir.Shape([sequence_length])
        else:
            output_shape = ir.Shape([None])  # Unknown sequence length

        # Output type would be a sequence type containing tensors
        # For now, we'll approximate with the input type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
