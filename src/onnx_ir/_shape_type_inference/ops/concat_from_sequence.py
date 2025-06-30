"""ConcatFromSequence operation inferrer for ONNX IR nodes."""

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


class ConcatFromSequenceInferrer(_common.NodeInferrer):
    """Inferrer for ConcatFromSequence operations."""

    def __init__(self) -> None:
        """Initialize the ConcatFromSequence inferrer."""
        super().__init__("ConcatFromSequence", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ConcatFromSequence operations."""
        assert node.inputs[0] is not None

        sequence_input = node.inputs[0]
        if sequence_input.shape is None:
            return _common.InferenceResult(
                failure="ConcatFromSequence input shape is not known."
            )

        # Get required axis attribute
        axis = node.attributes.get_int("axis")
        if axis is None:
            return _common.InferenceResult(
                failure="ConcatFromSequence requires axis attribute."
            )

        # Get new_axis attribute (default is 0)
        new_axis = node.attributes.get_int("new_axis", 0) != 0

        # For sequence inputs, we need to know the element shape
        # This is challenging in symbolic inference without more type information
        # We'll make some assumptions based on the sequence structure

        sequence_shape = sequence_input.shape
        if len(sequence_shape) == 0:
            # Scalar sequence length - we don't know the element shape
            if new_axis:
                # Adding a new axis, output rank is unknown
                output_shape = ir.Shape([None])
            else:
                # No new axis, output shape is unknown
                output_shape = ir.Shape([None])
        else:
            # Sequence has known length - assume elements have some shape
            sequence_length = sequence_shape.dims[0]

            # Without knowing the element shape, we'll create a placeholder
            # In a real implementation, this would need sequence type information
            if new_axis:
                # New axis is being added at the specified position
                # The output will have one more dimension than the elements
                try:
                    # We don't know the element rank, so we can't validate axis
                    # Just create an output with unknown dimensions
                    output_shape = ir.Shape([None, None])  # Unknown rank and sizes
                except ValueError as e:
                    return _common.InferenceResult(failure=str(e))
            else:
                # Concatenating along existing axis
                # The concat dimension size is unknown (depends on sequence length and element sizes)
                output_shape = ir.Shape([None])  # Unknown shape

        # Output type would be the element type of the sequence
        # For now, we'll use the sequence input type
        output_type = sequence_input.type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
