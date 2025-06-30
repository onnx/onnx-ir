"""SequenceAt operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class SequenceAtInferrer(_common.NodeInferrer):
    """Inferrer for SequenceAt operations."""

    def __init__(self) -> None:
        """Initialize the SequenceAt inferrer."""
        super().__init__("SequenceAt", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # sequence, position
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for SequenceAt operations."""
        assert node.inputs[0] is not None  # sequence
        assert node.inputs[1] is not None  # position

        sequence_shape = node.inputs[0].shape
        position_shape = node.inputs[1].shape

        if sequence_shape is None:
            return _common.InferenceResult(
                failure="SequenceAt sequence input shape is not known."
            )
        if position_shape is None:
            return _common.InferenceResult(
                failure="SequenceAt position input shape is not known."
            )

        # Position should be a scalar
        if len(position_shape) != 0:
            return _common.InferenceResult(
                failure="SequenceAt position input must be a scalar."
            )

        # For SequenceAt, we're extracting one element from a sequence
        # The output shape would be the shape of the sequence elements
        # Without full sequence type information, we'll make an approximation

        # If sequence has shape [N, ...], the element shape would be [...]
        if len(sequence_shape) > 0:
            element_shape = ir.Shape(list(sequence_shape.dims[1:]))  # Remove first dimension
        else:
            # Scalar sequence - element is also scalar
            element_shape = ir.Shape([])

        # Output type would be the element type of the sequence
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=element_shape, type=output_type),)
        )
