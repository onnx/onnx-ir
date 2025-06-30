"""SequenceInsert operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class SequenceInsertInferrer(_common.NodeInferrer):
    """Inferrer for SequenceInsert operations."""

    def __init__(self) -> None:
        """Initialize the SequenceInsert inferrer."""
        super().__init__("SequenceInsert", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # At least sequence and tensor
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for SequenceInsert operations."""
        assert node.inputs[0] is not None  # sequence
        assert node.inputs[1] is not None  # tensor

        sequence_shape = node.inputs[0].shape
        tensor_shape = node.inputs[1].shape

        if sequence_shape is None:
            return _common.InferenceResult(
                failure="SequenceInsert sequence input shape is not known."
            )
        if tensor_shape is None:
            return _common.InferenceResult(
                failure="SequenceInsert tensor input shape is not known."
            )

        # Check position input if present
        if len(node.inputs) >= 3 and node.inputs[2] is not None:
            position_shape = node.inputs[2].shape
            if position_shape is None:
                return _common.InferenceResult(
                    failure="SequenceInsert position input shape is not known."
                )
            if len(position_shape) != 0:
                return _common.InferenceResult(
                    failure="SequenceInsert position input must be a scalar."
                )

        # For SequenceInsert, we're adding one element to a sequence
        # The output sequence shape depends on the original sequence and the new element

        # If original sequence has shape [N, ...], and we're inserting an element,
        # the new sequence has shape [N+1, ...]
        if len(sequence_shape) > 0:
            original_length = sequence_shape.dims[0]
            if isinstance(original_length, int):
                new_length = original_length + 1
            else:
                # Symbolic length
                import sympy

                original_expr = _common.get_expr(sequence_shape, 0)
                new_length = original_expr + sympy.Integer(1)

            output_dims = [new_length] + list(sequence_shape.dims[1:])
            output_shape = ir.Shape(output_dims)
        else:
            # Empty sequence - output will have one element
            output_shape = ir.Shape([1] + list(tensor_shape.dims))

        # Output type is the sequence type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
