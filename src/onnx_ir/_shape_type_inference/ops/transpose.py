"""Transpose operation inferrer for ONNX IR nodes."""

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class TransposeInferrer(_common.NodeInferrer):
    """Inferrer for Transpose operations."""

    def __init__(self, opsets: Collection[int] | None = None) -> None:
        """Initialize the Transpose inferrer."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__("Transpose", opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Transpose operations."""
        if len(node.inputs) != 1:
            return _common.InferenceResult(
                failure=f"Transpose operation must have exactly one input, got {len(node.inputs)}."
            )
        if node.inputs[0] is None:
            return _common.InferenceResult(failure="Transpose operation input cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Transpose operation must have exactly one output, got {len(node.outputs)}."
            )

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Transpose input shape cannot be None.")

        rank = len(input_shape)

        # Get permutation from attributes
        perm = None
        for attr in node.attributes:
            if attr.name == "perm":
                perm = list(attr.value.ints)
                break

        # Default permutation is reversed order
        if perm is None:
            perm = list(reversed(range(rank)))

        # Validate permutation
        if len(perm) != rank:
            return _common.InferenceResult(
                failure=f"Permutation length {len(perm)} does not match input rank {rank}."
            )

        if sorted(perm) != list(range(rank)):
            return _common.InferenceResult(
                failure=f"Invalid permutation {perm}. Must be a permutation of [0, 1, ..., {rank - 1}]."
            )

        # Apply permutation to create output shape
        output_shape = ir.Shape([0] * rank)
        for i, axis in enumerate(perm):
            # Handle negative axis
            if axis < 0:
                axis += rank

            if axis < 0 or axis >= rank:
                return _common.InferenceResult(
                    failure=f"Permutation axis {axis} is out of bounds for rank {rank}."
                )

            # Copy dimension from input to output according to permutation
            input_dim_expr = _common.get_expr(input_shape, axis)
            _common.set_expr(output_shape, i, input_dim_expr)

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
        )
