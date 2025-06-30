"""Einsum operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class EinsumInferrer(_common.NodeInferrer):
    """Inferrer for Einsum operations."""

    def __init__(self) -> None:
        """Initialize the Einsum inferrer."""
        super().__init__("Einsum", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least one input
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Einsum operations."""
        # Get the equation attribute
        equation = node.attributes.get_string("equation")
        if equation is None:
            return _common.InferenceResult(failure="Einsum operation requires equation attribute.")

        # For symbolic inference, parsing the full einsum equation is complex
        # We'll do a simplified inference based on common patterns
        
        # Check that all inputs have known shapes
        for i, inp in enumerate(node.inputs):
            if inp is None:
                return _common.InferenceResult(failure=f"Einsum input {i} cannot be None.")
            if inp.shape is None:
                return _common.InferenceResult(failure=f"Einsum input {i} shape is not known.")

        # For now, we'll return an unknown shape since full einsum parsing is complex
        # A complete implementation would parse the equation and compute the output shape
        output_shape = ir.Shape([None])  # Unknown shape
        
        # Output type is the same as the first input type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )