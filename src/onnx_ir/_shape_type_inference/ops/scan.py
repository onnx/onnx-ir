"""Scan operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ScanInferrer(_common.NodeInferrer):
    """Inferrer for Scan operations."""

    def __init__(self) -> None:
        """Initialize the Scan inferrer."""
        super().__init__("Scan", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least one input
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Scan operations."""
        # Get scan attributes
        num_scan_inputs = node.attributes.get_int("num_scan_inputs")
        scan_input_axes = node.attributes.get_ints("scan_input_axes")
        scan_input_directions = node.attributes.get_ints("scan_input_directions")
        scan_output_axes = node.attributes.get_ints("scan_output_axes")
        scan_output_directions = node.attributes.get_ints("scan_output_directions")

        if num_scan_inputs is None:
            return _common.InferenceResult(failure="Scan requires num_scan_inputs attribute.")

        total_inputs = len(node.inputs)
        num_state_inputs = total_inputs - num_scan_inputs

        # Validate inputs
        for i, inp in enumerate(node.inputs):
            if inp is None:
                return _common.InferenceResult(failure=f"Scan input {i} cannot be None.")
            if inp.shape is None:
                return _common.InferenceResult(failure=f"Scan input {i} shape is not known.")

        # For Scan operations, we need to analyze the body graph
        # Since we don't have access to subgraph analysis here, we'll make approximations

        num_outputs = len(node.outputs)
        num_state_outputs = num_outputs - num_scan_inputs  # Approximate

        output_values = []

        # State outputs (final states) - same shape as initial state inputs
        for i in range(num_state_outputs):
            if i < num_state_inputs:
                state_input = node.inputs[i]
                output_shape = state_input.shape
                output_type = state_input.type
            else:
                output_shape = ir.Shape([None])
                output_type = None
            output_values.append(ir.Value(shape=output_shape, type=output_type))

        # Scan outputs - accumulated results over the scan dimension
        for i in range(num_scan_inputs):
            scan_input_idx = num_state_inputs + i
            if scan_input_idx < total_inputs:
                scan_input = node.inputs[scan_input_idx]
                scan_input_shape = scan_input.shape

                # The scan output typically has the same shape as the scan input
                # but may have modifications based on the body graph
                output_shape = scan_input_shape
                output_type = scan_input.type
            else:
                output_shape = ir.Shape([None])
                output_type = None
            output_values.append(ir.Value(shape=output_shape, type=output_type))

        return _common.InferenceResult(values=output_values)
