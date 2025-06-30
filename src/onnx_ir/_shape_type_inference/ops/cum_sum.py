"""CumSum operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class CumSumInferrer(_common.NodeInferrer):
    """Inferrer for CumSum operations."""

    def __init__(self) -> None:
        """Initialize the CumSum inferrer."""
        super().__init__("CumSum", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for CumSum operations."""
        assert node.inputs[0] is not None  # input
        assert node.inputs[1] is not None  # axis
        
        input_shape = node.inputs[0].shape
        axis_shape = node.inputs[1].shape
        
        if input_shape is None:
            return _common.InferenceResult(failure="CumSum input shape is not known.")
        if axis_shape is None:
            return _common.InferenceResult(failure="CumSum axis shape is not known.")

        # Axis should be a scalar
        if len(axis_shape) != 0:
            return _common.InferenceResult(failure="CumSum axis input must be a scalar.")

        # CumSum preserves the input shape
        output_shape = input_shape
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )