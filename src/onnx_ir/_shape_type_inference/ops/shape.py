"""Shape operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ShapeInferrer(_common.NodeInferrer):
    """Inferrer for Shape operations."""

    def __init__(self) -> None:
        """Initialize the Shape inferrer."""
        super().__init__("Shape", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Shape operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Shape input shape is not known.")

        # The output is a 1D tensor with length equal to the rank of the input
        rank = len(input_shape)
        output_shape = ir.Shape([rank])
        
        # Shape always outputs INT64
        output_type = ir.TensorType.INT64

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )