"""Size operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class SizeInferrer(_common.NodeInferrer):
    """Inferrer for Size operations."""

    def __init__(self) -> None:
        """Initialize the Size inferrer."""
        super().__init__("Size", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Size operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Size input shape is not known.")

        # Size always outputs a scalar (0-D tensor) containing the total number of elements
        output_shape = ir.Shape([])  # Scalar
        
        # Size always outputs INT64
        output_type = ir.TensorType.INT64

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )