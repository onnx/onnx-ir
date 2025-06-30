"""Constant operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ConstantInferrer(_common.NodeInferrer):
    """Inferrer for Constant operations."""

    def __init__(self) -> None:
        """Initialize the Constant inferrer."""
        super().__init__("Constant", opsets=range(sys.maxsize))

    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Constant operations."""
        # Get the value attribute
        value_attr = node.attributes.get_tensor("value")
        if value_attr is None:
            return _common.InferenceResult(failure="Constant operation requires value attribute.")

        # Create shape from the tensor dimensions
        output_shape = ir.Shape(list(value_attr.shape))
        
        # Get the data type from the tensor
        output_type = value_attr.dtype

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )