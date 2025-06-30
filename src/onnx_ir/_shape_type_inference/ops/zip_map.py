"""ZipMap operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ZipMapInferrer(_common.NodeInferrer):
    """Inferrer for ZipMap operations."""

    def __init__(self) -> None:
        """Initialize the ZipMap inferrer."""
        super().__init__("ZipMap", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ZipMap operations."""
        assert node.inputs[0] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="ZipMap input shape is not known.")

        # Input should be 2D: [N, C] where N is batch size, C is number of classes
        if len(input_shape) != 2:
            return _common.InferenceResult(failure="ZipMap input must be 2D.")

        batch_size = input_shape.dims[0]

        # ZipMap converts a tensor to a sequence of maps
        # Output shape is [N] where each element is a map
        output_shape = ir.Shape([batch_size])

        # Output type is a sequence of maps
        # For now, we'll approximate this with the input type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
