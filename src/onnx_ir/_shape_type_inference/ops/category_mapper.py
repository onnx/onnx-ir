"""CategoryMapper operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class CategoryMapperInferrer(_common.NodeInferrer):
    """Inferrer for CategoryMapper operations."""

    def __init__(self) -> None:
        """Initialize the CategoryMapper inferrer."""
        super().__init__("CategoryMapper", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for CategoryMapper operations."""
        assert node.inputs[0] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="CategoryMapper input shape is not known.")

        # CategoryMapper preserves the input shape but may change the type
        output_shape = input_shape

        # Determine output type based on the mapping attributes
        cats_int64s = node.attributes.get_ints("cats_int64s")
        cats_strings = node.attributes.get_strings("cats_strings")

        if cats_int64s is not None:
            # Mapping to integers
            output_type = ir.TensorType.INT64
        elif cats_strings is not None:
            # Mapping to strings
            output_type = ir.TensorType.STRING
        else:
            # No mapping specified, preserve input type
            output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
