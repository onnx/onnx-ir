"""NonZero operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class NonZeroInferrer(_common.NodeInferrer):
    """Inferrer for NonZero operations."""

    def __init__(self) -> None:
        """Initialize the NonZero inferrer."""
        super().__init__("NonZero", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for NonZero operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="NonZero input shape is not known.")

        rank = len(input_shape)
        
        # NonZero output shape is [rank, num_nonzero_elements]
        # where num_nonzero_elements is unknown at compile time
        output_shape = ir.Shape([rank, None])
        
        # NonZero always outputs INT64
        output_type = ir.TensorType.INT64

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )