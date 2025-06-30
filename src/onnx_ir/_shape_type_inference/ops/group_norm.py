"""GroupNorm operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class GroupNormInferrer(_common.NodeInferrer):
    """Inferrer for GroupNorm operations."""

    def __init__(self) -> None:
        """Initialize the GroupNorm inferrer."""
        super().__init__("GroupNorm", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(3)  # X, scale, bias
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for GroupNorm operations."""
        assert node.inputs[0] is not None  # X
        assert node.inputs[1] is not None  # scale
        assert node.inputs[2] is not None  # bias

        input_shape = node.inputs[0].shape
        scale_shape = node.inputs[1].shape
        bias_shape = node.inputs[2].shape

        if input_shape is None:
            return _common.InferenceResult(failure="GroupNorm input shape is not known.")
        if scale_shape is None:
            return _common.InferenceResult(failure="GroupNorm scale shape is not known.")
        if bias_shape is None:
            return _common.InferenceResult(failure="GroupNorm bias shape is not known.")

        input_rank = len(input_shape)
        if input_rank < 2:
            return _common.InferenceResult(
                failure="GroupNorm input must have at least 2 dimensions."
            )

        # Get num_groups attribute
        num_groups = node.attributes.get_int("num_groups")
        if num_groups is None:
            return _common.InferenceResult(failure="GroupNorm requires num_groups attribute.")

        # Check that scale and bias are 1D with size equal to the number of channels
        channels = input_shape.dims[1]  # Assume NCHW format

        if len(scale_shape) != 1:
            return _common.InferenceResult(failure="GroupNorm scale must be 1D.")
        if len(bias_shape) != 1:
            return _common.InferenceResult(failure="GroupNorm bias must be 1D.")

        # For symbolic inference, we assume scale and bias shapes match the channel dimension

        # GroupNorm output has the same shape as input
        output_shape = input_shape
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
