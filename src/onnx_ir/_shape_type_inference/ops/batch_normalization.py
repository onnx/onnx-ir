"""BatchNormalization operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class BatchNormalizationInferrer(_common.NodeInferrer):
    """Inferrer for BatchNormalization operations."""

    def __init__(self) -> None:
        """Initialize the BatchNormalization inferrer."""
        super().__init__("BatchNormalization", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(5)  # X, scale, B, input_mean, input_var
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for BatchNormalization operations."""
        assert node.inputs[0] is not None  # X
        assert node.inputs[1] is not None  # scale
        assert node.inputs[2] is not None  # B
        assert node.inputs[3] is not None  # input_mean
        assert node.inputs[4] is not None  # input_var

        input_shape = node.inputs[0].shape
        scale_shape = node.inputs[1].shape
        bias_shape = node.inputs[2].shape
        mean_shape = node.inputs[3].shape
        var_shape = node.inputs[4].shape

        if input_shape is None:
            return _common.InferenceResult(
                failure="BatchNormalization input shape is not known."
            )
        if scale_shape is None:
            return _common.InferenceResult(
                failure="BatchNormalization scale shape is not known."
            )
        if bias_shape is None:
            return _common.InferenceResult(
                failure="BatchNormalization bias shape is not known."
            )
        if mean_shape is None:
            return _common.InferenceResult(
                failure="BatchNormalization mean shape is not known."
            )
        if var_shape is None:
            return _common.InferenceResult(
                failure="BatchNormalization var shape is not known."
            )

        input_rank = len(input_shape)
        if input_rank < 2:
            return _common.InferenceResult(
                failure="BatchNormalization input must have at least 2 dimensions."
            )

        # Scale, bias, mean, var should all be 1D with size equal to the channel dimension
        channel_dim = input_shape.dims[1]  # Assume NCHW format

        for param_name, param_shape in [
            ("scale", scale_shape),
            ("bias", bias_shape),
            ("mean", mean_shape),
            ("var", var_shape),
        ]:
            if len(param_shape) != 1:
                return _common.InferenceResult(
                    failure=f"BatchNormalization {param_name} must be 1D."
                )
            # For symbolic inference, we assume the parameter shapes match the channel dimension

        # Determine number of outputs
        num_outputs = len(node.outputs)

        output_values = []

        # First output: normalized data (same shape as input)
        output_values.append(ir.Value(shape=input_shape, type=node.inputs[0].type))

        if num_outputs >= 2:
            # Second output: running mean (same shape as mean parameter)
            output_values.append(ir.Value(shape=mean_shape, type=node.inputs[3].type))

        if num_outputs >= 3:
            # Third output: running var (same shape as var parameter)
            output_values.append(ir.Value(shape=var_shape, type=node.inputs[4].type))

        return _common.InferenceResult(values=output_values)
