# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_dequantize_linear_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for DequantizeLinear op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    scale_dtype = shape_env.get_dtype(node.inputs[1])
    if input_shape is None or scale_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or scale dtype not available.")

    # Output shape is the same as the input shape
    # Output type is the same as the scale input type
    shape_env.set_shape_and_type(node.outputs[0], input_shape, scale_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)