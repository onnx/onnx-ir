# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_quantize_linear_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for QuantizeLinear op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    if input_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")

    # Output shape is the same as the input shape
    # Output type is the same as the zero_point input type (if provided), otherwise default to UINT8
    if len(node.inputs) > 2 and shape_env.get_dtype(node.inputs[2]) is not None:
        output_dtype = shape_env.get_dtype(node.inputs[2])
    else:
        output_dtype = ir.DataType.UINT8 # Default output type
    shape_env.set_shape_and_type(node.outputs[0], input_shape, output_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)