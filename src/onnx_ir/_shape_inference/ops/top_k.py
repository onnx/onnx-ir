# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_top_k_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for TopK op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    k_tensor = node.inputs[1]
    axis = node.attributes.get_int("axis", -1)

    if k_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "K input must be a constant for TopK op.")
    k = k_tensor.const_value.numpy().item()

    # Handle negative axis
    if axis < 0:
        axis += len(input_shape)

    output_shape = list(input_shape)
    output_shape[axis] = k

    # Output 0 (Values)
    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)

    # Output 1 (Indices)
    shape_env.set_shape_and_type(node.outputs[1], ir.Shape(output_shape), ir.DataType.INT64) # Indices are always INT64

    return InferenceResult(InferenceStatus.SUCCESS)