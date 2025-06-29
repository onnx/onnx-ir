# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_reshape_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Reshape op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    target_shape_tensor = node.inputs[1]
    if target_shape_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Target shape input must be a constant for Reshape op.")
    target_shape = list(target_shape_tensor.const_value.numpy())

    if -1 in target_shape:
        total_size = 1
        for dim in input_shape:
            total_size *= dim
        
        new_size = 1
        for dim in target_shape:
            if dim != -1:
                new_size *= dim
        
        target_shape[target_shape.index(-1)] = total_size // new_size

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(target_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)
