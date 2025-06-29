# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_one_hot_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for OneHot op."""
    indices_shape = shape_env.get_shape(node.inputs[0])
    if indices_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Indices shape not available.")
    indices_shape = list(indices_shape)

    depth_tensor = node.inputs[1]
    axis = node.attributes.get_int("axis", -1)

    if depth_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Depth input must be a constant for OneHot.")
    depth = depth_tensor.const_value.numpy().item()

    # Handle negative axis
    if axis < 0:
        axis += len(indices_shape) + 1

    output_shape = indices_shape[:axis] + [depth] + indices_shape[axis:]

    output_dtype = shape_env.get_dtype(node.inputs[2]) # Output type is from the 'on_value' or 'off_value' input
    if output_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Output dtype (on_value/off_value) not available.")

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), output_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)