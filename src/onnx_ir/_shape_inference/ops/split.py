# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_split_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Split op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    axis = node.attributes.get_int("axis", 0)
    num_outputs = len(node.outputs)

    split_attr = node.attributes.get_ints("split")
    if split_attr:
        splits = split_attr
    else:
        # If 'split' attribute is not provided, assume equal splits
        if input_shape[axis] % num_outputs != 0:
            return InferenceResult(InferenceStatus.FAILURE, "Input dimension is not divisible by number of outputs for Split op.")
        split_size = input_shape[axis] // num_outputs
        splits = [split_size] * num_outputs

    current_idx = 0
    for i, output_value in enumerate(node.outputs):
        output_shape = list(input_shape)
        output_shape[axis] = splits[i]
        shape_env.set_shape_and_type(output_value, ir.Shape(output_shape), input_dtype)
        current_idx += splits[i]

    return InferenceResult(InferenceStatus.SUCCESS)