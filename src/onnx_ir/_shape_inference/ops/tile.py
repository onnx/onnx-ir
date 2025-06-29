# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_tile_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Tile op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    repeats_tensor = node.inputs[1]

    if repeats_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Repeats input must be a constant for Tile op.")
    
    repeats = repeats_tensor.const_value.numpy().tolist()

    if len(input_shape) != len(repeats):
        return InferenceResult(InferenceStatus.FAILURE, "Input rank and repeats rank must be the same for Tile op.")

    output_shape = []
    for i, dim in enumerate(input_shape):
        if dim is None:
            output_shape.append(None) # Cannot infer if input dim is symbolic
        else:
            output_shape.append(dim * repeats[i])

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)