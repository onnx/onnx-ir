# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_pool2d_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen 2D pooling ops."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    # For pooling ops, the spatial dimensions (last two) are typically reduced.
    # The exact calculation depends on kernel, stride, padding, etc.
    # For now, we'll set them to None (dynamic) for simplicity.
    output_shape = input_shape[:-2] + [None, None]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)

    # Some pooling ops (like max_pool2d_with_indices) have a second output for indices
    if len(node.outputs) > 1:
        shape_env.set_shape_and_type(node.outputs[1], ir.Shape(output_shape), ir.DataType.INT64) # Indices are typically INT64

    return InferenceResult(InferenceStatus.SUCCESS)