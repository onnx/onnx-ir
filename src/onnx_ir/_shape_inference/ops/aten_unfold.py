# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_unfold_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen unfold op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    dimension = node.inputs[1].const_value.numpy().item()
    size = node.inputs[2].const_value.numpy().item()
    step = node.inputs[3].const_value.numpy().item()

    # Handle negative dimension
    if dimension < 0:
        dimension += len(input_shape)

    output_shape = list(input_shape)
    output_shape[dimension] = (input_shape[dimension] - size) // step + 1
    output_shape.append(size)

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)