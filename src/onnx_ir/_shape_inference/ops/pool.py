# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_pool_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for pooling ops (AveragePool, MaxPool)."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    kernel_shape = node.attributes.get_ints("kernel_shape")
    strides = node.attributes.get_ints("strides", [1] * len(kernel_shape))
    pads = node.attributes.get_ints("pads", [0] * 2 * len(kernel_shape))
    ceil_mode = node.attributes.get_int("ceil_mode", 0)

    output_shape = list(input_shape)
    for i in range(len(kernel_shape)):
        effective_input_size = input_shape[i + 2] + pads[i] + pads[i + len(kernel_shape)]
        if ceil_mode:
            output_shape[i + 2] = int(math.ceil(float(effective_input_size - kernel_shape[i]) / strides[i])) + 1
        else:
            output_shape[i + 2] = int(math.floor(float(effective_input_size - kernel_shape[i]) / strides[i])) + 1

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)