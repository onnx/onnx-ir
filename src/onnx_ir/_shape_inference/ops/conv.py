# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_conv_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Conv op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    if not all(isinstance(dim, int) for dim in input_shape):
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Symbolic dimensions not supported for Conv yet.")

    kernel_shape = node.attributes.get_ints("kernel_shape")
    if not kernel_shape and len(node.inputs) > 1:
        weight_shape = shape_env.get_shape(node.inputs[1])
        if weight_shape is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, "Weight shape not available.")
        kernel_shape = list(weight_shape)[2:] # Assuming NCHW format

    strides = node.attributes.get_ints("strides", [1] * (len(input_shape) - 2))
    pads = node.attributes.get_ints("pads", [0] * 2 * (len(input_shape) - 2))
    dilations = node.attributes.get_ints("dilations", [1] * (len(input_shape) - 2))
    group = node.attributes.get_int("group", 1)

    output_shape = list(input_shape)
    output_shape[1] = list(shape_env.get_shape(node.inputs[1]))[0] # Output channels

    for i in range(len(kernel_shape)):
        effective_kernel_size = dilations[i] * (kernel_shape[i] - 1) + 1
        padded_input_size = input_shape[i + 2] + pads[i] + pads[i + len(kernel_shape)]
        output_shape[i + 2] = (padded_input_size - effective_kernel_size) // strides[i] + 1

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)