# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_compress_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Compress op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    axis = node.attributes.get_int("axis", None)

    if axis is None:
        # If axis is not specified, the input is flattened before compression
        output_shape = [None]  # Output is 1D with unknown length
    else:
        # If axis is specified, the output shape is the same as input shape
        # except the dimension at the specified axis is unknown
        output_shape = list(input_shape)
        output_shape[axis] = None

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)