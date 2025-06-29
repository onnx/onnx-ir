# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_non_zero_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for NonZero op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    input_rank = len(input_shape)
    # Output is a 2D tensor with shape [rank, num_non_zero_elements]
    # num_non_zero_elements is dynamic, so we set it to None
    output_shape = [input_rank, None]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)