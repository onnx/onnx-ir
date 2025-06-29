# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_batch_normalization_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for BatchNormalization op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    # Output 0 has the same shape and type as input 0
    shape_env.set_shape_and_type(node.outputs[0], input_shape, input_dtype)

    # Outputs 1, 2, 3, 4 have the same shape as input 1 (scale)
    # and the same type as input 0
    scale_shape = shape_env.get_shape(node.inputs[1])
    if scale_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Scale input shape not available.")

    for i in range(1, min(len(node.outputs), 5)):  # Up to 5 outputs
        shape_env.set_shape_and_type(node.outputs[i], scale_shape, input_dtype)

    return InferenceResult(InferenceStatus.SUCCESS)