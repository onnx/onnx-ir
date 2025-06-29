# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference import utils
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_expand_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Expand op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    # The second input is the target shape
    if node.inputs[1].const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Expand target shape must be a constant.")
    
    target_shape = list(node.inputs[1].const_value.numpy())

    output_shape = utils.broadcast_shapes([input_shape, target_shape])

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)