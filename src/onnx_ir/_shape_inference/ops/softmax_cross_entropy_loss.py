# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_softmax_cross_entropy_loss_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for SoftmaxCrossEntropyLoss and related ops."""
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input dtype not available.")

    # The output is a scalar loss value
    shape_env.set_shape_and_type(node.outputs[0], ir.Shape([]), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)