# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_group_norm_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen group_norm op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    # Output 0 (normalized input) has the same shape and type as input 0
    shape_env.set_shape_and_type(node.outputs[0], input_shape, input_dtype)

    # Outputs 1 and 2 (mean and variance) are 1D tensors
    # Their size depends on the number of groups and batch size
    # For now, we'll set their shape to [None] (dynamic)
    for i in range(1, min(len(node.outputs), 3)): # Up to 3 outputs
        shape_env.set_shape_and_type(node.outputs[i], ir.Shape([None]), input_dtype)

    return InferenceResult(InferenceStatus.SUCCESS)