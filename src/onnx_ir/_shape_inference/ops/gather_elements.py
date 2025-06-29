# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_gather_elements_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for GatherElements op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    indices_shape = shape_env.get_shape(node.inputs[1])
    if input_shape is None or input_dtype is None or indices_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    # Output shape is the same as the indices shape
    shape_env.set_shape_and_type(node.outputs[0], indices_shape, input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)