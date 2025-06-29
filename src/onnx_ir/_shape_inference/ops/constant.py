# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_constant_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Constant op."""
    value = node.attributes.get_tensor("value")
    shape_env.set_shape_and_type(node.outputs[0], value.shape, value.dtype)
    node.outputs[0].const_value = value
    return InferenceResult(InferenceStatus.SUCCESS)
