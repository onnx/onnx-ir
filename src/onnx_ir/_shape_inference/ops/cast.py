# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_cast_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Cast op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    if input_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")

    to_dtype = _enums.DataType(node.attributes.get_int("to"))
    shape_env.set_shape_and_type(node.outputs[0], input_shape, to_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)
