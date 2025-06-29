# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_category_mapper_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for CategoryMapper op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    if input_dtype == _enums.DataType.STRING:
        output_dtype = _enums.DataType.INT64
    else:
        output_dtype = _enums.DataType.STRING

    shape_env.set_shape_and_type(node.outputs[0], input_shape, output_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)