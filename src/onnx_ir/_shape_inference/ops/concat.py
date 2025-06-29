# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_concat_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Concat op."""
    axis = node.attributes.get_int("axis")
    input_shapes = []
    input_dtype = None
    for input_value in node.inputs:
        shape = shape_env.get_shape(input_value)
        dtype = shape_env.get_dtype(input_value)
        if shape is None or dtype is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Input shape or dtype not available for {input_value.name}")
        input_shapes.append(shape)
        if input_dtype is None:
            input_dtype = dtype
        elif input_dtype != dtype:
            return InferenceResult(InferenceStatus.FAILURE, "Incompatible input dtypes for Concat op.")

    output_shape = list(input_shapes[0])
    for i in range(1, len(input_shapes)):
        output_shape[axis] += input_shapes[i][axis]
    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)
