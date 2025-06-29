# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_diagonal_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen diagonal op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    dim1_val = node.inputs[2].const_value
    dim2_val = node.inputs[3].const_value
    if dim1_val is None or dim2_val is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "dim1 and dim2 must be constant for ATen diagonal.")
    dim1 = dim1_val.numpy().item()
    dim2 = dim2_val.numpy().item()

    # Handle negative dimensions
    if dim1 < 0:
        dim1 += len(input_shape)
    if dim2 < 0:
        dim2 += len(input_shape)

    output_shape = []
    for i, dim in enumerate(input_shape):
        if i != dim1 and i != dim2:
            output_shape.append(dim)
    
    # The diagonal dimension is dynamic
    output_shape.append(None)

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)