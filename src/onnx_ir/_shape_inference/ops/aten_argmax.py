# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_argmax_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen argmax op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    if input_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")
    input_shape = list(input_shape)
    
    # If dim is not provided, argmax flattens the input
    if len(node.inputs) < 2 or node.inputs[1].const_value is None:
        output_shape = []
    else:
        dim = node.inputs[1].const_value.numpy().item()
        keepdim = node.inputs[2].const_value.numpy().item()

        # Handle negative dimension
        if dim < 0:
            dim += len(input_shape)

        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), _enums.DataType.INT64)
    return InferenceResult(InferenceStatus.SUCCESS)