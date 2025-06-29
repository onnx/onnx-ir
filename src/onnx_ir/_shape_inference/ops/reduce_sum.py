# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_reduce_sum_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ReduceSum op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    keep_dims = node.attributes.get_int("keepdims", 1)

    axes = None
    if len(node.inputs) > 1 and node.inputs[1].const_value is not None:
        axes = node.inputs[1].const_value.numpy().tolist()
    elif node.attributes.get_ints("axes") is not None:
        axes = node.attributes.get_ints("axes")

    output_shape = []
    if axes is None: # Reduce over all dimensions
        if keep_dims:
            output_shape = [1] * len(input_shape)
        else:
            output_shape = []
    else:
        # Handle negative axes
        axes = [axis if axis >= 0 else len(input_shape) + axis for axis in axes]
        
        for i, dim in enumerate(input_shape):
            if i in axes:
                if keep_dims:
                    output_shape.append(1)
            else:
                output_shape.append(dim)

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)