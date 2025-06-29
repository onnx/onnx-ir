# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_slice_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Slice op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    starts_value = node.inputs[1].const_value
    ends_value = node.inputs[2].const_value
    if starts_value is None or ends_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Starts and Ends must be constant for Slice op.")
    starts = starts_value.numpy()
    ends = ends_value.numpy()

    axes = None
    if len(node.inputs) > 3 and node.inputs[3].const_value is not None:
        axes = node.inputs[3].const_value.numpy()
    else:
        axes = list(range(len(input_shape)))

    steps = None
    if len(node.inputs) > 4 and node.inputs[4].const_value is not None:
        steps = node.inputs[4].const_value.numpy()
    else:
        steps = [1] * len(input_shape)

    output_shape = list(input_shape)
    for i, axis in enumerate(axes):
        start = starts[i]
        end = ends[i]
        step = steps[i]

        if start < 0:
            start += input_shape[axis]
        if end < 0:
            end += input_shape[axis]
        
        output_shape[axis] = (end - start + step - 1) // step

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)