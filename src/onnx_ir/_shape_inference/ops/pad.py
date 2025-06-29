# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_pad_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Pad op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)
    
    # Pads can be an attribute (opset <= 10) or an input (opset >= 11)
    pads = None
    if node.attributes.get_ints("pads") is not None:
        pads = node.attributes.get_ints("pads")
    elif len(node.inputs) > 1 and node.inputs[1].const_value is not None:
        pads = node.inputs[1].const_value.numpy().tolist()
    
    if pads is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Pads must be a constant for Pad op.")

    if len(pads) != 2 * len(input_shape):
        return InferenceResult(InferenceStatus.FAILURE, "Pads attribute/input must have 2 * rank elements.")

    output_shape = list(input_shape)
    for i in range(len(input_shape)):
        output_shape[i] = input_shape[i] + pads[i] + pads[i + len(input_shape)]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)