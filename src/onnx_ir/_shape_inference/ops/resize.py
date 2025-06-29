# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_resize_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Resize op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)
    
    # Opset 10 and below uses scales attribute
    # Opset 11 and above uses scales or sizes input
    scales = None
    sizes = None

    if node.attributes.get_floats("scales") is not None:
        scales = node.attributes.get_floats("scales")
    elif len(node.inputs) > 2 and node.inputs[2].const_value is not None:
        scales = node.inputs[2].const_value.numpy().tolist()
    elif len(node.inputs) > 3 and node.inputs[3].const_value is not None:
        sizes = node.inputs[3].const_value.numpy().tolist()

    output_shape = []
    if sizes is not None:
        output_shape = [int(s) for s in sizes]
    elif scales is not None:
        for i, dim in enumerate(input_shape):
            if dim is None:
                output_shape.append(None) # Cannot infer if input dim is symbolic
            else:
                output_shape.append(int(dim * scales[i]))
    else:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Scales or Sizes must be provided for Resize op.")

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)