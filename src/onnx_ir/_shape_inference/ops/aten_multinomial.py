# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_aten_multinomial_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ATen multinomial op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    if input_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")
    input_shape = list(input_shape)

    num_samples_tensor = node.inputs[1]

    if num_samples_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "num_samples input must be a constant for ATen multinomial.")
    num_samples = num_samples_tensor.const_value.numpy().item()

    output_shape = input_shape[:-1] + [num_samples]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), _enums.DataType.INT64)
    return InferenceResult(InferenceStatus.SUCCESS)