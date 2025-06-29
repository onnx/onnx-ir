# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_relative_position_bias_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for RelativePositionBias op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    input_shape = list(input_shape)

    if len(input_shape) < 2:
        return InferenceResult(InferenceStatus.FAILURE, "Input tensor must have at least 2 dimensions.")
    
    num_heads = input_shape[1] # Assuming num_heads is the second dimension

    seq_len_tensor = node.inputs[1]
    real_seq_len_tensor = node.inputs[2]

    if seq_len_tensor.const_value is None or real_seq_len_tensor.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "seq_len and real_seq_len must be constant for RelativePositionBias.")

    seq_len = seq_len_tensor.const_value.numpy().item()
    real_seq_len = real_seq_len_tensor.const_value.numpy().item()

    output_shape = [1, num_heads, seq_len, real_seq_len]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)