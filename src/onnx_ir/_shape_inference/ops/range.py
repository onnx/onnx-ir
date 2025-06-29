# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_range_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Range op."""
    start_val = node.inputs[0].const_value
    limit_val = node.inputs[1].const_value
    delta_val = node.inputs[2].const_value

    if start_val is None or limit_val is None or delta_val is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Start, limit, and delta must be constant for Range op.")

    start = start_val.numpy().item()
    limit = limit_val.numpy().item()
    delta = delta_val.numpy().item()

    output_len = math.ceil((limit - start) / delta)
    if output_len < 0:
        output_len = 0

    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input dtype not available.")

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape([output_len]), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)