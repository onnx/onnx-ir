# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference import utils
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_matmul_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for MatMul op."""
    lhs_shape = shape_env.get_shape(node.inputs[0])
    rhs_shape = shape_env.get_shape(node.inputs[1])
    lhs_dtype = shape_env.get_dtype(node.inputs[0])
    if lhs_shape is None or rhs_shape is None or lhs_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")
    lhs_shape = list(lhs_shape)
    rhs_shape = list(rhs_shape)

    output_shape = utils.broadcast_shapes([lhs_shape[:-2], rhs_shape[:-2]])
    output_shape.append(lhs_shape[-2])
    output_shape.append(rhs_shape[-1])

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), lhs_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)
