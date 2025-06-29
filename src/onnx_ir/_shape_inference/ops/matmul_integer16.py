# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_matmul_integer16_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for MatMulInteger16 op."""
    lhs_shape = shape_env.get_shape(node.inputs[0])
    rhs_shape = shape_env.get_shape(node.inputs[1])
    if lhs_shape is None or rhs_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")
    lhs_shape = list(lhs_shape)
    rhs_shape = list(rhs_shape)

    output_shape = ir.utils.broadcast_shapes([lhs_shape[:-2], rhs_shape[:-2]])
    output_shape.append(lhs_shape[-2])
    output_shape.append(rhs_shape[-1])

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), _enums.DataType.INT32)
    return InferenceResult(InferenceStatus.SUCCESS)