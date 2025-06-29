# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference import utils
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_symbolic_compute_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for symbolic compute ops (Equal, Floor, Max, Min, Where, Neg)."""
    input_shapes = []
    input_dtype = None
    for input_value in node.inputs:
        shape = shape_env.get_shape(input_value)
        dtype = shape_env.get_dtype(input_value)
        if shape is None or dtype is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Input shape or dtype not available for {input_value.name}")
        input_shapes.append(shape)
        if input_dtype is None:
            input_dtype = dtype
        elif input_dtype != dtype:
            # For Equal, output is bool, so dtypes can be different
            if node.op_type == "Equal":
                pass
            else:
                return InferenceResult(InferenceStatus.FAILURE, "Incompatible input dtypes for symbolic compute op.")

    output_shape = utils.broadcast_shapes(input_shapes)
    
    # For Equal, output dtype is bool
    if node.op_type == "Equal":
        output_dtype = ir.DataType.BOOL
    else:
        output_dtype = input_dtype

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), output_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)