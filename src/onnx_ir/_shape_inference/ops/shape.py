# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_shape_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Shape op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    if input_shape is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape not available.")

    output_shape = [len(input_shape)]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), _enums.DataType.INT64)

    if input_shape.is_static():
        node.outputs[0].const_value = ir.Tensor(np.array(input_shape.numpy(), dtype=np.int64))

    return InferenceResult(InferenceStatus.SUCCESS)