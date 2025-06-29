# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_non_max_suppression_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for NonMaxSuppression op."""
    # Output is a 2D tensor with shape [num_selected_indices, 3]
    # num_selected_indices is dynamic, so we set it to None
    output_shape = [None, 3]

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), _enums.DataType.INT64)
    return InferenceResult(InferenceStatus.SUCCESS)