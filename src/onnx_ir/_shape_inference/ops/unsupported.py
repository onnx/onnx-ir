# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_unsupported_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for unsupported ops."""
    return InferenceResult(InferenceStatus.UNSUPPORTED, f"Shape inference not implemented for {node.op_type} op.")