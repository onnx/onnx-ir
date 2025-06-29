# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_sequence_insert_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for SequenceInsert op."""
    input_type = node.inputs[0].type
    if not isinstance(input_type, ir.SequenceType):
        return InferenceResult(InferenceStatus.FAILURE, "Input to SequenceInsert must be a SequenceType.")
    
    # For SequenceType, we set the type directly on the output value
    node.outputs[0].type = input_type
    return InferenceResult(InferenceStatus.SUCCESS)