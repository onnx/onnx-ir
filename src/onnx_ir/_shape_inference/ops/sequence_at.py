# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_sequence_at_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for SequenceAt op."""
    input_type = node.inputs[0].type
    if not isinstance(input_type, ir.SequenceType):
        return InferenceResult(InferenceStatus.FAILURE, "Input to SequenceAt must be a SequenceType.")
    
    elem_type = input_type.elem_type
    if not isinstance(elem_type, ir.TensorType):
        return InferenceResult(InferenceStatus.FAILURE, "Sequence elements must be TensorType for SequenceAt.")

    shape_env.set_shape_and_type(node.outputs[0], elem_type.shape, elem_type.dtype)
    return InferenceResult(InferenceStatus.SUCCESS)