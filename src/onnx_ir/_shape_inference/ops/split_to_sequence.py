# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_split_to_sequence_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for SplitToSequence op."""
    input_shape = shape_env.get_shape(node.inputs[0])
    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_shape is None or input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    # The output is a sequence of tensors. The element type of the sequence
    # is the same as the input tensor's type and shape.

    # Create a TensorType for the elements of the sequence
    elem_type = ir.TensorType(input_dtype, shape=input_shape)

    # Create a SequenceType for the output
    output_sequence_type = ir.SequenceType(elem_type)

    # For SequenceType, we set the type directly on the output value
    node.outputs[0].type = output_sequence_type
    return InferenceResult(InferenceStatus.SUCCESS)