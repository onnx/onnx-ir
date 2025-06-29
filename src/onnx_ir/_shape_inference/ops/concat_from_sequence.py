# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_concat_from_sequence_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ConcatFromSequence op."""
    # ConcatFromSequence concatenates a sequence of tensors along an axis.
    # The output shape will have the specified axis dimension as unknown (None)
    # and other dimensions from the first element of the sequence.
    # Assuming the input is a sequence of tensors, and we only care about the element type and shape.
    # The actual sequence length is not known at shape inference time.

    # Get the element type and shape from the sequence input
    # For now, we assume the sequence contains TensorType elements.
    # TODO: Handle actual sequence type and its element type more robustly.
    input_type = shape_env.get_dtype(node.inputs[0]) # For sequence, get_dtype returns the element type
    if not isinstance(node.inputs[0].type, ir.SequenceType):
        return InferenceResult(InferenceStatus.FAILURE, "Input to ConcatFromSequence must be a SequenceType.")
    
    elem_type = node.inputs[0].type.elem_type
    if not isinstance(elem_type, ir.TensorType):
        return InferenceResult(InferenceStatus.FAILURE, "Sequence elements must be TensorType for ConcatFromSequence.")

    input_elem_shape = list(elem_type.shape)
    axis = node.attributes.get_int("axis")
    new_axis = node.attributes.get_int("new_axis", 0)

    output_shape = list(input_elem_shape)
    if new_axis:
        output_shape.insert(axis, None) # Insert a None for the new axis
    else:
        output_shape[axis] = None # The concatenated dimension becomes unknown

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), elem_type.dtype)
    return InferenceResult(InferenceStatus.SUCCESS)