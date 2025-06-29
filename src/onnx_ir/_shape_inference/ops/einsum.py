# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_einsum_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Einsum op."""
    equation = node.attributes.get_string("equation").decode("utf-8")
    equation = equation.replace(" ", "")
    mid_index = equation.find("->")
    left_equation = equation[:mid_index] if mid_index != -1 else equation

    num_operands = 0
    num_ellipsis = 0
    num_ellipsis_indices = 0

    letter_to_dim = {}

    terms = left_equation.split(",")
    for term in terms:
        input_value = node.inputs[num_operands]
        shape = shape_env.get_shape(input_value)
        if shape is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Input shape not available for {input_value.name}")
        shape = list(shape)

        ellipsis_index = term.find("...")
        rank = len(shape)
        if ellipsis_index != -1:
            if num_ellipsis == 0:
                num_ellipsis_indices = rank - len(term) + 3
            num_ellipsis = num_ellipsis + 1
        for i in range(1, rank + 1):
            letter = term[-i]
            if letter != ".":
                dim = shape[-i]
                # Original code used sympy.Symbol check, here we just check if it's already a symbolic dim
                if letter not in letter_to_dim or not isinstance(dim, int):
                    letter_to_dim[letter] = dim
        num_operands = num_operands + 1

    new_output_shape_dims = []
    num_letter_occurrences = OrderedDict()
    if mid_index != -1:
        right_equation = equation[mid_index + 2 :]
        right_ellipsis_index = right_equation.find("...")
        if right_ellipsis_index != -1:
            first_input_shape = shape_env.get_shape(node.inputs[0])
            if first_input_shape is None:
                return InferenceResult(InferenceStatus.UNSUPPORTED, "First input shape not available for Einsum.")
            for i in range(num_ellipsis_indices):
                new_output_shape_dims.append(list(first_input_shape)[i])
        for c in right_equation:
            if c != ".":
                new_output_shape_dims.append(letter_to_dim[c])
    else:
        first_input_shape = shape_env.get_shape(node.inputs[0])
        if first_input_shape is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, "First input shape not available for Einsum.")
        for i in range(num_ellipsis_indices):
            new_output_shape_dims.append(list(first_input_shape)[i])
        for c in left_equation:
            if c not in {',', '.'}:
                if c in num_letter_occurrences:
                    num_letter_occurrences[c] = num_letter_occurrences[c] + 1
                else:
                    num_letter_occurrences[c] = 1
        for key, value in num_letter_occurrences.items():
            if value == 1:
                new_output_shape_dims.append(letter_to_dim[key])

    input_dtype = shape_env.get_dtype(node.inputs[0])
    if input_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input dtype not available for Einsum.")

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(new_output_shape_dims), input_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)