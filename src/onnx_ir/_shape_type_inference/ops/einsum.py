"""Einsum operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys
import re
from typing import Dict, List, Set

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _parse_einsum_equation(equation: str) -> tuple[List[str], str]:
    """Parse einsum equation into input subscripts and output subscript."""
    if "->" in equation:
        input_part, output_part = equation.split("->")
        input_subscripts = [s.strip() for s in input_part.split(",")]
        output_subscript = output_part.strip()
    else:
        # Implicit output - all indices that appear exactly once
        input_subscripts = [s.strip() for s in equation.split(",")]

        # Count occurrences of each index
        index_count: Dict[str, int] = {}
        for subscript in input_subscripts:
            for char in subscript:
                if char.isalpha():
                    index_count[char] = index_count.get(char, 0) + 1

        # Output contains indices that appear exactly once, in alphabetical order
        output_indices = sorted([idx for idx, count in index_count.items() if count == 1])
        output_subscript = "".join(output_indices)

    return input_subscripts, output_subscript


def _validate_einsum_inputs(
    input_subscripts: List[str], input_shapes: List[ir.Shape]
) -> Dict[str, int | None]:
    """Validate einsum inputs and return dimension mapping."""
    if len(input_subscripts) != len(input_shapes):
        raise ValueError(
            f"Number of input subscripts {len(input_subscripts)} "
            f"doesn't match number of inputs {len(input_shapes)}"
        )

    # Map each index character to its dimension size
    index_dims: Dict[str, int | None] = {}

    for subscript, shape in zip(input_subscripts, input_shapes):
        if len(subscript) != len(shape):
            raise ValueError(
                f"Subscript '{subscript}' length {len(subscript)} "
                f"doesn't match shape rank {len(shape)}"
            )

        for i, char in enumerate(subscript):
            if char.isalpha():
                dim_size = shape.dims[i]
                if isinstance(dim_size, int):
                    if char in index_dims:
                        if index_dims[char] is not None and index_dims[char] != dim_size:
                            raise ValueError(
                                f"Index '{char}' has inconsistent dimensions: "
                                f"{index_dims[char]} vs {dim_size}"
                            )
                    else:
                        index_dims[char] = dim_size
                else:
                    # Symbolic dimension
                    if char not in index_dims:
                        index_dims[char] = None

    return index_dims


class EinsumInferrer(_common.NodeInferrer):
    """Inferrer for Einsum operations."""

    def __init__(self) -> None:
        """Initialize the Einsum inferrer."""
        super().__init__("Einsum", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least one input
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Einsum operations."""
        # Get the equation attribute
        equation = node.attributes.get_string("equation")
        if equation is None:
            return _common.InferenceResult(
                failure="Einsum operation requires equation attribute."
            )

        # Check that all inputs have known shapes
        input_shapes = []
        for i, inp in enumerate(node.inputs):
            if inp is None:
                return _common.InferenceResult(failure=f"Einsum input {i} cannot be None.")
            if inp.shape is None:
                return _common.InferenceResult(failure=f"Einsum input {i} shape is not known.")
            input_shapes.append(inp.shape)

        try:
            # Parse the einsum equation
            input_subscripts, output_subscript = _parse_einsum_equation(equation)

            # Validate inputs and get dimension mapping
            index_dims = _validate_einsum_inputs(input_subscripts, input_shapes)

            # Compute output shape
            output_dims = []
            for char in output_subscript:
                if char.isalpha():
                    if char in index_dims:
                        output_dims.append(index_dims[char])
                    else:
                        # This shouldn't happen with valid einsum equations
                        output_dims.append(None)

            output_shape = ir.Shape(output_dims)

        except ValueError as e:
            return _common.InferenceResult(failure=f"Einsum equation error: {str(e)}")
        except Exception as e:
            return _common.InferenceResult(failure=f"Einsum parsing failed: {str(e)}")

        # Output type is the same as the first input type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
