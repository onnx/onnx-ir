"""Tile operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class TileInferrer(_common.NodeInferrer):
    """Inferrer for Tile operations."""

    def __init__(self) -> None:
        """Initialize the Tile inferrer."""
        super().__init__("Tile", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Tile operations."""
        assert node.inputs[0] is not None
        assert node.inputs[1] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Tile input shape is not known.")

        # Get repeats from the second input
        repeats_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if repeats_tensor is not None:
            repeats = repeats_tensor.numpy().tolist()
            if not isinstance(repeats, list):
                repeats = [repeats]
            
            rank = len(input_shape)
            if len(repeats) != rank:
                return _common.InferenceResult(
                    failure=f"Repeats length {len(repeats)} does not match input rank {rank}."
                )
            
            # Calculate output dimensions by multiplying input dims with repeats
            output_dims = []
            for i, repeat in enumerate(repeats):
                input_dim = input_shape.dims[i]
                if isinstance(input_dim, int):
                    output_dim = input_dim * repeat
                    output_dims.append(output_dim)
                else:
                    # Symbolic dimension
                    import sympy
                    input_expr = _common.get_expr(input_shape, i)
                    output_expr = input_expr * sympy.Integer(repeat)
                    output_dims.append(output_expr)
            
            output_shape = ir.Shape(output_dims)
        else:
            # Repeats are not constant
            repeats_shape = node.inputs[1].shape
            if repeats_shape is None or len(repeats_shape) != 1:
                return _common.InferenceResult(
                    failure="Tile repeats input must be a 1D tensor with known shape."
                )
            
            repeats_length = repeats_shape[0]
            if not isinstance(repeats_length, int):
                return _common.InferenceResult(
                    failure="Tile repeats length must be statically known."
                )
            
            if repeats_length != len(input_shape):
                return _common.InferenceResult(
                    failure=f"Repeats length {repeats_length} does not match input rank {len(input_shape)}."
                )
            
            # Create output shape with unknown dimensions
            output_shape = ir.Shape([None] * len(input_shape))

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )