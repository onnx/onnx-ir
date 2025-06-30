"""Resize operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ResizeInferrer(_common.NodeInferrer):
    """Inferrer for Resize operations."""

    def __init__(self) -> None:
        """Initialize the Resize inferrer."""
        super().__init__("Resize", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least input data
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Resize operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Resize input shape is not known.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Resize input cannot be a scalar.")

        # Check if we have roi, scales, or sizes inputs
        roi_tensor = None
        scales_tensor = None
        sizes_tensor = None
        
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            roi_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        if len(node.inputs) >= 3 and node.inputs[2] is not None:
            scales_tensor = ir.convenience.get_const_tensor(node.inputs[2])
        if len(node.inputs) >= 4 and node.inputs[3] is not None:
            sizes_tensor = ir.convenience.get_const_tensor(node.inputs[3])

        output_dims = list(input_shape.dims)

        if sizes_tensor is not None:
            # Use sizes to determine output shape
            sizes = sizes_tensor.numpy().tolist()
            if not isinstance(sizes, list):
                sizes = [sizes]
            
            if len(sizes) != rank:
                return _common.InferenceResult(
                    failure=f"Resize sizes length {len(sizes)} does not match input rank {rank}."
                )
            
            output_dims = [int(size) for size in sizes]
            
        elif scales_tensor is not None:
            # Use scales to determine output shape
            scales = scales_tensor.numpy().tolist()
            if not isinstance(scales, list):
                scales = [scales]
            
            if len(scales) != rank:
                return _common.InferenceResult(
                    failure=f"Resize scales length {len(scales)} does not match input rank {rank}."
                )
            
            # Calculate output dimensions by scaling input dimensions
            for i, scale in enumerate(scales):
                input_dim = input_shape.dims[i]
                if isinstance(input_dim, int):
                    output_dims[i] = int(input_dim * scale)
                else:
                    # Symbolic dimension - create symbolic expression
                    import sympy
                    input_expr = _common.get_expr(input_shape, i)
                    output_expr = sympy.floor(input_expr * sympy.Float(scale))
                    output_dims[i] = output_expr
        else:
            # No sizes or scales provided, output shape is unknown
            output_dims = [None] * rank

        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )