"""Pad operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class PadInferrer(_common.NodeInferrer):
    """Inferrer for Pad operations."""

    def __init__(self) -> None:
        """Initialize the Pad inferrer."""
        super().__init__("Pad", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # At least input data
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Pad operations."""
        assert node.inputs[0] is not None
        
        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Pad input shape is not known.")

        rank = len(input_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Pad input cannot be a scalar.")

        # Get pads from attribute or input tensor
        pads_attr = node.attributes.get_ints("pads")
        pads_tensor = None
        if len(node.inputs) >= 2 and node.inputs[1] is not None:
            pads_tensor = ir.convenience.get_const_tensor(node.inputs[1])

        if pads_attr is not None:
            pads = pads_attr
        elif pads_tensor is not None:
            pads = pads_tensor.numpy().tolist()
            if not isinstance(pads, list):
                pads = [pads]
        else:
            return _common.InferenceResult(failure="Pad operation requires pads attribute or input.")

        # Pads should have length 2 * rank (begin_pads + end_pads)
        if len(pads) != 2 * rank:
            return _common.InferenceResult(
                failure=f"Pads length {len(pads)} should be {2 * rank} for rank {rank} input."
            )

        # Calculate output dimensions
        output_dims = []
        for i in range(rank):
            begin_pad = pads[i]
            end_pad = pads[i + rank]
            
            input_dim = input_shape.dims[i]
            if isinstance(input_dim, int):
                output_dim = input_dim + begin_pad + end_pad
                output_dims.append(output_dim)
            else:
                # Symbolic dimension - create symbolic expression for padding
                import sympy
                input_expr = _common.get_expr(input_shape, i)
                output_expr = input_expr + sympy.Integer(begin_pad) + sympy.Integer(end_pad)
                output_dims.append(output_expr)

        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )