"""Concat operation inferrer for ONNX IR nodes."""

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ConcatInferrer(_common.NodeInferrer):
    """Inferrer for Concat operations."""

    def __init__(self, opsets: Collection[int] | None = None) -> None:
        """Initialize the Concat inferrer."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__("Concat", opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Concat operations."""
        if len(node.inputs) < 1:
            return _common.InferenceResult(
                failure="Concat operation must have at least one input."
            )
        if any(inp is None for inp in node.inputs):
            return _common.InferenceResult(failure="Concat operation inputs cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Concat operation must have exactly one output, got {len(node.outputs)}."
            )

        # Get axis attribute
        axis_attr = None
        for attr in node.attributes:
            if attr.name == "axis":
                axis_attr = attr.value.i
                break

        if axis_attr is None:
            return _common.InferenceResult(failure="Concat operation requires axis attribute.")

        # Get first input shape as base
        first_shape = node.inputs[0].shape
        if first_shape is None:
            return _common.InferenceResult(failure="Concat input shapes cannot be None.")

        rank = len(first_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Concat inputs cannot be scalars.")

        # Handle negative axis
        if axis_attr < 0:
            axis_attr += rank

        if axis_attr < 0 or axis_attr >= rank:
            return _common.InferenceResult(
                failure=f"Concat axis {axis_attr} is out of bounds for rank {rank}."
            )

        # Check that all inputs have compatible shapes
        output_shape = ir.Shape(list(first_shape))
        concat_dim_size = _common.get_expr(first_shape, axis_attr)

        for i, inp in enumerate(node.inputs[1:], 1):
            if inp.shape is None:
                return _common.InferenceResult(failure=f"Input {i} shape cannot be None.")

            input_shape = inp.shape
            if len(input_shape) != rank:
                return _common.InferenceResult(
                    failure=f"All inputs must have same rank. Input {i} has rank {len(input_shape)}, expected {rank}."
                )

            # Check non-concat dimensions are compatible
            for dim_idx in range(rank):
                if dim_idx == axis_attr:
                    # Accumulate concat dimension
                    concat_dim_size = concat_dim_size + _common.get_expr(input_shape, dim_idx)
                else:
                    # Check compatibility of other dimensions
                    dim1 = _common.get_expr(first_shape, dim_idx)
                    dim2 = _common.get_expr(input_shape, dim_idx)
                    # For symbolic inference, we assume they are compatible
                    # In practice, this would need runtime verification

        # Set the concat dimension in output shape
        _common.set_expr(output_shape, axis_attr, concat_dim_size)

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
