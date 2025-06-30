"""Concat operation inferrer for ONNX IR nodes."""

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ConcatInferrer(_common.NodeInferrer):
    """Inferrer for Concat operations."""

    def __init__(self) -> None:
        """Initialize the Concat inferrer."""
        super().__init__("Concat", opsets=range(sys.maxsize))

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
        axis = node.attributes.get_int("axis")
        if axis is None:
            return _common.InferenceResult(failure="Concat operation requires axis attribute.")

        # Get first input shape as base
        first_shape = node.inputs[0].shape
        if first_shape is None:
            return _common.InferenceResult(failure="Concat input shapes cannot be None.")
        first_type = node.inputs[0].type

        rank = len(first_shape)
        if rank == 0:
            return _common.InferenceResult(failure="Concat inputs cannot be scalars.")

        # Handle negative axis
        if axis < 0:
            axis += rank

        if axis < 0 or axis >= rank:
            return _common.InferenceResult(
                failure=f"Concat axis {axis} is out of bounds for rank {rank}."
            )

        # Check that all inputs have compatible shapes
        output_dims = list(first_shape.dims)
        concat_dim_size = _common.get_expr(first_shape, axis)

        for i, inp in enumerate(node.inputs[1:], 1):
            if inp is None:
                return _common.InferenceResult(failure=f"Input {i} cannot be None.")
            if inp.shape is None:
                return _common.InferenceResult(failure=f"Input {i} shape cannot be None.")

            input_shape = inp.shape
            if len(input_shape) != rank:
                return _common.InferenceResult(
                    failure=f"All inputs must have same rank. Input {i} has rank {len(input_shape)}, expected {rank}."
                )

            # TODO(justinchuby): Check non-concat dimensions are compatible
            concat_dim_size = concat_dim_size + _common.get_expr(input_shape, axis)
            if inp.type != first_type:
                return _common.InferenceResult(
                    failure=f"Input {i} type {inp.type} does not match first input type {first_type}."
                )

        # Set the concat dimension in output shape
        output_dims[axis] = concat_dim_size
        return _common.InferenceResult(
            values=(ir.Value(shape=ir.Shape(output_dims), type=first_type),)
        )
