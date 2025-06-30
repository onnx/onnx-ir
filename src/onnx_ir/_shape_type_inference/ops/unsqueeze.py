"""Unsqueeze operation inferrer for ONNX IR nodes."""

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class UnsqueezeInferrer(_common.NodeInferrer):
    """Inferrer for Unsqueeze operations."""

    def __init__(self, opsets: Collection[int] | None = None) -> None:
        """Initialize the Unsqueeze inferrer."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__("Unsqueeze", opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Unsqueeze operations."""
        if len(node.inputs) < 1 or len(node.inputs) > 2:
            return _common.InferenceResult(
                failure=f"Unsqueeze operation must have 1 or 2 inputs, got {len(node.inputs)}."
            )
        if node.inputs[0] is None:
            return _common.InferenceResult(failure="Unsqueeze operation input cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Unsqueeze operation must have exactly one output, got {len(node.outputs)}."
            )

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Unsqueeze input shape cannot be None.")

        input_rank = len(input_shape)

        # Get axes to unsqueeze
        axes = None

        # Check for axes in second input (opset >= 13)
        if len(node.inputs) == 2 and node.inputs[1] is not None:
            if (
                hasattr(node.inputs[1], "initializer_value")
                and node.inputs[1].initializer_value is not None
            ):
                axes = node.inputs[1].initializer_value.tolist()
                if not isinstance(axes, list):
                    axes = [axes]
        else:
            # Check for axes attribute (opset < 13)
            for attr in node.attributes:
                if attr.name == "axes":
                    axes = list(attr.value.ints)
                    break

        if axes is None:
            return _common.InferenceResult(failure="Unsqueeze operation requires axes.")

        # Calculate output rank
        output_rank = input_rank + len(axes)

        # Normalize negative axes relative to output rank
        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis += output_rank
            if axis < 0 or axis >= output_rank:
                return _common.InferenceResult(
                    failure=f"Unsqueeze axis {axis} is out of bounds for output rank {output_rank}."
                )
            normalized_axes.append(axis)

        # Check for duplicate axes
        if len(set(normalized_axes)) != len(normalized_axes):
            return _common.InferenceResult(failure="Unsqueeze axes must be unique.")

        # Build output shape by inserting 1s at specified axes
        output_shape = ir.Shape([0] * output_rank)
        input_axis = 0

        for output_axis in range(output_rank):
            if output_axis in normalized_axes:
                # Insert dimension of size 1
                _common.set_expr(output_shape, output_axis, 1)
            else:
                # Copy dimension from input
                input_dim_expr = _common.get_expr(input_shape, input_axis)
                _common.set_expr(output_shape, output_axis, input_dim_expr)
                input_axis += 1

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
        )
