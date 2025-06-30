"""Squeeze operation inferrer for ONNX IR nodes."""

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class SqueezeInferrer(_common.NodeInferrer):
    """Inferrer for Squeeze operations."""

    def __init__(self, opsets: Collection[int] | None = None) -> None:
        """Initialize the Squeeze inferrer."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__("Squeeze", opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Squeeze operations."""
        if len(node.inputs) < 1 or len(node.inputs) > 2:
            return _common.InferenceResult(
                failure=f"Squeeze operation must have 1 or 2 inputs, got {len(node.inputs)}."
            )
        if node.inputs[0] is None:
            return _common.InferenceResult(failure="Squeeze operation input cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Squeeze operation must have exactly one output, got {len(node.outputs)}."
            )

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="Squeeze input shape cannot be None.")

        rank = len(input_shape)

        # Get axes to squeeze
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
            # No axes specified - squeeze all dimensions of size 1
            output_dims = []
            for i, dim in enumerate(input_shape):
                dim_expr = _common.get_expr(input_shape, i)
                # For symbolic dimensions, we assume they are not 1
                # Only squeeze literal 1s
                if isinstance(dim, int) and dim == 1:
                    continue  # Skip dimension of size 1
                else:
                    output_dims.append(dim_expr)
        else:
            # Normalize negative axes
            normalized_axes = []
            for axis in axes:
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    return _common.InferenceResult(
                        failure=f"Squeeze axis {axis} is out of bounds for rank {rank}."
                    )
                normalized_axes.append(axis)

            # Validate that specified axes have dimension 1 (for literal dimensions)
            for axis in normalized_axes:
                dim = input_shape[axis]
                if isinstance(dim, int) and dim != 1:
                    return _common.InferenceResult(
                        failure=f"Cannot squeeze axis {axis} with dimension {dim} (must be 1)."
                    )

            # Build output shape excluding squeezed axes
            output_dims = []
            for i in range(rank):
                if i not in normalized_axes:
                    output_dims.append(_common.get_expr(input_shape, i))

        # Create output shape
        output_shape = ir.Shape([0] * len(output_dims))
        for i, dim_expr in enumerate(output_dims):
            _common.set_expr(output_shape, i, dim_expr)

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
        )
