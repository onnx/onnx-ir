"""AveragePool operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _compute_pool_output_size(
    input_size: int | None,
    kernel_size: int,
    padding_before: int,
    padding_after: int,
    stride: int,
    ceil_mode: bool = False,
) -> int | None:
    """Compute output size for pooling dimension."""
    if input_size is None:
        return None
    padded_input_size = input_size + padding_before + padding_after
    if ceil_mode:
        return (padded_input_size - kernel_size + stride - 1) // stride + 1
    else:
        return (padded_input_size - kernel_size) // stride + 1


class AveragePoolInferrer(_common.NodeInferrer):
    """Inferrer for AveragePool operations."""

    def __init__(self) -> None:
        """Initialize the AveragePool inferrer."""
        super().__init__("AveragePool", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for AveragePool operations."""
        assert node.inputs[0] is not None

        input_shape = node.inputs[0].shape
        if input_shape is None:
            return _common.InferenceResult(failure="AveragePool input shape is not known.")

        input_rank = len(input_shape)
        if input_rank < 3:
            return _common.InferenceResult(
                failure="AveragePool input must have at least 3 dimensions."
            )

        # Input: [N, C, D1, D2, ..., Dn]
        # Output: [N, C, out_d1, out_d2, ..., out_dn]

        batch_size = input_shape.dims[0]
        channels = input_shape.dims[1]
        spatial_dims = input_rank - 2

        # Get required kernel_shape attribute
        kernel_shape = node.attributes.get_ints("kernel_shape")
        if kernel_shape is None:
            return _common.InferenceResult(
                failure="AveragePool requires kernel_shape attribute."
            )

        if len(kernel_shape) != spatial_dims:
            return _common.InferenceResult(
                failure=f"AveragePool kernel_shape length {len(kernel_shape)} must match spatial dimensions {spatial_dims}."
            )

        # Get optional attributes
        strides = node.attributes.get_ints("strides", kernel_shape)  # Default to kernel_shape
        pads = node.attributes.get_ints("pads", [0] * (2 * spatial_dims))
        ceil_mode = node.attributes.get_int("ceil_mode", 0) != 0

        if len(strides) != spatial_dims:
            return _common.InferenceResult(
                failure=f"AveragePool strides length {len(strides)} must match spatial dimensions {spatial_dims}."
            )
        if len(pads) != 2 * spatial_dims:
            return _common.InferenceResult(
                failure=f"AveragePool pads length {len(pads)} must be 2x spatial dimensions {spatial_dims}."
            )

        # Compute output spatial dimensions
        output_spatial_dims = []
        for i in range(spatial_dims):
            input_dim = input_shape.dims[2 + i]
            kernel_size = kernel_shape[i]
            stride = strides[i]
            padding_before = pads[i]
            padding_after = pads[i + spatial_dims]

            if isinstance(input_dim, int):
                output_dim = _compute_pool_output_size(
                    input_dim, kernel_size, padding_before, padding_after, stride, ceil_mode
                )
            else:
                # Symbolic dimensions - we can't compute exact size
                output_dim = None

            output_spatial_dims.append(output_dim)

        output_dims = [batch_size, channels] + output_spatial_dims
        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
