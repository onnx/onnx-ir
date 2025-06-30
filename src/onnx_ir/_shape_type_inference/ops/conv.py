"""Conv operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


def _compute_conv_output_size(
    input_size: int | None,
    kernel_size: int,
    padding_before: int,
    padding_after: int,
    stride: int,
    dilation: int,
) -> int | None:
    """Compute output size for convolution dimension."""
    if input_size is None:
        return None
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    padded_input_size = input_size + padding_before + padding_after
    return (padded_input_size - effective_kernel_size) // stride + 1


class ConvInferrer(_common.NodeInferrer):
    """Inferrer for Conv operations."""

    def __init__(self) -> None:
        """Initialize the Conv inferrer."""
        super().__init__("Conv", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # At least X and W
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Conv operations."""
        assert node.inputs[0] is not None  # X
        assert node.inputs[1] is not None  # W
        
        input_shape = node.inputs[0].shape
        weight_shape = node.inputs[1].shape
        
        if input_shape is None:
            return _common.InferenceResult(failure="Conv input shape is not known.")
        if weight_shape is None:
            return _common.InferenceResult(failure="Conv weight shape is not known.")

        input_rank = len(input_shape)
        weight_rank = len(weight_shape)
        
        if input_rank < 3:
            return _common.InferenceResult(failure="Conv input must have at least 3 dimensions.")
        if weight_rank < 3:
            return _common.InferenceResult(failure="Conv weight must have at least 3 dimensions.")

        # Input: [N, C, D1, D2, ..., Dn]
        # Weight: [M, C/group, k1, k2, ..., kn]
        # Output: [N, M, out_d1, out_d2, ..., out_dn]
        
        batch_size = input_shape.dims[0]
        output_channels = weight_shape.dims[0]
        spatial_dims = input_rank - 2
        
        if weight_rank != spatial_dims + 2:
            return _common.InferenceResult(
                failure=f"Conv weight rank {weight_rank} incompatible with input rank {input_rank}."
            )

        # Get attributes
        strides = node.attributes.get_ints("strides", [1] * spatial_dims)
        pads = node.attributes.get_ints("pads", [0] * (2 * spatial_dims))
        dilations = node.attributes.get_ints("dilations", [1] * spatial_dims)
        
        if len(strides) != spatial_dims:
            return _common.InferenceResult(
                failure=f"Conv strides length {len(strides)} must match spatial dimensions {spatial_dims}."
            )
        if len(pads) != 2 * spatial_dims:
            return _common.InferenceResult(
                failure=f"Conv pads length {len(pads)} must be 2x spatial dimensions {spatial_dims}."
            )
        if len(dilations) != spatial_dims:
            return _common.InferenceResult(
                failure=f"Conv dilations length {len(dilations)} must match spatial dimensions {spatial_dims}."
            )

        # Compute output spatial dimensions
        output_spatial_dims = []
        for i in range(spatial_dims):
            input_dim = input_shape.dims[2 + i]
            kernel_dim = weight_shape.dims[2 + i]
            stride = strides[i]
            padding_before = pads[i]
            padding_after = pads[i + spatial_dims]
            dilation = dilations[i]
            
            if isinstance(input_dim, int) and isinstance(kernel_dim, int):
                output_dim = _compute_conv_output_size(
                    input_dim, kernel_dim, padding_before, padding_after, stride, dilation
                )
            else:
                # Symbolic dimensions - we can't compute exact size
                output_dim = None
            
            output_spatial_dims.append(output_dim)

        output_dims = [batch_size, output_channels] + output_spatial_dims
        output_shape = ir.Shape(output_dims)
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )