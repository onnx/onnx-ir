"""Range operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class RangeInferrer(_common.NodeInferrer):
    """Inferrer for Range operations."""

    def __init__(self) -> None:
        """Initialize the Range inferrer."""
        super().__init__("Range", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(3)
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Range operations."""
        assert node.inputs[0] is not None  # start
        assert node.inputs[1] is not None  # limit
        assert node.inputs[2] is not None  # delta

        start_shape = node.inputs[0].shape
        limit_shape = node.inputs[1].shape
        delta_shape = node.inputs[2].shape

        # All inputs should be scalars
        if start_shape is None or len(start_shape) != 0:
            return _common.InferenceResult(failure="Range start input must be a scalar.")
        if limit_shape is None or len(limit_shape) != 0:
            return _common.InferenceResult(failure="Range limit input must be a scalar.")
        if delta_shape is None or len(delta_shape) != 0:
            return _common.InferenceResult(failure="Range delta input must be a scalar.")

        # Try to get constant values to compute output size
        start_tensor = ir.convenience.get_const_tensor(node.inputs[0])
        limit_tensor = ir.convenience.get_const_tensor(node.inputs[1])
        delta_tensor = ir.convenience.get_const_tensor(node.inputs[2])

        if start_tensor is not None and limit_tensor is not None and delta_tensor is not None:
            # All parameters are constant
            start_val = start_tensor.numpy().item()
            limit_val = limit_tensor.numpy().item()
            delta_val = delta_tensor.numpy().item()

            if delta_val == 0:
                return _common.InferenceResult(failure="Range delta cannot be zero.")

            # Calculate output size
            if delta_val > 0:
                size = max(0, (limit_val - start_val + delta_val - 1) // delta_val)
            else:
                size = max(0, (limit_val - start_val + delta_val + 1) // delta_val)

            output_shape = ir.Shape([int(size)])
        else:
            # Parameters are not all constant, output size is unknown
            output_shape = ir.Shape([None])

        # Output type is the same as input type
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
