"""If operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class IfInferrer(_common.NodeInferrer):
    """Inferrer for If operations."""

    def __init__(self) -> None:
        """Initialize the If inferrer."""
        super().__init__("If", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(1)  # condition
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for If operations."""
        assert node.inputs[0] is not None  # condition

        condition_shape = node.inputs[0].shape
        if condition_shape is None:
            return _common.InferenceResult(failure="If condition input shape is not known.")

        # Condition should be a scalar boolean
        if len(condition_shape) != 0:
            return _common.InferenceResult(failure="If condition input must be a scalar.")

        # For If operations, we need to analyze the then_branch and else_branch graphs
        # Since we don't have access to subgraph analysis here, we'll return unknown shapes

        num_outputs = len(node.outputs)
        output_values = []

        for i in range(num_outputs):
            # Each output shape is unknown since it depends on the branch execution
            output_shape = ir.Shape([None])  # Unknown shape
            output_type = None  # Unknown type
            output_values.append(ir.Value(shape=output_shape, type=output_type))

        return _common.InferenceResult(values=output_values)
