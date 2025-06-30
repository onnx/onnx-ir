"""Loop operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class LoopInferrer(_common.NodeInferrer):
    """Inferrer for Loop operations."""

    def __init__(self) -> None:
        """Initialize the Loop inferrer."""
        super().__init__("Loop", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # At least M and cond
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Loop operations."""
        assert node.inputs[0] is not None  # M (max trip count)
        assert node.inputs[1] is not None  # cond (initial condition)

        m_shape = node.inputs[0].shape
        cond_shape = node.inputs[1].shape

        if m_shape is None:
            return _common.InferenceResult(failure="Loop M input shape is not known.")
        if cond_shape is None:
            return _common.InferenceResult(failure="Loop cond input shape is not known.")

        # M and cond should be scalars
        if len(m_shape) != 0:
            return _common.InferenceResult(failure="Loop M input must be a scalar.")
        if len(cond_shape) != 0:
            return _common.InferenceResult(failure="Loop cond input must be a scalar.")

        # For Loop operations, we need to analyze the body graph
        # Since we don't have access to subgraph analysis here, we'll return unknown shapes

        num_outputs = len(node.outputs)
        output_values = []

        # Check if we have loop-carried dependencies (additional inputs beyond M and cond)
        loop_inputs = node.inputs[2:]  # Skip M and cond

        for i in range(num_outputs):
            if i < len(loop_inputs) and loop_inputs[i] is not None:
                # This might be a loop-carried dependency
                # The output shape might be related to the corresponding input
                input_shape = loop_inputs[i].shape
                if input_shape is not None:
                    # For scan outputs, we might need to add a time dimension
                    # For now, we'll keep the same shape
                    output_shape = input_shape
                    output_type = loop_inputs[i].type
                else:
                    output_shape = ir.Shape([None])
                    output_type = None
            else:
                # Unknown output
                output_shape = ir.Shape([None])
                output_type = None

            output_values.append(ir.Value(shape=output_shape, type=output_type))

        return _common.InferenceResult(values=output_values)
