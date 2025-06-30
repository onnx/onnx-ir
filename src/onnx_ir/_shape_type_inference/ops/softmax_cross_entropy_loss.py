"""SoftmaxCrossEntropyLoss operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class SoftmaxCrossEntropyLossInferrer(_common.NodeInferrer):
    """Inferrer for SoftmaxCrossEntropyLoss operations."""

    def __init__(self) -> None:
        """Initialize the SoftmaxCrossEntropyLoss inferrer."""
        super().__init__("SoftmaxCrossEntropyLoss", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # At least scores and labels
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for SoftmaxCrossEntropyLoss operations."""
        assert node.inputs[0] is not None  # scores
        assert node.inputs[1] is not None  # labels

        scores_shape = node.inputs[0].shape
        labels_shape = node.inputs[1].shape

        if scores_shape is None:
            return _common.InferenceResult(
                failure="SoftmaxCrossEntropyLoss scores input shape is not known."
            )
        if labels_shape is None:
            return _common.InferenceResult(
                failure="SoftmaxCrossEntropyLoss labels input shape is not known."
            )

        # Get reduction attribute (default is "mean")
        reduction = node.attributes.get_string("reduction", "mean")

        num_outputs = len(node.outputs)
        output_values = []

        # First output: loss
        if reduction == "none":
            # No reduction - output has same shape as input batch dimensions
            if len(scores_shape) >= 2:
                loss_shape = ir.Shape(list(scores_shape.dims[:-1]))  # Remove class dimension
            else:
                loss_shape = ir.Shape([])  # Scalar
        else:
            # Reduction applied - output is scalar
            loss_shape = ir.Shape([])

        loss_type = node.inputs[0].type  # Same type as scores
        output_values.append(ir.Value(shape=loss_shape, type=loss_type))

        # Second output: log_prob (if present)
        if num_outputs >= 2:
            # Log probabilities have the same shape as scores
            log_prob_shape = scores_shape
            log_prob_type = node.inputs[0].type
            output_values.append(ir.Value(shape=log_prob_shape, type=log_prob_type))

        return _common.InferenceResult(values=output_values)
