"""NonMaxSuppression operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class NonMaxSuppressionInferrer(_common.NodeInferrer):
    """Inferrer for NonMaxSuppression operations."""

    def __init__(self) -> None:
        """Initialize the NonMaxSuppression inferrer."""
        super().__init__("NonMaxSuppression", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # At least boxes and scores
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for NonMaxSuppression operations."""
        assert node.inputs[0] is not None  # boxes
        assert node.inputs[1] is not None  # scores

        boxes_shape = node.inputs[0].shape
        scores_shape = node.inputs[1].shape

        if boxes_shape is None:
            return _common.InferenceResult(
                failure="NonMaxSuppression boxes input shape is not known."
            )
        if scores_shape is None:
            return _common.InferenceResult(
                failure="NonMaxSuppression scores input shape is not known."
            )

        # Boxes should be [num_batches, spatial_dimension, 4]
        if len(boxes_shape) != 3:
            return _common.InferenceResult(failure="NonMaxSuppression boxes input must be 3D.")
        if isinstance(boxes_shape.dims[2], int) and boxes_shape.dims[2] != 4:
            return _common.InferenceResult(
                failure="NonMaxSuppression boxes last dimension must be 4."
            )

        # Scores should be [num_batches, num_classes, spatial_dimension]
        if len(scores_shape) != 3:
            return _common.InferenceResult(
                failure="NonMaxSuppression scores input must be 3D."
            )

        # Check compatibility between boxes and scores
        # boxes: [N, spatial_dimension, 4]
        # scores: [N, num_classes, spatial_dimension]
        boxes_batch = boxes_shape.dims[0]
        boxes_spatial = boxes_shape.dims[1]
        scores_batch = scores_shape.dims[0]
        scores_spatial = scores_shape.dims[2]

        # For symbolic inference, we assume batch and spatial dimensions match
        # In practice, this would need runtime verification

        # The number of selected indices is unknown at compile time
        # Output shape is [num_selected_indices, 3] where each row is [batch_index, class_index, box_index]
        output_shape = ir.Shape([None, 3])

        # NonMaxSuppression always outputs INT64 indices
        output_type = ir.TensorType.INT64

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
