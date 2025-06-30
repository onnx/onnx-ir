"""ArrayFeatureExtractor operation inferrer for ONNX IR nodes."""

from __future__ import annotations

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ArrayFeatureExtractorInferrer(_common.NodeInferrer):
    """Inferrer for ArrayFeatureExtractor operations."""

    def __init__(self) -> None:
        """Initialize the ArrayFeatureExtractor inferrer."""
        super().__init__("ArrayFeatureExtractor", opsets=range(sys.maxsize))

    @_common.requires_non_none_inputs(2)  # X, Y
    @_common.requires_outputs(1)
    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for ArrayFeatureExtractor operations."""
        assert node.inputs[0] is not None  # X (data)
        assert node.inputs[1] is not None  # Y (indices)

        data_shape = node.inputs[0].shape
        indices_shape = node.inputs[1].shape

        if data_shape is None:
            return _common.InferenceResult(
                failure="ArrayFeatureExtractor data input shape is not known."
            )
        if indices_shape is None:
            return _common.InferenceResult(
                failure="ArrayFeatureExtractor indices input shape is not known."
            )

        # Data should be 2D: [N, F] where N is batch size, F is feature size
        if len(data_shape) != 2:
            return _common.InferenceResult(
                failure="ArrayFeatureExtractor data input must be 2D."
            )

        # Indices should be 1D: [K] where K is number of features to extract
        if len(indices_shape) != 1:
            return _common.InferenceResult(
                failure="ArrayFeatureExtractor indices input must be 1D."
            )

        batch_size = data_shape.dims[0]
        num_features = indices_shape.dims[0]

        # Output shape: [N, K] where N is batch size, K is number of extracted features
        output_shape = ir.Shape([batch_size, num_features])
        output_type = node.inputs[0].type

        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
