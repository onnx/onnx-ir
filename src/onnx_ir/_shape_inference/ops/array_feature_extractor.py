# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_array_feature_extractor_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ArrayFeatureExtractor op."""
    data_shape = shape_env.get_shape(node.inputs[0])
    indices_shape = shape_env.get_shape(node.inputs[1])
    data_dtype = shape_env.get_dtype(node.inputs[0])

    if data_shape is None or indices_shape is None or data_dtype is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Input shape or dtype not available.")

    output_shape = list(data_shape)[:-1] + list(indices_shape)

    shape_env.set_shape_and_type(node.outputs[0], ir.Shape(output_shape), data_dtype)
    return InferenceResult(InferenceStatus.SUCCESS)