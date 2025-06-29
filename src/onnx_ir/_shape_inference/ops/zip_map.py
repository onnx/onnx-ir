# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_zip_map_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ZipMap op."""
    # ZipMap outputs a sequence of maps. The shape inference for this is complex
    # and depends on the input types and the 'classlabels' attribute.
    # For now, we will mark it as unsupported.
    return InferenceResult(InferenceStatus.UNSUPPORTED, "ZipMap shape inference is not yet implemented.")