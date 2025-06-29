# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_loop_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for Loop op."""
    body_graph = node.attributes.get_graph("body")
    num_loop_carried_deps = len(node.inputs) - 2 # M - initial_count, condition

    # The outputs of the Loop node correspond to the scan outputs and loop carried dependencies.
    # The body graph outputs are: [cond, loop_carried_deps..., scan_outputs...]

    # For loop carried dependencies (M outputs of Loop node correspond to M-1 outputs of body graph)
    for i in range(num_loop_carried_deps):
        # Loop carried dependency output from body graph is at index i + 1 (after condition)
        body_output = body_graph.outputs[i + 1]
        inferred_shape = shape_env.get_shape(body_output)
        inferred_dtype = shape_env.get_dtype(body_output)
        if inferred_shape is None or inferred_dtype is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Subgraph output shape or dtype not available for {body_output.name}")
        shape_env.set_shape_and_type(node.outputs[i], inferred_shape, inferred_dtype)

    # For scan outputs
    # The scan outputs have an additional dimension at axis 0 for the number of iterations.
    # For now, we'll assume this dimension is unknown (None).
    for i in range(num_loop_carried_deps, len(node.outputs)):
        body_output = body_graph.outputs[i + 1] # +1 for condition
        inferred_shape = shape_env.get_shape(body_output)
        inferred_dtype = shape_env.get_dtype(body_output)
        if inferred_shape is None or inferred_dtype is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Subgraph output shape or dtype not available for {body_output.name}")
        scan_output_shape = [None] + list(inferred_shape) # Add an unknown dimension for iterations
        shape_env.set_shape_and_type(node.outputs[i], ir.Shape(scan_output_shape), inferred_dtype)

    return InferenceResult(InferenceStatus.SUCCESS)