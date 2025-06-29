# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_if_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for If op."""
    then_branch = node.attributes.get_graph("then_branch")
    else_branch = node.attributes.get_graph("else_branch")

    # For now, assume subgraphs are already shape-inferred by the main engine.
    # The main engine iterates through all nodes, including those in subgraphs.

    # Check if the condition input is a constant
    cond_input = node.inputs[0]
    if cond_input.const_value is not None:
        # If condition is constant, only one branch is taken
        cond_value = cond_input.const_value.numpy().item()
        if cond_value > 0:  # True branch
            source_branch_outputs = then_branch.outputs
        else:  # False branch
            source_branch_outputs = else_branch.outputs

        if len(node.outputs) != len(source_branch_outputs):
            return InferenceResult(InferenceStatus.FAILURE, "Number of outputs mismatch between If node and selected branch.")

        for i, output_value in enumerate(node.outputs):
            inferred_shape = shape_env.get_shape(source_branch_outputs[i])
            inferred_dtype = shape_env.get_dtype(source_branch_outputs[i])
            if inferred_shape is None or inferred_dtype is None:
                return InferenceResult(InferenceStatus.UNSUPPORTED, f"Subgraph output shape or dtype not available for {source_branch_outputs[i].name}")
            shape_env.set_shape_and_type(output_value, inferred_shape, inferred_dtype)
            if source_branch_outputs[i].const_value is not None:
                output_value.const_value = source_branch_outputs[i].const_value
        return InferenceResult(InferenceStatus.SUCCESS)

    # If condition is not constant, both branches must have compatible outputs
    if len(then_branch.outputs) != len(else_branch.outputs) or \
       len(node.outputs) != len(then_branch.outputs):
        return InferenceResult(InferenceStatus.FAILURE, "Number of outputs mismatch between then_branch, else_branch, or If node.")

    for i, output_value in enumerate(node.outputs):
        then_output = then_branch.outputs[i]
        else_output = else_branch.outputs[i]

        then_shape = shape_env.get_shape(then_output)
        then_dtype = shape_env.get_dtype(then_output)
        else_shape = shape_env.get_shape(else_output)
        else_dtype = shape_env.get_dtype(else_output)

        if then_shape is None or then_dtype is None or else_shape is None or else_dtype is None:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Subgraph output shape or dtype not available for If node output {i}")

        # Check if types are compatible
        if then_dtype != else_dtype:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Incompatible output types for If node output {i}: {then_dtype} vs {else_dtype}")

        # Check if shapes are compatible. For now, require exact match.
        # TODO: Implement more sophisticated shape fusion if needed (e.g., merging symbolic dims)
        if then_shape != else_shape:
            return InferenceResult(InferenceStatus.UNSUPPORTED, f"Incompatible output shapes for If node output {i}: {then_shape} vs {else_shape}")

        shape_env.set_shape_and_type(output_value, then_shape, then_dtype)
        # If both are constant and equal, then output is constant
        if then_output.const_value is not None and else_output.const_value is not None and \
           then_output.const_value.numpy() == else_output.const_value.numpy():
            output_value.const_value = then_output.const_value

    return InferenceResult(InferenceStatus.SUCCESS)