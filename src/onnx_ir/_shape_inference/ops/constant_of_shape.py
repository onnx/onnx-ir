# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx_ir as ir
from onnx_ir import _enums
from onnx_ir._shape_inference.result import InferenceResult, InferenceStatus
from onnx_ir._shape_inference.shape_env import ShapeEnv


def infer_constant_of_shape_shape(node: ir.Node, shape_env: ShapeEnv) -> InferenceResult:
    """Shape inference for ConstantOfShape op."""
    # The output shape is determined by the first input, which is a 1D tensor containing the desired output shape.
    shape_input_value = node.inputs[0]
    if shape_input_value.const_value is None:
        return InferenceResult(InferenceStatus.UNSUPPORTED, "Shape input must be a constant for ConstantOfShape.")
    
    output_shape_dims = list(shape_input_value.const_value.numpy().astype(np.int64))
    output_shape = ir.Shape(output_shape_dims)

    # The output data type is determined by the 'value' attribute, or defaults to float32.
    value_attr = node.attributes.get_tensor("value")
    if value_attr is not None:
        output_dtype = value_attr.dtype
    else:
        output_dtype = _enums.DataType.FLOAT32 # Default value type

    shape_env.set_shape_and_type(node.outputs[0], output_shape, output_dtype)

    # If the value is constant, we can also infer the constant value of the output
    if value_attr is not None and output_shape.is_static():
        fill_value = value_attr.numpy().item()
        const_array = np.full(output_shape.numpy(), fill_value, dtype=output_dtype.numpy())
        node.outputs[0].const_value = ir.Tensor(const_array)

    return InferenceResult(InferenceStatus.SUCCESS)