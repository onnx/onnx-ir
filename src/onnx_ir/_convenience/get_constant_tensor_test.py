# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from onnx_ir import _core, _enums
from onnx_ir._convenience import get_constant_tensor


class GetConstantTensorTest(unittest.TestCase):
    def test_direct_const_value(self):
        # Test when value has a direct const_value
        tensor = _core.Tensor(np.array([1, 2, 3], dtype=np.int64), name="test_tensor")
        value = _core.Value(name="test_value", type=_core.TensorType(elem_type=_enums.DataType.INT64))
        value.const_value = tensor
        
        result = get_constant_tensor(value)
        
        self.assertIsNotNone(result)
        self.assertEqual(result, tensor)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2, 3], dtype=np.int64))

    def test_no_producer_node(self):
        # Test when value has no producer node
        value = _core.Value(name="test_value", type=_core.TensorType(elem_type=_enums.DataType.FLOAT))
        
        result = get_constant_tensor(value)
        
        self.assertIsNone(result)

    def test_non_constant_producer_node(self):
        # Test when producer node is not a Constant
        graph = _core.Graph(name="test_graph")
        node = _core.Node(
            graph=graph,
            name="test_node",
            domain="",
            op_type="Add",
            inputs=[],
            outputs=["output"],
        )
        
        output_value = graph.get_or_create_value(name="output", type=_core.TensorType(elem_type=_enums.DataType.FLOAT))
        
        result = get_constant_tensor(output_value)
        
        self.assertIsNone(result)

    def test_constant_float_value(self):
        # Test with Constant node with float value
        graph = _core.Graph(name="test_graph")
        value = graph.get_or_create_value(name="output", type=_core.TensorType(elem_type=_enums.DataType.FLOAT))
        
        node = _core.Node(
            graph=graph,
            name="constant_node",
            domain="",
            op_type="Constant",
            inputs=[],
            outputs=["output"],
        )
        
        node.attributes["value_float"] = _core.Attr(value=3.14)
        
        result = get_constant_tensor(value)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "output")
        np.testing.assert_almost_equal(result.numpy(), np.array(3.14, dtype=np.float32))

    def test_constant_int_value(self):
        # Test with Constant node with int value
        graph = _core.Graph(name="test_graph")
        value = graph.get_or_create_value(name="output", type=_core.TensorType(elem_type=_enums.DataType.INT64))
        
        node = _core.Node(
            graph=graph,
            name="constant_node",
            domain="",
            op_type="Constant",
            inputs=[],
            outputs=["output"],
        )
        
        node.attributes["value_int"] = _core.Attr(value=42)
        
        result = get_constant_tensor(value)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "output")
        np.testing.assert_array_equal(result.numpy(), np.array(42, dtype=np.int64))

    def test_constant_tensor_value(self):
        # Test with Constant node with tensor value
        graph = _core.Graph(name="test_graph")
        value = graph.get_or_create_value(name="output", type=_core.TensorType(elem_type=_enums.DataType.FLOAT))
        
        node = _core.Node(
            graph=graph,
            name="constant_node",
            domain="",
            op_type="Constant",
            inputs=[],
            outputs=["output"],
        )
        
        tensor = _core.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="tensor_value")
        node.attributes["value"] = _core.Attr(value=tensor)
        
        result = get_constant_tensor(value)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()