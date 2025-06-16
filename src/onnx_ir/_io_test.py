# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the _io module."""

import os
import tempfile
import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir import _io


def _create_initializer(tensor: ir.TensorProtocol) -> ir.Value:
    return ir.Value(
        name=tensor.name,
        shape=tensor.shape,
        type=ir.TensorType(tensor.dtype),
        const_value=tensor,
    )


def _create_simple_model_with_initializers() -> ir.Model:
    tensor_0 = ir.tensor([0.0], dtype=ir.DataType.FLOAT, name="initializer_0")
    initializer = _create_initializer(tensor_0)
    tensor_1 = ir.tensor([1.0], dtype=ir.DataType.FLOAT)
    identity_node = ir.Node("", "Identity", inputs=(initializer,))
    identity_node.outputs[0].shape = ir.Shape([1])
    identity_node.outputs[0].dtype = ir.DataType.FLOAT
    identity_node.outputs[0].name = "identity_0"
    const_node = ir.Node(
        "",
        "Constant",
        inputs=(),
        outputs=(
            ir.Value(name="const_0", shape=tensor_1.shape, type=ir.TensorType(tensor_1.dtype)),
        ),
        attributes=ir.convenience.convert_attributes(dict(value=tensor_1)),
    )
    graph = ir.Graph(
        inputs=[initializer],
        outputs=[*identity_node.outputs, *const_node.outputs],
        nodes=[identity_node, const_node],
        initializers=[initializer],
        name="test_graph",
    )
    return ir.Model(graph, ir_version=10)


class IOFunctionsTest(unittest.TestCase):
    def test_load(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            _io.save(model, path)
            loaded_model = _io.load(path)
        self.assertEqual(loaded_model.ir_version, model.ir_version)
        self.assertEqual(loaded_model.graph.name, model.graph.name)
        self.assertEqual(len(loaded_model.graph.initializers), 1)
        self.assertEqual(len(loaded_model.graph), 2)
        np.testing.assert_array_equal(
            loaded_model.graph.initializers["initializer_0"].const_value.numpy(),
            np.array([0.0]),
        )
        np.testing.assert_array_equal(
            loaded_model.graph.node(1).attributes["value"].as_tensor().numpy(), np.array([1.0])
        )
        self.assertEqual(loaded_model.graph.inputs[0].name, "initializer_0")
        self.assertEqual(loaded_model.graph.outputs[0].name, "identity_0")
        self.assertEqual(loaded_model.graph.outputs[1].name, "const_0")

    def test_save_with_external_data_does_not_modify_model(self):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        # There may be clean up errors on Windows, so we ignore them
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file, size_threshold_bytes=0)
            self.assertTrue(os.path.exists(path))
            external_data_path = os.path.join(tmpdir, external_data_file)
            self.assertTrue(os.path.exists(external_data_path))
            loaded_model = _io.load(path)

            # The loaded model contains external data
            initializer_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_tensor, ir.ExternalTensor)
            # The attribute is not externalized
            const_attr_tensor = loaded_model.graph.node(1).attributes["value"].as_tensor()
            self.assertIsInstance(const_attr_tensor, ir.TensorProtoTensor)
            np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
            np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

        # The original model is not changed and can be accessed even if the
        # external data file is deleted
        initializer_tensor = model.graph.initializers["initializer_0"].const_value
        self.assertIsInstance(initializer_tensor, ir.Tensor)
        const_attr_tensor = model.graph.node(1).attributes["value"].as_tensor()
        self.assertIsInstance(const_attr_tensor, ir.Tensor)
        np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))
        np.testing.assert_array_equal(const_attr_tensor.numpy(), np.array([1.0]))

    def test_save_raise_when_external_data_is_not_relative_path(self):
        model = _create_simple_model_with_initializers()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = os.path.join(tmpdir, "model.data")
            with self.assertRaises(ValueError):
                _io.save(model, path, external_data=external_data_file)

    def test_save_with_external_data_invalidates_obsolete_external_tensors(self):
        model = _create_simple_model_with_initializers()
        self.assertIsInstance(model.graph.initializers["initializer_0"].const_value, ir.Tensor)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(model, path, external_data=external_data_file, size_threshold_bytes=0)
            loaded_model = _io.load(path)
            # Now if we load the model back, create a different initializer and save
            # the model to the same external data file, the existing external tensor
            # should be invalidated
            tensor_2 = ir.tensor([2.0], dtype=ir.DataType.FLOAT, name="initializer_2")
            initializer_2 = _create_initializer(tensor_2)
            loaded_model.graph.initializers["initializer_2"] = initializer_2
            _io.save(
                loaded_model, path, external_data=external_data_file, size_threshold_bytes=0
            )
            initializer_0_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_0_tensor, ir.ExternalTensor)
            self.assertFalse(initializer_0_tensor.valid())
            with self.assertRaisesRegex(ValueError, "is invalidated"):
                # The existing model has to be modified to use in memory tensors
                # for the values to stay correct. Saving again should raise an error
                _io.save(
                    loaded_model,
                    path,
                    external_data=external_data_file,
                    size_threshold_bytes=0,
                )

    def test_save_with_external_data_calls_callback_with_correct_metadata(self):
        """Test that the callback is invoked with correct metadata when saving with external data."""
        model = _create_simple_model_with_initializers()
        callback_calls = []
        
        def test_callback(tensor, metadata):
            callback_calls.append((tensor, metadata.copy()))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            _io.save(
                model, 
                path, 
                external_data=external_data_file, 
                size_threshold_bytes=0,
                callback=test_callback
            )
            
            # Verify callback was called once for the one initializer
            self.assertEqual(len(callback_calls), 1)
            
            tensor, metadata = callback_calls[0]
            # Verify tensor is the expected initializer tensor
            self.assertEqual(tensor.name, "initializer_0")
            np.testing.assert_array_equal(tensor.numpy(), np.array([0.0]))
            
            # Verify metadata contains expected keys and values
            self.assertIn("total", metadata)
            self.assertIn("index", metadata)
            self.assertIn("offset", metadata)
            self.assertIn("size_bytes", metadata)
            
            self.assertEqual(metadata["total"], 1)
            self.assertEqual(metadata["index"], 0)
            self.assertIsInstance(metadata["offset"], int)
            self.assertGreaterEqual(metadata["offset"], 0)
            self.assertIsInstance(metadata["size_bytes"], int)
            self.assertGreater(metadata["size_bytes"], 0)

    def test_save_with_external_data_callback_none_works_correctly(self):
        """Test that saving with callback=None works the same as before."""
        model = _create_simple_model_with_initializers()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            external_data_file = "model.data"
            # This should work without any issues
            _io.save(
                model, 
                path, 
                external_data=external_data_file, 
                size_threshold_bytes=0,
                callback=None
            )
            
            self.assertTrue(os.path.exists(path))
            external_data_path = os.path.join(tmpdir, external_data_file)
            self.assertTrue(os.path.exists(external_data_path))
            
            # Verify model can be loaded correctly
            loaded_model = _io.load(path)
            initializer_tensor = loaded_model.graph.initializers["initializer_0"].const_value
            self.assertIsInstance(initializer_tensor, ir.ExternalTensor)
            np.testing.assert_array_equal(initializer_tensor.numpy(), np.array([0.0]))

    def test_save_without_external_data_ignores_callback(self):
        """Test that callback is ignored when not using external data."""
        model = _create_simple_model_with_initializers()
        callback_mock = unittest.mock.Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            # Save without external data but with callback
            _io.save(model, path, callback=callback_mock)
            
            # Callback should not have been called since no external data
            callback_mock.assert_not_called()
            
            # Model should still save correctly
            self.assertTrue(os.path.exists(path))
            loaded_model = _io.load(path)
            self.assertEqual(loaded_model.ir_version, model.ir_version)


if __name__ == "__main__":
    unittest.main()
