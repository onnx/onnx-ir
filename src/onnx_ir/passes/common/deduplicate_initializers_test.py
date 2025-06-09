# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeduplicateInitializersPass."""

import unittest
import onnx
import numpy as np

import onnx_ir as ir
import onnx_ir.passes.common.deduplicate_initializers as dedup_pass


class DeduplicateInitializersTest(unittest.TestCase):
    def apply_pass(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model_ir = ir.serde.deserialize_model(model)
        dedup_pass.DeduplicateInitializersPass()(model_ir)
        return ir.serde.serialize_model(model_ir)

    def test_deduplicates_identical_initializers(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[3] w1 = {1.0, 2.0, 3.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        self.assertEqual(len(model.graph.initializer), 2)
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)
        add_node = new_model.graph.node[0]
        self.assertEqual(add_node.input[0], add_node.input[1])

    
    def test_initializers_with_different_shapes_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[3] w2 = {1.0, 2.0, 3.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    def test_initializers_with_different_dtypes_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, double[2] w2 = {1.0, 2.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    def test_scalar_initializer_deduplication(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float w1 = {5.0}, float w2 = {5.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)

    def test_multiple_duplicates(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 1.0}, float[2] w2 = {1.0, 1.0}, float[2] w3 = {1.0, 1.0}> {
                temp = Add(w1, w2)
                out = Add(temp, w3)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 1)

    def test_unique_values_not_deduplicated(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: ["" : 17]>
            agraph () => ()
            <float[2] w1 = {1.0, 2.0}, float[2] w2 = {2.0, 1.0}> {
                sum = Add(w1, w2)
            }
            """
        )
        new_model = self.apply_pass(model)
        self.assertEqual(len(new_model.graph.initializer), 2)

    
if __name__ == "__main__":
    unittest.main()
