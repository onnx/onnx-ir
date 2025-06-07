# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeduplicateInitializersPass."""

import unittest
import numpy as np

from onnx_ir._core import Tensor, Value, Node, Graph
from onnx_ir.passes.common.deduplicate_initializers import DeduplicateInitializersPass


class DeduplicateInitializersPassTest(unittest.TestCase):
    def setUp(self):
        # Shared tensor content
        self.arr = np.array([1, 2, 3])
        self.tensor1 = Tensor(self.arr)
        self.tensor2 = Tensor(self.arr.copy())  # Identical but separate object
        self.tensor3 = Tensor(self.arr.copy())  # For subgraph

    def test_deduplication_in_main_and_subgraph(self):
        v1 = Value(name="w1", const_value=self.tensor1)
        v2 = Value(name="w2", const_value=self.tensor2)
        v3 = Value(name="w3", const_value=self.tensor3)

        # Main graph node using w1 and w2
        main_node = Node("", "Add", inputs=[v1, v2], outputs=[])

        # Subgraph node using w3
        sub_node = Node("", "Conv", inputs=[v3], outputs=[])
        subgraph = Graph(
            inputs=[],
            outputs=[],
            nodes=[sub_node],
            initializers=[v3],
            name="subgraph"
        )

        # Link subgraph to main node
        main_node.blocks = [subgraph]

        # Main graph with w1 and w2 (duplicates)
        main_graph = Graph(
            inputs=[],
            outputs=[],
            nodes=[main_node],
            initializers=[v1, v2],
            name="main_graph"
        )

        DeduplicateInitializersPass().apply(main_graph)

        # Post conditions
        self.assertIn("w1", main_graph.initializers)
        self.assertNotIn("w2", main_graph.initializers)
        self.assertEqual(main_node.inputs[0].name, "w1")
        self.assertEqual(main_node.inputs[1].name, "w1")

        # Subgraph should be untouched (no cross-graph deduplication)
        self.assertIn("w3", subgraph.initializers)
        self.assertEqual(sub_node.inputs[0].name, "w3")


if __name__ == "__main__":
    unittest.main()
