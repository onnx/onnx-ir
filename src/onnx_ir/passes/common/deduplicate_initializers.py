# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pass for removing duplicated initializer tensors from a graph."""

from __future__ import annotations

__all__ = [
    "DeduplicateInitializersPass",
]

import hashlib

import onnx_ir
import onnx_ir.traversal


class DeduplicateInitializersPass(onnx_ir.passes.InPlacePass):
    """Remove duplicated initializer tensors from the graph.

    This pass detects initializers with identical shape, dtype, and content,
    and replaces all duplicate references with a canonical one.

    For efficiency, it uses a hash of tensor bytes to group candidates,
    then confirms exact match using the full byte content (to avoid collisions).
    Subgraphs are handled via RecursiveGraphIterator.
    """

    def call(self, model: onnx_ir.Model) -> onnx_ir.passes.PassResult:
        graph = model.graph
        seen = {}      # (dtype, shape) → {hash: [(name, tobytes)]}
        name_map = {}  # Duplicate name → canonical name

        for initializer in list(graph.initializers.values()):
            dtype = initializer.const_value.dtype
            shape = tuple(initializer.const_value.shape)
            content = initializer.const_value.tobytes()
            content_hash = hashlib.sha256(content).hexdigest()

            key = (dtype, shape)
            if key not in seen:
                seen[key] = {}

            group = seen[key]
            if content_hash in group:
                # Verify collision using full bytes
                for existing_name, existing_bytes in group[content_hash]:
                    if existing_bytes == content:
                        name_map[initializer.name] = existing_name
                        graph.initializers.pop(initializer.name)
                        break
                else:
                    group[content_hash].append((initializer.name, content))
            else:
                group[content_hash] = [(initializer.name, content)]

        for node in onnx_ir.traversal.RecursiveGraphIterator(graph):
            for i, input_val in enumerate(node.inputs):
                if input_val and input_val.name in name_map:
                    canonical_name = name_map[input_val.name]
                    replacement = graph.initializers[canonical_name]
                    node.replace_input_with(i, replacement)

        return onnx_ir.passes.PassResult(model=model, modified=bool(name_map))
