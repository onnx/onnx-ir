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
        seen: dict[tuple[str, tuple[int, ...]], dict[str, list[onnx_ir.Value]]] = {}

        name_map = {}  # Duplicate name → canonical name

        for initializer in list(graph.initializers.values()):
            const_val = initializer.const_value
            if const_val is None:
                continue  # Skip if initializer has no constant value
            dtype = const_val.dtype.name
            shape = tuple(int(dim) if isinstance(dim, int) else -1 for dim in const_val.shape)
            content = const_val.tobytes()
            content_hash = hashlib.sha256(content).hexdigest()

            key = (dtype, shape)
            if key not in seen:
                seen[key] = {}

            group = seen[key]
            if content_hash in group:
                for existing_val in group[content_hash]:
                    if (
                        existing_val.const_value is not None
                        and existing_val.const_value.tobytes() == content
                    ):
                        assert initializer.name is not None
                        name_map[initializer.name] = existing_val.name
                        graph.initializers.pop(initializer.name)
                        break  # only break when deduplication is successful
                    else:
                        # no matching content found: append as a new entry
                        assert initializer.name is not None
                        group[content_hash].append(initializer)
            else:
                assert initializer.name is not None
                group[content_hash] = [initializer]

        for node in onnx_ir.traversal.RecursiveGraphIterator(graph):
            for i, input_val in enumerate(node.inputs):
                if input_val and input_val.name in name_map:
                    canonical_name = name_map[input_val.name]
                    if canonical_name is not None:
                        replacement = graph.initializers[canonical_name]
                        node.replace_input_with(i, replacement)

        return onnx_ir.passes.PassResult(model=model, modified=bool(name_map))
