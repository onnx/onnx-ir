from onnx_ir import ir
from onnx_ir.passes.base import GraphTransformPass
from onnx_ir.traversal import iterate_graph


class DeduplicateInitializersPass(GraphTransformPass):
    """
    Graph transformation pass to remove duplicate initializer tensors.

    Identifies duplicates based on:
    - Data type
    - Shape
    - Byte content (used only if dtype and shape match)

    Updates all node inputs (including subgraphs) to refer to the canonical tensor.
    """

    def apply(self, graph: ir.Graph) -> ir.Graph:
        seen = {}      # (dtype, shape) → {tobytes: name}
        name_map = {}  # Duplicate name → canonical name

        # Iterate through initializers and group by dtype and shape first
        for initializer in list(graph.initializers.values()):
            dtype = initializer.const_value.dtype
            shape = tuple(initializer.const_value.shape)
            content = initializer.const_value.tobytes()

            if (dtype, shape) not in seen:
                seen[(dtype, shape)] = {}

            group = seen[(dtype, shape)]
            if content in group:
                # Duplicate found
                canonical_name = group[content]
                name_map[initializer.name] = canonical_name
                graph.initializers.pop(initializer.name)
            else:
                group[content] = initializer.name

        # Update all node inputs (including subgraphs)
        for node in iterate_graph(graph):
            for i, input_value in enumerate(node.inputs):
                if input_value is not None and input_value.name in name_map:
                    canonical = name_map[input_value.name]
                    replacement = graph.initializers[canonical]
                    node.replace_input_with(i, replacement)

        return graph
