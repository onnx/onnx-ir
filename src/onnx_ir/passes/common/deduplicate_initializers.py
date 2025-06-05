from onnx_ir import ir
from onnx_ir.passes.base import GraphTransformPass


class DeduplicateInitializersPass(GraphTransformPass):
    """
    This pass removes duplicate initializer tensors from the graph.

    It identifies duplicates based on a content-based fingerprint consisting of:
    - Tensor byte content (`tobytes()`)
    - Data type (`dtype`)
    - Shape

    All duplicates are replaced with the first (canonical) occurrence, and node
    inputs referring to redundant initializers are updated accordingly.
    """

    def apply(self, graph: ir.Graph) -> ir.Graph:
        seen = {}      # Maps (tobytes, dtype, shape) -> canonical initializer name
        name_map = {}  # Maps duplicate initializer name -> canonical name

        # Iterate over all initializers in the graph
        for initializer in list(graph.initializers.values()):
            key = (
                initializer.const_value.tobytes(),              # Content fingerprint
                initializer.const_value.dtype,                  # Data type
                tuple(initializer.const_value.shape),           # Shape tuple
            )

            if key in seen:
                # Found a duplicate: store the name mapping and remove it from graph
                canonical_name = seen[key]
                name_map[initializer.name] = canonical_name
                graph.initializers.pop(initializer.name)
            else:
                # First time seeing this tensor → keep it
                seen[key] = initializer.name

        # Update node inputs to use the canonical initializer names
        for node in graph:
            for i, input_value in enumerate(node.inputs):
                if input_value is not None and input_value.name in name_map:
                    # Replace input with the deduplicated initializer
                    new_name = name_map[input_value.name]
                    replacement = graph.initializers[new_name]
                    node.replace_input_with(i, replacement)

        return graph

