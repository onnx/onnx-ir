"""Reshape operation inferrer for ONNX IR nodes."""

import sys
from collections.abc import Collection

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common


class ReshapeInferrer(_common.NodeInferrer):
    """Inferrer for Reshape operations."""

    def __init__(self, opsets: Collection[int] | None = None) -> None:
        """Initialize the Reshape inferrer."""
        if opsets is None:
            opsets = range(sys.maxsize)
        super().__init__("Reshape", opsets=opsets)

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for Reshape operations."""
        if len(node.inputs) != 2:
            return _common.InferenceResult(
                failure=f"Reshape operation must have exactly two inputs, got {len(node.inputs)}."
            )
        if node.inputs[0] is None or node.inputs[1] is None:
            return _common.InferenceResult(failure="Reshape operation inputs cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"Reshape operation must have exactly one output, got {len(node.outputs)}."
            )

        input_shape = node.inputs[0].shape
        shape_input = node.inputs[1]

        if input_shape is None:
            return _common.InferenceResult(failure="Reshape input shape cannot be None.")

        # Try to get the shape values from the second input
        # For symbolic inference, we may not have concrete values
        if (
            hasattr(shape_input, "initializer_value")
            and shape_input.initializer_value is not None
        ):
            shape_values = shape_input.initializer_value.tolist()
            return self._infer_with_shape_values(
                input_shape, shape_values, node.inputs[0].type
            )
        else:
            # Handle symbolic case where shape is not known at compile time
            shape_shape = shape_input.shape
            if shape_shape is None or len(shape_shape) != 1:
                return _common.InferenceResult(
                    failure="Reshape shape input must be a 1D tensor."
                )

            shape_rank = shape_shape[0]
            if isinstance(shape_rank, int):
                # Create symbolic dimensions for the output
                output_shape = ir.Shape([])
                for i in range(shape_rank):
                    output_shape.append(ir.SymbolicDim(f"reshape_dim_{i}"))

                return _common.InferenceResult(
                    values=(ir.Value(shape=output_shape, type=node.inputs[0].type),)
                )
            else:
                return _common.InferenceResult(
                    failure="Cannot infer reshape output shape with symbolic rank."
                )

    def _infer_with_shape_values(
        self, input_shape: ir.Shape, shape_values: list, input_type
    ) -> _common.InferenceResult:
        """Infer output shape when shape values are known."""
        # Calculate total elements in input
        total_elements = sympy.Integer(1)
        for dim in input_shape:
            if isinstance(dim, ir.SymbolicDim):
                if dim.expr is not None:
                    total_elements *= dim.expr
                else:
                    total_elements *= sympy.Symbol(dim.value)
            else:
                total_elements *= sympy.Integer(dim)

        # Process shape values
        output_dims = []
        deferred_dim_idx = -1
        non_deferred_size = sympy.Integer(1)

        for i, dim_value in enumerate(shape_values):
            if dim_value == -1:
                if deferred_dim_idx != -1:
                    return _common.InferenceResult(
                        failure="Reshape can have at most one -1 dimension."
                    )
                deferred_dim_idx = i
                output_dims.append(None)  # Placeholder
            elif dim_value == 0:
                # Copy from input shape
                if i >= len(input_shape):
                    return _common.InferenceResult(
                        failure=f"Cannot copy dimension {i} from input shape of rank {len(input_shape)}."
                    )
                dim_expr = _common.get_expr(input_shape, i)
                output_dims.append(dim_expr)
                non_deferred_size *= dim_expr
            else:
                output_dims.append(sympy.Integer(dim_value))
                non_deferred_size *= sympy.Integer(dim_value)

        # Calculate deferred dimension
        if deferred_dim_idx != -1:
            deferred_dim = total_elements // non_deferred_size
            output_dims[deferred_dim_idx] = deferred_dim

        # Create output shape
        output_shape = ir.Shape([0] * len(output_dims))
        for i, dim_expr in enumerate(output_dims):
            _common.set_expr(output_shape, i, dim_expr)

        return _common.InferenceResult(values=(ir.Value(shape=output_shape, type=input_type),))
