"""MatMul operation inferrer for ONNX IR nodes."""

import sys

import onnx_ir as ir
from onnx_ir._shape_type_inference import _common
from onnx_ir._shape_type_inference.ops.standard_ops import broadcast_shapes_bidirectional


class MatMulInferrer(_common.NodeInferrer):
    """Inferrer for MatMul operations."""

    def __init__(self) -> None:
        """Initialize the MatMul inferrer."""
        super().__init__("MatMul", opsets=range(sys.maxsize))

    def infer(self, node: ir.Node) -> _common.InferenceResult:
        """Infer the output shape and type for MatMul operations."""
        if len(node.inputs) != 2:
            return _common.InferenceResult(
                failure=f"MatMul operation must have exactly two inputs, got {len(node.inputs)}."
            )
        if node.inputs[0] is None or node.inputs[1] is None:
            return _common.InferenceResult(failure="MatMul operation inputs cannot be None.")
        if len(node.outputs) != 1:
            return _common.InferenceResult(
                failure=f"MatMul operation must have exactly one output, got {len(node.outputs)}."
            )

        lhs_shape = node.inputs[0].shape
        rhs_shape = node.inputs[1].shape
        if lhs_shape is None or rhs_shape is None:
            return _common.InferenceResult(failure="MatMul input shapes cannot be None.")

        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        if lhs_rank == 0 or rhs_rank == 0:
            return _common.InferenceResult(failure="MatMul inputs cannot be scalars.")

        # Compute output shape based on matrix multiplication rules
        if lhs_rank == 1 and rhs_rank == 1:
            # Vector dot product: (n,) x (n,) -> scalar
            output_shape = ir.Shape([])
        elif lhs_rank == 1:
            # Matrix-vector: (n,) x (..., n, k) -> (..., k)
            output_dims = list(rhs_shape.dims[:-2]) + [rhs_shape.dims[-1]]
            output_shape = ir.Shape(output_dims)
        elif rhs_rank == 1:
            # Vector-matrix: (..., m, n) x (n,) -> (..., m)
            output_dims = list(lhs_shape.dims[:-1])
            output_shape = ir.Shape(output_dims)
        else:
            # Matrix-matrix: (..., m, n) x (..., n, k) -> (..., m, k)
            # Broadcast batch dimensions
            lhs_batch = lhs_shape.dims[:-2]
            rhs_batch = rhs_shape.dims[:-2]
            if lhs_batch and rhs_batch:
                batch_shape = broadcast_shapes_bidirectional(
                    ir.Shape(lhs_batch), ir.Shape(rhs_batch)
                )
                output_dims = list(batch_shape.dims) + [lhs_shape.dims[-2], rhs_shape.dims[-1]]
                output_shape = ir.Shape(output_dims)
            elif lhs_batch:
                output_dims = list(lhs_batch) + [lhs_shape.dims[-2], rhs_shape.dims[-1]]
                output_shape = ir.Shape(output_dims)
            elif rhs_batch:
                output_dims = list(rhs_batch) + [lhs_shape.dims[-2], rhs_shape.dims[-1]]
                output_shape = ir.Shape(output_dims)
            else:
                output_dims = [lhs_shape.dims[-2], rhs_shape.dims[-1]]
                output_shape = ir.Shape(output_dims)

        # For symbolic inference, we assume dimensions are compatible
        # In practice, this would need runtime verification

        output_type = node.inputs[0].type
        return _common.InferenceResult(
            values=(ir.Value(shape=output_shape, type=output_type),)
        )
