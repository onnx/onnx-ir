# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, List, Optional, Union
import onnx_ir as ir


class ShapeEnv:
    """Manages symbolic dimensions and inferred shapes/types for values."""

    def __init__(self):
        # Stores symbolic expressions for dimensions, mapping string dim_param to sympy.Symbol or int
        self._symbolic_dims: Dict[str, Any] = {}
        # Stores inferred shapes and types for ir.Value objects
        self._value_info: Dict[ir.Value, ir.Type] = {}

    def set_shape_and_type(self, value: ir.Value, shape: ir.Shape, dtype: ir.DataType):
        """Sets the inferred shape and type for a given ir.Value."""
        value.shape = shape
        value.dtype = dtype
        self._value_info[value] = ir.TensorType(dtype, shape)

    def get_shape(self, value: ir.Value) -> Optional[ir.Shape]:
        """Gets the inferred shape for a given ir.Value."""
        return self._value_info.get(value).shape if value in self._value_info else None

    def get_dtype(self, value: ir.Value) -> Optional[ir.DataType]:
        """Gets the inferred data type for a given ir.Value."""
        return self._value_info.get(value).dtype if value in self._value_info else None

    def get_symbolic_dim(self, dim_param: str) -> Any:
        """Retrieves or creates a symbolic dimension."""
        if dim_param not in self._symbolic_dims:
            # For now, just store the string. If sympy integration is needed, this is where it would happen.
            self._symbolic_dims[dim_param] = dim_param
        return self._symbolic_dims[dim_param]

    def merge_symbolic_dims(self, dim1: str, dim2: str):
        """Merges two symbolic dimensions, asserting equality or setting one to the other."""
        # This is a placeholder for more complex symbolic dimension merging logic.
        # For now, it assumes they must be equal or one is None.
        if dim1 is None:
            self._symbolic_dims[dim2] = dim2
        elif dim2 is None:
            self._symbolic_dims[dim1] = dim1
        elif dim1 != dim2:
            # In a real symbolic inference, this would involve equating sympy symbols
            # For now, we'll just pick one or raise an error if they are conflicting literals
            if isinstance(dim1, int) and isinstance(dim2, int) and dim1 != dim2:
                raise ValueError(f"Conflicting literal dimensions: {dim1} and {dim2}")
            # Prefer the one that's already a symbolic dim, otherwise pick dim1
            if dim2 in self._symbolic_dims and not isinstance(self._symbolic_dims[dim2], int):
                self._symbolic_dims[dim1] = self._symbolic_dims[dim2]
            else:
                self._symbolic_dims[dim2] = self._symbolic_dims[dim1]


