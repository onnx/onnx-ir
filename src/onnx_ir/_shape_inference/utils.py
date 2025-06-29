# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def broadcast_shapes(shapes):
    """Broadcasts a list of shapes to a single shape."""
    # Find the maximum rank
    max_rank = 0
    for shape in shapes:
        max_rank = max(max_rank, len(shape))

    # Pad the shapes with 1s to the maximum rank
    padded_shapes = []
    for shape in shapes:
        padded_shapes.append([1] * (max_rank - len(shape)) + list(shape))

    # Broadcast the shapes
    output_shape = [1] * max_rank
    for i in range(max_rank):
        for shape in padded_shapes:
            if shape[i] != 1:
                if output_shape[i] == 1:
                    output_shape[i] = shape[i]
                elif output_shape[i] != shape[i]:
                    raise ValueError("Incompatible shapes for broadcasting")

    return output_shape
