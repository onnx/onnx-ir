# Migration Guide: From ONNX Helper APIs to ONNX-IR

This guide helps you transition from using `onnx.helper` APIs to the more powerful and Pythonic ONNX-IR APIs.

## Overview

ONNX-IR provides a modern, mutable, and memory-efficient alternative to the traditional ONNX helper functions. While `onnx.helper` creates immutable protobuf objects, ONNX-IR offers a rich in-memory representation that supports direct manipulation, optimization passes, and seamless integration with ML frameworks.

## Key Differences

| Aspect | onnx.helper | ONNX-IR |
|--------|-------------|---------|
| **Mutability** | Immutable protobuf objects | Fully mutable graphs and nodes |
| **Memory Usage** | Full model in memory | Memory-mapped external tensors |
| **Type Safety** | Basic protobuf validation | Rich type system with inference |
| **Framework Integration** | Manual conversion required | Direct PyTorch/NumPy support |
| **Graph Modification** | Rebuild entire graph | In-place modifications |
| **Optimization** | Manual passes | Built-in pass infrastructure |

## API Migration Map

### Creating Tensors

**Before (onnx.helper):**
```python
import onnx
from onnx import helper, TensorProto
import numpy as np

# Create tensor from numpy array
np_array = np.array([1, 2, 3], dtype=np.float32)
tensor = helper.make_tensor(
    name="my_tensor",
    data_type=TensorProto.FLOAT,
    dims=np_array.shape,
    vals=np_array.flatten()
)
```

**After (ONNX-IR):**
```python
import onnx_ir as ir
import numpy as np

# Create tensor directly from numpy array
np_array = np.array([1, 2, 3], dtype=np.float32)
tensor = ir.tensor(np_array, name="my_tensor")

# Or from PyTorch tensor
import torch
torch_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor = ir.tensor(torch_tensor, name="my_tensor")

# Or from Python list with automatic type inference
tensor = ir.tensor([1, 2, 3], name="my_tensor")
```

### Creating Nodes

**Before (onnx.helper):**
```python
# Create a node with attributes
node = helper.make_node(
    op_type="Conv",
    inputs=["X", "W"],
    outputs=["Y"],
    name="conv_node",
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[1, 1]
)
```

**After (ONNX-IR):**
```python
# Create node with Python-friendly attributes
node = ir.node(
    op_type="Conv",
    inputs=[input_x, weight_w],  # Can use Value objects directly
    outputs=[output_y],
    name="conv_node",
    attributes={
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1]
    }
)

# Or with individual parameters
node = ir.node(
    "Conv", 
    inputs=[input_x, weight_w],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[1, 1],
    name="conv_node"
)
```

### Creating Value Info (Type Definitions)

**Before (onnx.helper):**
```python
# Define input/output shapes and types
input_info = helper.make_tensor_value_info(
    "X", 
    TensorProto.FLOAT, 
    [1, 3, 224, 224]
)
output_info = helper.make_tensor_value_info(
    "Y", 
    TensorProto.FLOAT, 
    [1, 64, 224, 224]
)
```

**After (ONNX-IR):**
```python
# Create inputs with rich type system
input_x = ir.Input(
    name="X",
    type=ir.TensorType(ir.DataType.FLOAT, [1, 3, 224, 224])
)

# Or with automatic type inference from tensors
input_tensor = ir.tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
input_x = ir.Input.from_tensor(input_tensor, name="X")

# Support for symbolic dimensions
input_x = ir.Input(
    name="X",
    type=ir.TensorType(
        ir.DataType.FLOAT, 
        [ir.SymbolicDim("batch"), 3, 224, 224]
    )
)
```

### Creating Graphs

**Before (onnx.helper):**
```python
# Build graph from components
graph = helper.make_graph(
    nodes=[conv_node, relu_node],
    name="simple_cnn",
    inputs=[input_info],
    outputs=[output_info],
    initializer=[weight_tensor, bias_tensor]
)
```

**After (ONNX-IR):**
```python
# Create mutable graph
graph = ir.Graph(name="simple_cnn")

# Add inputs
graph.inputs.append(input_x)

# Add nodes (can modify after creation)
conv_node = graph.add_node("Conv", [input_x, weight_w], kernel_shape=[3, 3])
relu_node = graph.add_node("Relu", [conv_node.outputs[0]])

# Set outputs
graph.outputs.append(relu_node.outputs[0])

# Initializers are handled automatically when creating constant tensors
```

### Creating Models

**Before (onnx.helper):**
```python
# Create model from graph
model = helper.make_model(
    graph,
    producer_name="my_model",
    opset_imports=[helper.make_opsetid("", 11)]
)
```

**After (ONNX-IR):**
```python
# Create model with metadata
model = ir.Model(
    graph=graph,
    producer_name="my_model",
    opset_imports=[("", 11)]
)

# Or let ONNX-IR handle opset automatically
model = ir.Model(graph=graph, producer_name="my_model")
```

## Advanced Features Only Available in ONNX-IR

### 1. Graph Modifications

```python
# Safe iteration and modification
for node in model.graph:
    if node.op_type == "Relu":
        # Replace with LeakyRelu
        new_node = ir.node("LeakyRelu", node.inputs, alpha=0.1)
        node.replace_with(new_node)

# Add nodes in the middle of existing graph
insert_point = model.graph.find_node("conv1")
batch_norm = model.graph.insert_node_after(
    insert_point, 
    "BatchNormalization", 
    [insert_point.outputs[0], scale, bias, mean, var]
)
```

### 2. Optimization Passes

```python
# Apply built-in optimization passes
pass_manager = ir.passes.PassManager()
pass_manager.append(ir.passes.ShapeInferencePass())
pass_manager.append(ir.passes.CommonSubexpressionEliminationPass())
pass_manager.append(ir.passes.RemoveUnusedNodesPass())

# Run optimizations
pass_manager.run(model)
```

### 3. Constant Folding and Analysis

```python
# Extract constant values easily
for node in model.graph:
    for input_val in node.inputs:
        if constant_tensor := ir.convenience.get_const_tensor(input_val):
            print(f"Constant value: {constant_tensor.numpy()}")

# Replace values throughout graph
ir.convenience.replace_all_uses_with(
    old_values=[old_output],
    new_values=[new_output]
)
```

### 4. External Data Handling

```python
# Save large models with external data
ir.save(
    model, 
    "large_model.onnx",
    external_data_path="large_model.bin",
    external_threshold=1024  # Tensors > 1KB stored externally
)

# Load with memory mapping for efficiency
model = ir.load("large_model.onnx", mmap=True)
```

### 5. Framework Integration

```python
# Direct PyTorch integration
import torch

# Use PyTorch tensors directly
torch_weight = torch.randn(64, 3, 3, 3)
ir_tensor = ir.tensor(torch_weight, name="conv_weight")

# Convert between PyTorch and ONNX-IR types
torch_dtype = ir.tensor_adapters.to_torch_dtype(ir.DataType.FLOAT)
ir_dtype = ir.tensor_adapters.from_torch_dtype(torch.float32)
```

## Migration Strategy

### 1. Gradual Migration
Start by replacing model loading/saving:
```python
# Old
model = onnx.load("model.onnx")

# New  
model = ir.load("model.onnx")
```

### 2. Leverage Existing ONNX Models
```python
# Convert existing ONNX models to IR
onnx_model = onnx.load("existing_model.onnx")
ir_model = ir.from_proto(onnx_model)

# Work with IR APIs
# ... modifications ...

# Convert back if needed
final_onnx_model = ir.to_proto(ir_model)
```

### 3. Focus on New Development
Use ONNX-IR for all new graph construction while gradually migrating existing helper-based code.

## Performance Benefits

- **Memory Efficiency**: External data support and memory mapping reduce memory usage by up to 10x for large models
- **Faster Iterations**: Mutable graphs eliminate the need to rebuild entire models during development
- **Zero-Copy Operations**: Direct framework tensor support avoids unnecessary data copying
- **Optimized Analysis**: Built-in graph traversal and analysis utilities are significantly faster than manual protobuf iteration

## Best Practices

1. **Use `ir.tensor()` for all tensor creation** - It handles multiple input types automatically
2. **Leverage the pass system** - Apply optimizations systematically rather than ad-hoc modifications  
3. **Use symbolic dimensions** - Better support for dynamic shapes compared to ONNX helper
4. **Take advantage of mutability** - Modify graphs in-place rather than rebuilding
5. **Use external data for large models** - Keep memory usage manageable

## Common Pitfalls

1. **Don't mix protobuf and IR objects** - Convert fully to IR before manipulation
2. **Remember that IR is mutable** - Changes affect the original object, unlike protobuf
3. **Use appropriate pass ordering** - Some passes depend on others (e.g., shape inference before constant folding)
4. **Handle symbolic dimensions properly** - They require different handling than concrete shapes

This migration guide should help you transition from ONNX helper APIs to the more powerful ONNX-IR system while taking advantage of its advanced features and better performance characteristics.