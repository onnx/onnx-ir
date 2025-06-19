# Migration Guide: From onnx.helper to onnx_ir

This guide helps you migrate from using `onnx.helper` to construct ONNX models to using the modern `onnx_ir` library. The `onnx_ir` library provides a more Pythonic and intuitive API for creating, manipulating, and analyzing ONNX models.

## Why Migrate to onnx_ir?

- **More Pythonic API**: Natural Python objects instead of protobuf manipulation
- **Better Performance**: Zero-copy tensor handling and memory-mapped external data
- **Robust Graph Manipulation**: Safe mutation with iterator stability
- **Type Safety**: Better type hints and validation
- **Modern Features**: Support for latest ONNX features and data types
- **No Protobuf Dependency**: Work with models without protobuf once loaded

## Quick Comparison

### Creating a Simple Model

**onnx.helper approach:**

```python
import onnx
from onnx import helper, TensorProto

# Create input
input_tensor = helper.make_tensor_value_info(
    'input', TensorProto.FLOAT, [1, 3, 224, 224]
)

# Create output
output_tensor = helper.make_tensor_value_info(
    'output', TensorProto.FLOAT, [1, 1000]
)

# Create a node
node = helper.make_node(
    'Relu',
    inputs=['input'],
    outputs=['relu_output'],
    name='relu_node'
)

# Create graph
graph = helper.make_graph(
    [node],
    'simple_model',
    [input_tensor],
    [output_tensor]
)

# Create model
model = helper.make_model(graph)
```

**onnx_ir approach:**

```python
import onnx_ir as ir

# Create input and output values
input_value = ir.Value('input', shape=ir.Shape([1, 3, 224, 224]),
                      type=ir.TensorType(ir.DataType.FLOAT))
output_value = ir.Value('output', shape=ir.Shape([1, 1000]),
                       type=ir.TensorType(ir.DataType.FLOAT))

# Create a node
relu_node = ir.node('Relu', inputs=[input_value], name='relu_node')

# Create graph with all components
graph = ir.Graph(
    inputs=[input_value],
    outputs=[relu_node.outputs[0]],
    nodes=[relu_node],
    name='simple_model'
)

# Create model
model = ir.Model(graph, ir_version=8)
```

## Core Concepts Mapping

### 1. Tensors and Values

| onnx.helper | onnx_ir | Notes |
|-------------|---------|-------|
| `helper.make_tensor()` | `ir.tensor()` | More flexible, supports various input types |
| `helper.make_tensor_value_info()` | `ir.Input()` or `ir.Value()` | Type-safe value creation |
| Manual shape/type handling | `ir.Shape()`, `ir.TensorType()` | Dedicated objects for shapes and types |

**Examples:**

```python
# onnx.helper
tensor = helper.make_tensor(
    'weights', TensorProto.FLOAT, [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
)

# onnx_ir
import numpy as np
weights_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
tensor = ir.tensor(weights_array, name='weights')
```

### 2. Nodes and Operations

| onnx.helper | onnx_ir | Notes |
|-------------|---------|-------|
| `helper.make_node()` | `ir.node()` | Supports Python objects as attributes |
| Manual attribute creation | Direct Python values | Automatic conversion |
| `helper.make_attribute()` | `ir.AttrFloat32()`, etc. | Type-specific constructors when needed |

**Examples:**

```python
# onnx.helper
conv_node = helper.make_node(
    'Conv',
    inputs=['input', 'weight'],
    outputs=['output'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[1, 1]
)

# onnx_ir
conv_node = ir.node(
    'Conv',
    inputs=[input_value, weight_value],
    attributes={
        'kernel_shape': [3, 3],
        'pads': [1, 1, 1, 1],
        'strides': [1, 1]
    }
)
```

### 3. Graphs and Models

| onnx.helper | onnx_ir | Notes |
|-------------|---------|-------|
| `helper.make_graph()` | `ir.Graph()` | More intuitive construction |
| `helper.make_model()` | `ir.Model()` | Automatic metadata handling |
| Manual initializer handling | Automatic constant folding | Better constant management |

## Step-by-Step Migration

### Step 1: Update Imports

```python
# Old
import onnx
from onnx import helper, TensorProto, AttributeProto

# New
import onnx_ir as ir
import numpy as np
```

### Step 2: Migrate Tensor Creation

```python
# Old: Creating constant tensors
weight_tensor = helper.make_tensor(
    'conv_weight',
    TensorProto.FLOAT,
    [64, 3, 7, 7],
    np.random.randn(64, 3, 7, 7).flatten().tolist()
)

# New: Direct numpy array usage
weight_array = np.random.randn(64, 3, 7, 7).astype(np.float32)
weight_tensor = ir.tensor(weight_array, name='conv_weight')
```

### Step 3: Migrate Value Info Creation

```python
# Old: Input/output value info
input_info = helper.make_tensor_value_info(
    'input', TensorProto.FLOAT, [1, 3, 224, 224]
)
output_info = helper.make_tensor_value_info(
    'output', TensorProto.FLOAT, [1, 1000]
)

# New: Input/output values
input_value = ir.Value(
    'input',
    shape=ir.Shape([1, 3, 224, 224]),
    type=ir.TensorType(ir.DataType.FLOAT)
)
output_value = ir.Value(
    'output',
    shape=ir.Shape([1, 1000]),
    type=ir.TensorType(ir.DataType.FLOAT)
)
```

### Step 4: Migrate Node Creation

```python
# Old: Node with attributes
relu_node = helper.make_node(
    'Relu',
    inputs=['input'],
    outputs=['relu_output']
)

conv_node = helper.make_node(
    'Conv',
    inputs=['input', 'weight'],
    outputs=['conv_output'],
    kernel_shape=[7, 7],
    strides=[2, 2],
    pads=[3, 3, 3, 3]
)

# New: Nodes with type safety
relu_node = ir.node('Relu', inputs=[input_value])

conv_node = ir.node(
    'Conv',
    inputs=[input_value, weight_value],
    attributes={
        'kernel_shape': [7, 7],
        'strides': [2, 2],
        'pads': [3, 3, 3, 3]
    }
)
```

### Step 5: Migrate Graph Construction

```python
# Old: Manual graph assembly
nodes = [conv_node, relu_node]
inputs = [input_info]
outputs = [output_info]
initializers = [weight_tensor]

graph = helper.make_graph(
    nodes, 'my_model', inputs, outputs, initializers
)

# New: Direct graph construction
# Create weight as Value with const_value
weight_value = ir.Value(name='conv_weight', const_value=weight_tensor)

graph = ir.Graph(
    inputs=[input_value],
    outputs=[relu_node.outputs[0]],  # Use actual node output
    nodes=[conv_node, relu_node],
    initializers=[weight_value],
    name='my_model'
)
```

### Step 6: Migrate Model Creation

```python
# Old: Model creation
model = helper.make_model(graph, producer_name='my_producer')
onnx.checker.check_model(model)

# New: Model creation with automatic validation
model = ir.Model(
    graph,
    ir_version=8,
    producer_name='my_producer'
)
```

## Advanced Migration Patterns

### Working with Dynamic Shapes

```python
# Old: Dynamic shapes with onnx.helper
dynamic_input = helper.make_tensor_value_info(
    'input', TensorProto.FLOAT, ['batch', 3, 'height', 'width']
)

# New: Symbolic dimensions in onnx_ir
dynamic_input = ir.Value(
    'input',
    shape=ir.Shape([ir.SymbolicDim('batch'), 3,
                   ir.SymbolicDim('height'), ir.SymbolicDim('width')]),
    type=ir.TensorType(ir.DataType.FLOAT)
)
```

### Complex Attribute Handling

```python
# Old: Complex attributes
attr = helper.make_attribute('scales', [2.0, 2.0])
resize_node = helper.make_node(
    'Resize',
    inputs=['input'],
    outputs=['output'],
    attributes=[attr]
)

# New: Direct Python objects
resize_node = ir.node(
    'Resize',
    inputs=[input_value],
    attributes={'scales': [2.0, 2.0]}
)
```

### Subgraph Handling (If/Loop nodes)

```python
# Old: Manual subgraph creation
then_graph = helper.make_graph(...)
else_graph = helper.make_graph(...)

if_node = helper.make_node(
    'If',
    inputs=['condition'],
    outputs=['output'],
    then_branch=then_graph,
    else_branch=else_graph
)

# New: First-class subgraph support
then_graph = ir.Graph(...)
else_graph = ir.Graph(...)

if_node = ir.node(
    'If',
    inputs=[condition_value],
    attributes={
        'then_branch': then_graph,
        'else_branch': else_graph
    }
)
```

## Common Migration Pitfalls

### 1. Data Type Conversion

```python
# ⚠️ Common mistake: Assuming TensorProto constants
# Old way might use TensorProto.FLOAT
tensor_type = TensorProto.FLOAT

# ✅ Correct: Use onnx_ir DataType enum
tensor_type = ir.DataType.FLOAT
```

### 2. String vs Bytes for String Tensors

```python
# ⚠️ Old way: Manual string encoding
string_data = ['hello', 'world']
encoded = [s.encode('utf-8') for s in string_data]

# ✅ New way: Automatic handling
string_tensor = ir.tensor(['hello', 'world'])  # Automatically encoded
```

### 3. Graph Connectivity

```python
# ⚠️ Old way: Manual name-based connections
node1 = helper.make_node('Op1', inputs=['input'], outputs=['intermediate'])
node2 = helper.make_node('Op2', inputs=['intermediate'], outputs=['output'])

# ✅ New way: Value-based connections
node1 = ir.node('Op1', inputs=[input_value])
node2 = ir.node('Op2', inputs=[node1.outputs[0]])
```

## Serialization and I/O

### Saving Models

```python
# Old: Direct protobuf save
onnx.save(model, 'model.onnx')

# New: High-level save function
ir.save(model, 'model.onnx')
```

### Loading Models

```python
# Old: Load and potentially convert
model_proto = onnx.load('model.onnx')
# Work with protobuf...

# New: Load directly to IR
model = ir.load('model.onnx')
# Work with high-level objects
```

## Best Practices for Migration

1. **Start Small**: Migrate simple models first to get familiar with the API
2. **Use Type Hints**: Take advantage of onnx_ir's better type safety
3. **Leverage Convenience Functions**: Use `ir.tensor()` and `ir.node()` for most cases
4. **Test Thoroughly**: Compare outputs between old and new implementations
5. **Use Modern Python Features**: onnx_ir supports context managers, iterators, etc.

## Key API Differences

### Graph Construction

- **onnx.helper**: Create graph with lists of nodes, inputs, outputs, and initializers
- **onnx_ir**: Create graph with all components specified at construction time

### Value Connections

- **onnx.helper**: String-based connections using names
- **onnx_ir**: Direct object references between values and nodes

### Tensor Creation

- **onnx.helper**: Manual protobuf tensor creation with `make_tensor()`
- **onnx_ir**: Direct numpy array support with `ir.tensor()`

### Type System

- **onnx.helper**: Protobuf enums (`TensorProto.FLOAT`)
- **onnx_ir**: Python enums (`ir.DataType.FLOAT`)

### Graph Modification

- **onnx.helper**: Create new graph for modifications
- **onnx_ir**: In-place graph mutations with safe iteration

## Performance Considerations

- **Memory Usage**: onnx_ir uses lazy loading and memory mapping for better performance
- **Zero-Copy Operations**: Tensors can be shared between frameworks without copying
- **Graph Mutations**: onnx_ir allows safe in-place graph modifications

## Getting Help

- Check the [API documentation](api/index.md) for detailed reference
- Look at the [getting started guide](getting_started.ipynb) for examples
- Refer to the [tensor documentation](tensors.md) for advanced tensor usage
- Run the [migration example script](migration_example.py) to see the differences in action

## Complete Example: ResNet Block Migration

Here's a complete example showing the migration of a ResNet block:

```python
# === OLD: onnx.helper approach ===
import onnx
from onnx import helper, TensorProto
import numpy as np

def create_resnet_block_old():
    # Input
    input_info = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 64, 56, 56]
    )

    # Weights
    conv1_weight = helper.make_tensor(
        'conv1_weight', TensorProto.FLOAT, [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).flatten().tolist()
    )

    conv2_weight = helper.make_tensor(
        'conv2_weight', TensorProto.FLOAT, [64, 64, 3, 3],
        np.random.randn(64, 64, 3, 3).flatten().tolist()
    )

    # Nodes
    conv1 = helper.make_node(
        'Conv', ['input', 'conv1_weight'], ['conv1_out'],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )

    relu1 = helper.make_node('Relu', ['conv1_out'], ['relu1_out'])

    conv2 = helper.make_node(
        'Conv', ['relu1_out', 'conv2_weight'], ['conv2_out'],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )

    add = helper.make_node('Add', ['conv2_out', 'input'], ['add_out'])
    relu2 = helper.make_node('Relu', ['add_out'], ['output'])

    output_info = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 64, 56, 56]
    )

    graph = helper.make_graph(
        [conv1, relu1, conv2, add, relu2],
        'resnet_block',
        [input_info],
        [output_info],
        [conv1_weight, conv2_weight]
    )

    return helper.make_model(graph)

# === NEW: onnx_ir approach ===
import onnx_ir as ir
import numpy as np

def create_resnet_block_new():
    # Create input and output values
    input_value = ir.Value(
        'input',
        shape=ir.Shape([1, 64, 56, 56]),
        type=ir.TensorType(ir.DataType.FLOAT)
    )

    # Create weight tensors and values
    conv1_weight_tensor = ir.tensor(
        np.random.randn(64, 64, 3, 3).astype(np.float32),
        name='conv1_weight'
    )
    conv1_weight_value = ir.Value(name='conv1_weight', const_value=conv1_weight_tensor)

    conv2_weight_tensor = ir.tensor(
        np.random.randn(64, 64, 3, 3).astype(np.float32),
        name='conv2_weight'
    )
    conv2_weight_value = ir.Value(name='conv2_weight', const_value=conv2_weight_tensor)

    # Create nodes with natural connections
    conv1 = ir.node('Conv',
                   inputs=[input_value, conv1_weight_value],
                   attributes={'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1]})

    relu1 = ir.node('Relu', inputs=[conv1.outputs[0]])

    conv2 = ir.node('Conv',
                   inputs=[relu1.outputs[0], conv2_weight_value],
                   attributes={'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1]})

    add = ir.node('Add', inputs=[conv2.outputs[0], input_value])
    relu2 = ir.node('Relu', inputs=[add.outputs[0]])

    # Create graph
    graph = ir.Graph(
        inputs=[input_value],
        outputs=[relu2.outputs[0]],
        nodes=[conv1, relu1, conv2, add, relu2],
        initializers=[conv1_weight_value, conv2_weight_value],
        name='resnet_block'
    )

    return ir.Model(graph, ir_version=8, producer_name='migration_example')

# Both functions create equivalent models, but the onnx_ir version is:
# - More readable and maintainable
# - Type-safe with better error messages
# - Easier to modify and extend
# - More performant for large models
```

This migration guide should help you transition from `onnx.helper` to `onnx_ir` while taking advantage of the modern, Pythonic API and improved performance characteristics.
