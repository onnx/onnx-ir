# ONNX IR Concepts

The onnx_ir package provides a set of classes and functions to work with the ONNX Intermediate Representation. It follows closely with the ONNX protobuf definitions to provide a familiar experience, with a few additions to improve usability.

## Core Data Structures

### Nodes and Values

The computation graph in ONNX is represented by two types of vertices: {py:class}`Node <onnx_ir.Node>` and {py:class}`Value <onnx_ir.Value>`. In ONNX proto, values are represented as strings in NodeProto inputs and outputs. An additional `ValueInfoProto` list is used to store shape and type information about these values.

In `onnx_ir`, we have introduced the `Value` class to unify the representation of values in the graph. Instead of a name string, each `Value` object represents a distinct value in the computation graph.

### Model

{py:class}`Model <onnx_ir.Model>` is the top-level container for an ONNX model. It contains the model's metadata, the main graph and definitions for model local functions.

```{eval-rst}
.. note::
    Whereas {py:class}`onnx.ModelProto <onnx.ModelProto>` stores imported opsets, onnx_ir stores them in the {py:class}`Graph <onnx_ir.Graph>` class.
```

You may initialize a model as follows:

```python
import onnx_ir as ir

model = ir.Model(
    graph,  # An ir.Graph object
    ir_version=10,
    opset_imports={
        "": 23,  # Default ONNX opset
        "my_domain": 1  # Custom domain opset
    },
    ... # Other model fields
)
```

### Graph

The {py:class}`Graph <onnx_ir.Graph>` closely resembles the ONNX `GraphProto` structure, and implements the `Sequence[Node]` interface to support Pythonic iteration.

