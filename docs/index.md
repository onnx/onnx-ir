# ONNX IR

A- **No protobuf dependency**: The IR does not require protobuf once the model is converted to the IR representation, decoupling from the serialization format.

## Migration from onnx.helper

If you're currently using `onnx.helper` to build ONNX models, check out our [migration guide](migration_guide.md) to learn how to transition to the more modern and Pythonic `onnx_ir` API. in-memory IR that supports the full ONNX spec, designed for graph construction, analysis and transformation.

## Features ✨

- Full ONNX spec support: all valid models representable by ONNX protobuf, and a subset of invalid models (so you can load and fix them).
- Low memory footprint: mmap'ed external tensors; unified interface for ONNX TensorProto, Numpy arrays and PyTorch Tensors etc. No tensor size limitation. Zero copies.
- Straightforward access patterns: Access value information and traverse the graph topology at ease.
- Robust mutation: Create as many iterators as you like on the graph while mutating it.
- Speed: Performant graph manipulation, serialization/deserialization to Protobuf.
- Pythonic and familiar APIs: Classes define Pythonic apis and still map to ONNX protobuf concepts in an intuitive way.
- No protobuf dependency: The IR does not require protobuf once the model is converted to the IR representation, decoupling from the serialization format.

## Get started

```{toctree}
:maxdepth: 1

Overview <self>
getting_started
migration_guide
tensors
api/index
```
