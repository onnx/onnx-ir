# Models for end-to-end testing

The models under this directory are generated with [tools/create_test_model.py](/tools/create_test_model.py). ONNX models have all initializer data stripped and save as the textproto format.

If a particular test requires the initializer data to be present, it should create random weights based on the tensor dtype/shape and add them to the `TensorProto`s in `model.graph.initializer`.
