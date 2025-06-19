#!/usr/bin/env python3
"""Example script demonstrating migration from onnx.helper to onnx_ir.

This script creates the same model using both approaches to show the differences.
Run this script to see the migration in action.
"""

import numpy as np


def create_model_with_onnx_helper():
    """Create a simple model using onnx.helper (old approach)."""
    try:
        import onnx  # noqa: TID251
        from onnx import helper, TensorProto  # noqa: TID251
    except ImportError:
        print("onnx package not installed, skipping onnx.helper example")
        return None

    print("=== Creating model with onnx.helper ===")

    # Create input
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 3, 224, 224]
    )

    # Create weight tensor
    weight_data = np.random.randn(64, 3, 7, 7).astype(np.float32)
    weight_tensor = helper.make_tensor(
        'conv_weight',
        TensorProto.FLOAT,
        [64, 3, 7, 7],
        weight_data.flatten().tolist()
    )

    # Create output
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, 64, 112, 112]
    )

    # Create nodes
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv_weight'],
        outputs=['conv_output'],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
        name='conv1'
    )

    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_output'],
        outputs=['output'],
        name='relu1'
    )

    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_cnn',
        [input_tensor],
        [output_tensor],
        [weight_tensor]
    )

    # Create model
    model = helper.make_model(graph, producer_name='migration_example')

    print(f"Model created with {len(graph.node)} nodes")
    print(f"Input shape: {[dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]}")
    print(f"Output shape: {[dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]}")

    return model


def create_model_with_onnx_ir():
    """Create the same model using onnx_ir (new approach)."""
    try:
        import onnx_ir as ir
    except ImportError:
        print("onnx_ir package not installed, skipping onnx_ir example")
        return None

    print("\n=== Creating model with onnx_ir ===")

    # Create input and output values
    input_value = ir.Value(
        name='input',
        shape=ir.Shape([1, 3, 224, 224]),
        type=ir.TensorType(ir.DataType.FLOAT)
    )
    output_value = ir.Value(
        name='output',
        shape=ir.Shape([1, 64, 112, 112]),
        type=ir.TensorType(ir.DataType.FLOAT)
    )

    # Create weight tensor (much simpler!)
    weight_data = np.random.randn(64, 3, 7, 7).astype(np.float32)
    weight_tensor = ir.tensor(weight_data, name='conv_weight')
    weight_value = ir.Value(name='conv_weight', const_value=weight_tensor)

    # Create nodes with direct value connections
    conv_node = ir.node(
        'Conv',
        inputs=[input_value, weight_value],
        attributes={
            'kernel_shape': [7, 7],
            'strides': [2, 2],
            'pads': [3, 3, 3, 3]
        },
        name='conv1'
    )

    relu_node = ir.node(
        'Relu',
        inputs=[conv_node.outputs[0]],
        name='relu1'
    )

    # Create graph
    graph = ir.Graph(
        inputs=[input_value],
        outputs=[relu_node.outputs[0]],
        nodes=[conv_node, relu_node],
        initializers=[weight_value],
        name='simple_cnn'
    )

    # Create model
    model = ir.Model(graph, ir_version=8, producer_name='migration_example')

    print(f"Model created with {len(graph)} nodes")
    if input_value.shape:
        print(f"Input shape: {input_value.shape.dims}")
    if relu_node.outputs[0].shape:
        print(f"Output shape: {relu_node.outputs[0].shape.dims}")

    return model


def compare_models():
    """Compare the two approaches."""
    print("\n=== Comparison ===")
    print("onnx.helper approach:")
    print("  - More verbose tensor creation")
    print("  - Manual protobuf manipulation")
    print("  - String-based connections between nodes")
    print("  - Manual type and shape specification")

    print("\nonnx_ir approach:")
    print("  - Natural numpy array handling")
    print("  - Pythonic object model")
    print("  - Direct value-based connections")
    print("  - Type-safe operations")
    print("  - Better error messages")
    print("  - More maintainable code")


def main():
    """Run the migration example."""
    print("ONNX Model Creation: onnx.helper vs onnx_ir")
    print("=" * 50)

    # Create models with both approaches
    helper_model = create_model_with_onnx_helper()
    ir_model = create_model_with_onnx_ir()

    # Show the comparison
    compare_models()

    # Optional: Save models for inspection
    if helper_model is not None:
        try:
            import onnx  # noqa: TID251
            onnx.save(helper_model, 'model_helper.onnx')
            print("\nSaved onnx.helper model to: model_helper.onnx")
        except Exception as e:
            print(f"Could not save helper model: {e}")

    if ir_model is not None:
        try:
            import onnx_ir as ir
            ir.save(ir_model, 'model_ir.onnx')
            print("Saved onnx_ir model to: model_ir.onnx")
        except Exception as e:
            print(f"Could not save ir model: {e}")

    print("\n🎉 Migration example completed!")
    print("Both approaches create functionally equivalent models,")
    print("but onnx_ir provides a much better development experience.")


if __name__ == '__main__':
    main()
