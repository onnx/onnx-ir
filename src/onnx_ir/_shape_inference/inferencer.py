# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import onnx_ir as ir
from onnx_ir._shape_inference.ops import (
    add,
    array_feature_extractor,
    aten_argmax,
    aten_bitwise_or,
    aten_diagonal,
    aten_group_norm,
    aten_multinomial,
    aten_pool2d,
    aten_unfold,
    aten_upsample,
    batch_normalization,
    cast,
    category_mapper,
    compress,
    concat,
    concat_from_sequence,
    constant,
    constant_of_shape,
    conv,
    dequantize_linear,
    div,
    einsum,
    expand,
    gather,
    gather_elements,
    gather_nd,
    if_op,
    identity,
    loop,
    matmul,
    matmul_integer16,
    mul,
    non_max_suppression,
    non_zero,
    one_hot,
    pad,
    passthrough,
    pool,
    quantize_linear,
    range_op,
    reduce_prod,
    reduce_sum,
    relative_position_bias,
    reshape,
    resize,
    scan,
    scatter_elements,
    sequence_at,
    sequence_insert,
    shape,
    slice,
    softmax_cross_entropy_loss,
    split,
    split_to_sequence,
    squeeze,
    sub,
    size,
    symbolic_compute,
    tile,
    top_k,
    transpose,
    unsqueeze,
    zip_map,
)
from onnx_ir._shape_inference.result import InferenceStatus

logger = logging.getLogger(__name__)


class ShapeInferenceEngine:
    def __init__(self, graph: ir.Graph):
        self._graph = graph
        self._dispatcher = {
            "Add": add.infer_add_shape,
            "Cast": cast.infer_cast_shape,
            "Concat": concat.infer_concat_shape,
            "Constant": constant.infer_constant_shape,
            "Div": div.infer_div_shape,
            "Gather": gather.infer_gather_shape,
            "Identity": identity.infer_identity_shape,
            "MatMul": matmul.infer_matmul_shape,
            "Mul": mul.infer_mul_shape,
            "Reshape": reshape.infer_reshape_shape,
            "Squeeze": squeeze.infer_squeeze_shape,
            "Sub": sub.infer_sub_shape,
            "Transpose": transpose.infer_transpose_shape,
            "Unsqueeze": unsqueeze.infer_unsqueeze_shape,
            "CumSum": passthrough.infer_passthrough_shape,
            "Reciprocal": passthrough.infer_passthrough_shape,
            "Round": passthrough.infer_passthrough_shape,
            "MemcpyFromHost": passthrough.infer_passthrough_shape,
            "MemcpyToHost": passthrough.infer_passthrough_shape,
            "AllReduce": passthrough.infer_passthrough_shape,
            "MoE": passthrough.infer_passthrough_shape,
            "Shape": shape.infer_shape_shape,
            "Size": size.infer_size_shape,
            "Slice": slice.infer_slice_shape,
            "Split": split.infer_split_shape,
            "Equal": symbolic_compute.infer_symbolic_compute_shape,
            "Floor": symbolic_compute.infer_symbolic_compute_shape,
            "Max": symbolic_compute.infer_symbolic_compute_shape,
            "Min": symbolic_compute.infer_symbolic_compute_shape,
            "Where": symbolic_compute.infer_symbolic_compute_shape,
            "Neg": symbolic_compute.infer_symbolic_compute_shape,
            "AveragePool": pool.infer_pool_shape,
            "MaxPool": pool.infer_pool_shape,
            "BatchNormalization": batch_normalization.infer_batch_normalization_shape,
            "Conv": conv.infer_conv_shape,
            "MatMulInteger16": matmul_integer16.infer_matmul_integer16_shape,
            "Einsum": einsum.infer_einsum_shape,
            "Expand": expand.infer_expand_shape,
            "GatherElements": gather_elements.infer_gather_elements_shape,
            "GatherND": gather_nd.infer_gather_nd_shape,
            "If": if_op.infer_if_shape,
            "Loop": loop.infer_loop_shape,
            "NonMaxSuppression": non_max_suppression.infer_non_max_suppression_shape,
            "NonZero": non_zero.infer_non_zero_shape,
            "OneHot": one_hot.infer_one_hot_shape,
            "Pad": pad.infer_pad_shape,
            "Range": range_op.infer_range_shape,
            "ReduceSum": reduce_sum.infer_reduce_sum_shape,
            "ReduceProd": reduce_prod.infer_reduce_prod_shape,
            "Resize": resize.infer_resize_shape,
            "Scan": scan.infer_scan_shape,
            "ScatterElements": scatter_elements.infer_scatter_elements_shape,
            "SequenceAt": sequence_at.infer_sequence_at_shape,
            "SequenceInsert": sequence_insert.infer_sequence_insert_shape,
            "SoftmaxCrossEntropyLoss": softmax_cross_entropy_loss.infer_softmax_cross_entropy_loss_shape,
            "SoftmaxCrossEntropyLossInternal": softmax_cross_entropy_loss.infer_softmax_cross_entropy_loss_shape,
            "NegativeLogLikelihoodLossInternal": softmax_cross_entropy_loss.infer_softmax_cross_entropy_loss_shape,
            "SplitToSequence": split_to_sequence.infer_split_to_sequence_shape,
            "Tile": tile.infer_tile_shape,
            "TopK": top_k.infer_top_k_shape,
            "ZipMap": zip_map.infer_zip_map_shape,
            "NhwcConv": nhwc_conv.infer_nhwc_conv_shape,
            "DequantizeLinear": dequantize_linear.infer_dequantize_linear_shape,
            "QuantizeLinear": quantize_linear.infer_quantize_linear_shape,
            "RelativePositionBias": relative_position_bias.infer_relative_position_bias_shape,
            "embedding": gather.infer_gather_shape, # Re-use Gather for embedding
            "bitwise_or": aten_bitwise_or.infer_aten_bitwise_or_shape,
            "diagonal": aten_diagonal.infer_aten_diagonal_shape,
            "max_pool2d_with_indices": aten_pool2d.infer_aten_pool2d_shape,
            "max": symbolic_compute.infer_symbolic_compute_shape, # Re-use symbolic_compute for max
            "min": symbolic_compute.infer_symbolic_compute_shape, # Re-use symbolic_compute for min
            "multinomial": aten_multinomial.infer_aten_multinomial_shape,
            "unfold": aten_unfold.infer_aten_unfold_shape,
            "argmax": aten_argmax.infer_aten_argmax_shape,
            "avg_pool2d": aten_pool2d.infer_aten_pool2d_shape,
            "_adaptive_avg_pool2d": aten_pool2d.infer_aten_pool2d_shape,
            "numpy_T": transpose.infer_transpose_shape, # Re-use Transpose for numpy_T
            "native_group_norm": aten_group_norm.infer_aten_group_norm_shape,
            "upsample_nearest1d": aten_upsample.infer_aten_upsample_shape,
            "upsample_nearest2d": aten_upsample.infer_aten_upsample_shape,
            "upsample_nearest3d": aten_upsample.infer_aten_upsample_shape,
            "upsample_bicubic2d": aten_upsample.infer_aten_upsample_shape,
            "Attention": unsupported.infer_unsupported_shape,
            "BiasAdd": unsupported.infer_unsupported_shape,
            "BiasGelu": unsupported.infer_unsupported_shape,
            "BiasSplitGelu": unsupported.infer_unsupported_shape,
            "DecoderMaskedMultiHeadAttention": unsupported.infer_unsupported_shape,
            "EmbedLayerNormalization": unsupported.infer_unsupported_shape,
            "FastGelu": unsupported.infer_unsupported_shape,
            "GatedRelativePositionBias": unsupported.infer_unsupported_shape,
            "Gelu": unsupported.infer_unsupported_shape,
            "GemmFastGelu": unsupported.infer_unsupported_shape,
            "GemmFloat8": unsupported.infer_unsupported_shape,
            "GroupNorm": unsupported.infer_unsupported_shape,
            "SkipGroupNorm": unsupported.infer_unsupported_shape,
            "LayerNormalization": unsupported.infer_unsupported_shape,
            "LongformerAttention": unsupported.infer_unsupported_shape,
            "MultiHeadAttention": unsupported.infer_unsupported_shape,
            "PackedAttention": unsupported.infer_unsupported_shape,
            "PackedMultiHeadAttention": unsupported.infer_unsupported_shape,
            "MultiScaleDeformableAttnTRT": unsupported.infer_unsupported_shape,
            "PythonOp": unsupported.infer_unsupported_shape,
            "QuickGelu": unsupported.infer_unsupported_shape,
            "RotaryEmbedding": unsupported.infer_unsupported_shape,
            "SimplifiedLayerNormalization": unsupported.infer_unsupported_shape,
            "SkipLayerNormalization": unsupported.infer_unsupported_shape,
            "SkipSimplifiedLayerNormalization": unsupported.infer_unsupported_shape,
        }

    def infer_shapes(self) -> None:
        for node in self._graph.nodes:
            if node.op_type in self._dispatcher:
                result = self._dispatcher[node.op_type](node, self._shape_env)
                if result.status != InferenceStatus.SUCCESS:
                    logger.warning(
                        f"Shape inference failed for node {node.name} ({node.op_type}): {result.reason}"
                    )
            # Reconcile information from ShapeEnv to node outputs
            for output_value in node.outputs:
                inferred_shape = self._shape_env.get_shape(output_value)
                inferred_dtype = self._shape_env.get_dtype(output_value)
                if inferred_shape is not None:
                    output_value.shape = inferred_shape
                if inferred_dtype is not None:
                    output_value.dtype = inferred_dtype
            else:
                logger.warning(
                    f"Shape inference not implemented for op type: {node.op_type}"
                )
