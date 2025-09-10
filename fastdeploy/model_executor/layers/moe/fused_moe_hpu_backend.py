"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle
from paddle import nn

from fastdeploy.distributed.communication_op import \
    tensor_model_parallel_all_reduce_custom
from ..utils import create_and_set_parameter
from .fused_moe_backend_base import MoEMethodBase

class HpuMoEMethod(MoEMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        # bf16
        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)
        stacked_ffn1_weights = paddle.stack(ffn1_weights, axis=0)
        stacked_ffn2_weights = paddle.stack(ffn2_weights, axis=0)
        for idx, weight_tensor in enumerate(
            [stacked_ffn1_weights, stacked_ffn2_weights]):
            weight_name = self.added_weight_attrs[idx]
            create_and_set_parameter(layer, weight_name, weight_tensor)

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError


    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError


    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True
        '''
        weights = paddle.nn.functional.softmax(gate_out, axis=-1)
        if layer.moe_use_gate_correction_bias:
            scores = weights + layer.gate_correction_bias
            _, selected_experts = paddle.topk(scores, layer.top_k, axis=-1)
            routing_weights = paddle.index_sample(weights, selected_experts)
        else:
            routing_weights, selected_experts = paddle.topk(weights, layer.top_k, axis=-1)
        routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)

        common_inputs = (x, selected_experts, routing_weights.cast("bfloat16"))

        common_params = (
            False,  #permuted_weights
            "silu", #activation,
            0,
            layer.num_experts - 1,
        )

        weights = (
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
        )

        fused_moe_out, _ = mixture_of_experts(
            *common_inputs, *weights, *common_params, False
        )

        # if norm_topk_prob:
        #     routing_weights_norm = paddle.sum(routing_weights, axis=-1, keepdim=True).cast("bfloat16")
        #     fused_moe_out = fused_moe_out / routing_weights_norm
        '''
        chunk_size = 64
        from paddlenlp_ops import fused_gate_moe
        fused_moe_out = fused_gate_moe(x, gate_out, layer.gate_correction_bias,
                                       layer.moe_ffn1_weight,
                                       layer.moe_ffn2_weight,
                                       layer.top_k, layer.moe_use_gate_correction_bias,
                                       norm_topk_prob=True,
                                       permuted_weights=False,
                                       activation="silu",
                                       experts_min=layer.expert_id_offset,
                                       experts_max=layer.expert_id_offset+layer.num_local_experts-1,
                                       chunk_size=chunk_size,)

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

        return fused_moe_out


class HpuTensorWiseFP8MoEMethod(HpuMoEMethod):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        # bf16
        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)

        import paddlenlp_ops
        self.quant_fn = paddlenlp_ops.fused_quant
        self.moe_quant_type = "tensor_wise_fp8"

        align_dummy = paddle.zeros([1], dtype=ffn1_weights[0].dtype)
        padding_list = []
        # align to 0x80 (128 bytes) / 2 (bf16) = 64, add 63 padding tensors
        for j in range(63):
            padding_list.append(align_dummy)

        for idx, weights_tensor in enumerate([ffn1_weights, ffn2_weights]):
            weights_name = self.added_weight_attrs[idx]
            scales_name = self.added_scale_attrs[idx]

            weights_list = []
            scales_list = []

            for i in range(layer.num_local_experts):
                # quantize loaded weights
                quant_weight, scale = self.quant_fn(weights_tensor[i])
                weights_list.append(quant_weight)
                scales_list.append(scale)
                scales_list.extend(padding_list)

            quanted_weight = paddle.stack(weights_list, axis=0)
            create_and_set_parameter(layer, weights_name, quanted_weight)

            quanted_weight_scale = paddle.stack(scales_list, axis=0)
            create_and_set_parameter(layer, scales_name, quanted_weight_scale)


    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError


    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError


    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True

        from paddlenlp_ops import fused_gate_moe_fp8
        fused_moe_out = fused_gate_moe_fp8(x, gate_out, layer.gate_correction_bias,
                                           layer.moe_ffn1_weight,
                                           layer.moe_ffn2_weight,
                                           None, # intermediate_hidden_states_scales
                                           layer.moe_ffn1_weight_scale,
                                           layer.moe_ffn2_weight_scale,
                                           layer.top_k, layer.moe_use_gate_correction_bias,
                                           norm_topk_prob=True,
                                           permuted_weights=False,
                                           activation="silu",
                                           experts_min=0,
                                           experts_max=layer.num_experts - 1,)

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

        return fused_moe_out
