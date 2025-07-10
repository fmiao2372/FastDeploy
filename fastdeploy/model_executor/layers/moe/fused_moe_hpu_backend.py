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
from paddle.nn.quant import weight_quantize
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy.distributed.communication_op import \
    tensor_model_parallel_all_reduce
from fastdeploy.platforms import current_platform

from ..utils import create_and_set_parameter, get_tensor
from .fused_moe_backend_base import MoEMethodBase

from fastdeploy.model_executor.ops.intel_hpu import mixture_of_experts

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
        # stacked_ffn1_weights = paddle.stack(ffn1_weights, axis=0)
        # stacked_ffn2_weights = paddle.stack(ffn2_weights, axis=0)
        # for idx, weight_tensor in enumerate(
        #     [stacked_ffn1_weights, stacked_ffn2_weights]):
        #     weight_name = self.added_weight_attrs[idx]
        #     setattr(
        #         layer, weight_name,
        #         layer.create_parameter(
        #             shape=weight_tensor.shape,
        #             dtype=weight_tensor.dtype,
        #             default_initializer=paddle.nn.initializer.Constant(0),
        #         ))
        #     getattr(layer, weight_name).set_value(weight_tensor)

        for idx, weights_tensor in enumerate([ffn1_weights, ffn2_weights]):
            weights_name = self.added_weight_attrs[idx]

            weights_list = []
            for i in range(layer.num_local_experts):
                weight_tensor = weights_tensor[i]
                weight = layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0))
                weight.set_value(weight_tensor)
                weights_list.append(weight)
            setattr(layer, weights_name, weights_list)

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

        norm_topk_prob = False if layer.topk_method == "noaux_tc" else True

        weights = paddle.nn.functional.softmax(gate_out, axis=-1)
        if layer.moe_use_gate_correction_bias:
            scores = weights + layer.gate_correction_bias
            _, selected_experts = paddle.topk(scores, layer.top_k, axis=-1)
            routing_weights = paddle.index_sample(weights, selected_experts)
        else:
            routing_weights, selected_experts = paddle.topk(weights, layer.top_k, axis=-1)


        experts_min = 0
        experts_max = layer.num_experts
        expert_slice = 1
        expert_chunk = max(1, layer.num_experts // expert_slice)

        common_inputs = (x, selected_experts, routing_weights.cast("bfloat16"))
        fused_moe_out = paddle.zeros_like(x)

        for idx in range(expert_slice):
            slice_experts_min = experts_min + (expert_chunk * idx)
            slice_experts_max = min(
                slice_experts_min + expert_chunk, experts_max
            )

            common_params = (
                False,  #permuted_weights
                "silu", #activation,
                slice_experts_min,
                slice_experts_max,
            )
            up_gate_weights = layer.moe_ffn1_weight
            down_weights = layer.moe_ffn2_weight
            slice_weights = (
                up_gate_weights[slice_experts_min : slice_experts_max],
                down_weights[slice_experts_min : slice_experts_max],
            )

            slice_result, _ = mixture_of_experts(
                *common_inputs, *slice_weights, *common_params, False
            )
            fused_moe_out += slice_result

        if norm_topk_prob:
            routing_weights_norm = paddle.sum(routing_weights, axis=-1, keepdim=True).cast("bfloat16")
            fused_moe_out = fused_moe_out / routing_weights_norm

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out