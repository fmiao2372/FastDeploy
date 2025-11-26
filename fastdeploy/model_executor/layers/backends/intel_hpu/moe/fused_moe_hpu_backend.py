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

import os
import paddle
from paddle import nn

from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce_custom
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase, UnquantizedFusedMoEMethod
from fastdeploy.model_executor.layers.utils import get_tensor


class HpuMoEMethod(UnquantizedFusedMoEMethod):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle HPU load weight process.
        """
        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            layer.extract_moe_ffn_weights(state_dict)
        )
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)

        layer.up_gate_proj_weight.set_value(stacked_up_gate_proj_weights)
        layer.down_proj_weight.set_value(stacked_down_proj_weights)


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
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True
        chunk_size = int(os.environ.get("HPU_CHUNK_SIZE", 64))
        measurement_mode = getattr(layer, "measurement_mode", False)
        if measurement_mode:
            from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe_ref

            fused_moe_out = fused_gate_moe_ref(
                x,
                gate.weight,
                layer.gate_correction_bias,
                layer.up_gate_proj_weight,
                layer.down_proj_weight,
                layer.top_k,
                norm_topk_prob=True,
                permuted_weights=False,
                activation="silu",
                experts_min=layer.expert_id_offset,
                experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
                chunk_size=chunk_size,
                measurement_mode=True,
                up_gate_act_scale_key=self.up_gate_proj_act_scale_key,
                down_act_scale_key=self.down_proj_expert_act_scale_key,
            )
        else:
            from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe

            fused_moe_out = fused_gate_moe(
                x,
                gate.weight,
                layer.gate_correction_bias,
                layer.up_gate_proj_weight,
                layer.down_proj_weight,
                layer.top_k,
                norm_topk_prob=True,
                permuted_weights=False,
                activation="silu",
                experts_min=layer.expert_id_offset,
                experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
                chunk_size=chunk_size,
            )
        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

        return fused_moe_out


class HpuTensorWiseFP8MoEMethod(HpuMoEMethod):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle HPU process prequanted weights.
        """
        up_gate_proj_weight, down_proj_weight, logical_expert_ids, _ = layer.extract_moe_ffn_weights(state_dict)
        up_gate_proj_weight = [t.view(paddle.float8_e4m3fn) for t in up_gate_proj_weight]
        down_proj_weight = [t.view(paddle.float8_e4m3fn) for t in down_proj_weight]

        def _extract_scale_tensor(key_template, logical_expert_ids):
            result = []
            for i in logical_expert_ids:
                result.append(get_tensor(state_dict.pop(key_template.format(i))))
            return result  # bf16 tensor list

        def _extract_descale_tensor(key_template, logical_expert_ids):
            if key_template.format(0) in state_dict:
                # Extract scale tensors for all logical_expert_ids
                scale_tensors = []
                for i in logical_expert_ids:
                    scale_tensor = get_tensor(state_dict.pop(key_template.format(i)))
                    scale_tensors.append(scale_tensor)
                # Stack all scale tensors into one tensor
                stacked = paddle.stack(scale_tensors)
                reciprocal = 1.0 / stacked
                # Take max over all logical_expert_ids (axis=0)
                max_tensor = paddle.min(reciprocal, axis=0)
                return max_tensor.cast(paddle.get_default_dtype())
            else:
                key = key_template.replace("{}.", "")
                scale_tensor = get_tensor(state_dict.pop(key))
                reciprocal = 1.0 / scale_tensor
                return reciprocal.cast(paddle.get_default_dtype())

        weight_key_map = layer.weight_key_map

        up_gate_proj_expert_weight_scale_key = weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = weight_key_map.get("down_proj_expert_weight_scale_key", None)
        up_gate_proj_expert_in_scale_key = weight_key_map.get("up_gate_proj_expert_in_scale_key", None)
        down_proj_expert_in_scale_key = weight_key_map.get("down_proj_expert_in_scale_key", None)

        up_gate_proj_weight_scale = _extract_scale_tensor(up_gate_proj_expert_weight_scale_key, logical_expert_ids)
        down_proj_weight_scale = _extract_scale_tensor(down_proj_expert_weight_scale_key, logical_expert_ids)
        up_gate_proj_in_scale = _extract_descale_tensor(up_gate_proj_expert_in_scale_key, logical_expert_ids)
        down_proj_in_scale = _extract_scale_tensor(down_proj_expert_in_scale_key, logical_expert_ids)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
            "up_gate_proj_in_scale": up_gate_proj_in_scale,
            "down_proj_in_scale": down_proj_in_scale,
        }
        for name, tensor_list in name_tensor_map.items():
            setattr(layer, name, tensor_list)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        # TODO: split create_parameter from process_loaded_weights
        return NotImplemented

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle HPU load weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)

        from fastdeploy.model_executor.ops.intel_hpu import fused_quant

        self.quant_fn = fused_quant
        self.moe_quant_type = "tensor_wise_fp8"

        for idx, weights_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weights_name = self.added_weight_attrs[idx]
            scales_name = self.added_scale_attrs[idx]

            weights_list = []
            scales_list = []

            for i in range(layer.num_local_experts):
                # quantize loaded weights
                quant_weight, scale = self.quant_fn(weights_tensor[i])
                weights_list.append(quant_weight)
                scales_list.append(scale)

            setattr(layer, weights_name, weights_list)
            setattr(layer, scales_name, scales_list)

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
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True

        chunk_size = 64
        from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe_fp8

        fused_moe_out = fused_gate_moe_fp8(
            x,
            gate.weight,
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            layer.up_gate_proj_in_scale,
            layer.down_proj_in_scale,
            layer.up_gate_proj_weight_scale,
            layer.down_proj_weight_scale,
            layer.top_k,
            norm_topk_prob=True,
            permuted_weights=False,
            activation="silu",
            experts_min=layer.expert_id_offset,
            experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
            chunk_size=chunk_size,
        )

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

        return fused_moe_out
