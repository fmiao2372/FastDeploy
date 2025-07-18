"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from fastdeploy.utils import console_logger as logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import LayerNorm, RMSNorm
from fastdeploy.worker.forward_meta import ForwardMeta_HPU

from fastdeploy.model_executor.ops.intel_hpu import \
            fused_mlp, rebuild_padding_v2

from .model_base import ModelForCasualLM

import pdb


class Ernie45TMLP_HPU(nn.Layer):
    """
    fused RmsNormMLP Layer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.nranks = fd_config.parallel_config.tensor_parallel_degree

        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

    def load_state_dict(self, state_dict):
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)
        
    def forward(self, hidden_states: paddle.Tensor):
        out = fused_mlp(
            hidden_states,
            self.gate_up_proj.linear_weight,
            None,
            self.down_proj.linear_weight,
        )

        # all_reduce
        if self.nranks > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(out)

        return out
    

class Ernie45TAttention_HPU(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            eps=1e-5,
            prefix=prefix,
        )

    def load_state_dict(self, state_dict):

        self.attn.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
        residual_input: paddle.Tensor,
    ):
        attn_out, residual = self.attn(
            src=hidden_states,
            residual_input = residual_input,
            forward_meta=forward_meta,
        )

        return attn_out, residual

class Ernie4_5_MoE(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, 'moe_quant_type'):
            moe_quant_type = fd_config.quant_config.moe_quant_type

        if moe_quant_type == "w4a8":
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_in_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "ffn2_expert_in_scale_key":
                f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4w2":
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_super_scales_key":
                f"{prefix}.experts.{{}}.up_gate_proj.super_scales",
                "ffn2_expert_super_scales_key":
                f"{prefix}.experts.{{}}.down_proj.super_scales",
                "ffn1_expert_code_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.code_scale",
                "ffn2_expert_code_scale_key":
                f"{prefix}.experts.{{}}.down_proj.code_scale",
                "ffn1_expert_code_zp_key":
                f"{prefix}.experts.{{}}.up_gate_proj.code_zp",
                "ffn2_expert_code_zp_key":
                f"{prefix}.experts.{{}}.down_proj.code_zp",
            }
        elif moe_quant_type == "tensor_wise_fp8" or (
                moe_quant_type == "block_wise_fp8"
                and fd_config.model_config.is_quantized):
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_in_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "ffn2_expert_in_scale_key":
                f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        else:
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.weight",
            }

        self.fused_moe = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.moe_config.moe_intermediate_size,
            num_experts=fd_config.moe_config.num_experts,
            top_k=fd_config.moe_config.top_k,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

        self.num_shared_experts = fd_config.moe_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            shared_experts_hidden_dim = self.num_shared_experts * fd_config.moe_config.moe_intermediate_size
            self.shared_experts = Ernie45TMLP_HPU(
                fd_config=fd_config,
                intermediate_size=shared_experts_hidden_dim,
                prefix=f"{prefix}.shared_experts",
            )

    def load_state_dict(self, state_dict):
        self.fused_moe.load_state_dict(state_dict)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        out = self.fused_moe(hidden_states)
        if self.num_shared_experts > 0:
            s_x = self.shared_experts(hidden_states)
            out = out + s_x
        return out

class Ernie45TDecoderLayer_HPU(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Ernie45TAttention_HPU(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}",
        )

        if (fd_config.moe_config.num_experts is not None
                and layer_id >= fd_config.moe_config.moe_layer_start_index):
            self.mlp = Ernie4_5_MoE(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie45TMLP_HPU(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.ffn_hidden_size,
                prefix=f"{prefix}.mlp",
            )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
        residual_input: paddle.Tensor = None,
    ):
        logger.info(f"start forward layer attention")
        hidden_states, residual = self.self_attn(
            hidden_states=hidden_states,
            residual_input=residual_input,
            forward_meta=forward_meta,
        )

        hidden_states = hidden_states + residual
        residual = hidden_states

        batch, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.reshape([batch, -1, hidden_dim])
        logger.info(f"start forward layer mlp/moe")
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Ernie45TModel_HPU(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie45TModel class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "ernie"

        self.embeddings = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"),
        )

        self.hidden_layers = nn.LayerList([
            Ernie45TDecoderLayer_HPU(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ])

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{fd_config.model_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.embeddings.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta_HPU,
    ):
        """
        """

        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(axis=1)

        residual = paddle.zeros_like(hidden_states, hidden_states.dtype)
        for i in range(self.num_layers):
            logger.info(f"start forward layer {i}")
            hidden_states, residual = self.hidden_layers[i](forward_meta,
                                                            hidden_states,
                                                            residual)

        hidden_states = hidden_states + residual

        out = rebuild_padding_v2(
            hidden_states,
            forward_meta.batch_ids,
            forward_meta.seq_lens_encoder,
            forward_meta.is_prompt,
        )
        out = self.norm(out)

        return out


