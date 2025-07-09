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
from fastdeploy.model_executor.layers.attention.attention import Attention_HPU
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear_hpu import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import LayerNorm, RMSNorm
from fastdeploy.worker.forward_meta import ForwardMeta_HPU

from fastdeploy.model_executor.ops.intel_hpu import \
            fused_rms_mlp_res, rebuild_padding_v2

from .model_base import ModelForCasualLM

import pdb


class Qwen2MLP_HPU(nn.Layer):
    """
    fused RmsNormMLP_HPU Layer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        from fastdeploy.model_executor.layers.normalization import RMSNorm
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.post_attention_layernorm",
        )

        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.mlp.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.ffn_hidden_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.mlp.down_proj",
            input_size=fd_config.model_config.ffn_hidden_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )
        
    def load_state_dict(self, state_dict):
        self.post_attention_layernorm.load_state_dict(state_dict)
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)
        
    def forward(self, 
                hidden_states: paddle.Tensor,
                residual: paddle.Tensor = None,
    ):
        out = fused_rms_mlp_res(
            hidden_states,
            self.post_attention_layernorm.ln_weight,
            self.gate_up_proj.linear_weight,
            self.down_proj.linear_weight,
            residual,
            epsilon=1e-6,
        )

        # all_reduce
        if self.nranks > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(out)

        return out, residual
    

class Qwen2Attention_HPU(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        nranks = fd_config.parallel_config.tensor_parallel_degree

        self.attn = Attention_HPU(
            fd_config=fd_config,
            layer_id=layer_id,
            with_bias=True,
            eps=1e-6,
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


class Qwen2DecoderLayer_HPU(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Qwen2Attention_HPU(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}",
        )

        if (fd_config.moe_config.num_experts is not None
                and layer_id >= fd_config.moe_config.moe_layer_start_index):
            self.mlp = FusedMoE(
                fd_config=fd_config,
                moe_intermediate_size=fd_config.moe_config.
                moe_intermediate_size,
                num_experts=fd_config.moe_config.num_experts,
                top_k=fd_config.moe_config.top_k,
                moe_use_gate_correction_bias=fd_config.moe_config.
                moe_use_gate_correction_bias,
                moe_quant_type=fd_config.moe_config.moe_quant_type,
                layer_idx=layer_id,
                gate_weight_key=f"{prefix}.mlp.gate.weight",
                gate_correction_bias_key=
                f"{prefix}.mlp.moe_statics.e_score_correction_bias",
                ffn1_expert_weight_key=
                f"{prefix}.mlp.experts.{{}}.up_gate_proj.weight",
                ffn2_expert_weight_key=
                f"{prefix}.mlp.experts.{{}}.down_proj.weight",
                prefix=prefix,
            )
        else:
            self.mlp = Qwen2MLP_HPU(
                fd_config=fd_config,
                prefix=f"{prefix}",
            )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
        residual_input: paddle.Tensor = None,
    ):
        hidden_states, residual = self.self_attn(
            hidden_states=hidden_states,
            residual_input=residual_input,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.mlp(hidden_states, residual)

        return hidden_states, residual


class Qwen2Model_HPU(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Qwen2Model class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "qwen2"

        self.embeddings = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"),
        )

        self.hidden_layers = [
            Qwen2DecoderLayer_HPU(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ]

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-6,
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
            hidden_states, residual = self.hidden_layers[i](forward_meta,
                                                            hidden_states,
                                                            residual)

        print("************* hidden_states:", hidden_states)
        print("************* residual:", residual)
        hidden_states = hidden_states + residual

        out = rebuild_padding_v2(
            hidden_states,
            forward_meta.batch_ids,
            forward_meta.seq_lens_encoder,
            forward_meta.is_prompt,
        )
        out = self.norm(out)
        print("************* out:", out)

        return out


