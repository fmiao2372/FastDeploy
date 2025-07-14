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

from __future__ import annotations

from functools import partial

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
# from fastdeploy.model_executor.graph_optimization.decorator import \
#     support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention_HPU
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear_hpu import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.worker.forward_meta import ForwardMeta_HPU
from fastdeploy.model_executor.ops.intel_hpu import fused_mlp
from fastdeploy.worker.forward_meta import ForwardMeta


class Qwen2MLP_HPU(nn.Layer):
    """
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.ffn_hidden_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=fd_config.model_config.ffn_hidden_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=getattr(self.gate_up_proj, "linear_bias", None),
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x):
        """
        """
        out = fused_mlp(
            x,
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
    

class Qwen2Attention_HPU(nn.Layer):
    """
    """

    def __init__(self,
                 fd_config: FDConfig,
                 layer_id: int,
                 prefix: str = "") -> None:
        super().__init__()


        self.attn = Attention_HPU(
            fd_config=fd_config,
            layer_id=layer_id,
            with_bias=True,
            prefix=prefix,
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.attn.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
    ):
        """
        """
        atten_out = self.attn(
            src=hidden_states,
            forward_meta=forward_meta,
        )

        return atten_out


class Qwen2DecoderLayer_HPU(nn.Layer):
    """
    """

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
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Qwen2MLP_HPU(
            fd_config=fd_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        """
        """
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# @support_graph_optimization
class Qwen2Model_HPU(nn.Layer):
    """
    """

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

        self.layers = nn.LayerList([
            Qwen2DecoderLayer_HPU(
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
            self.layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta_HPU,
    ):
        """
        """

        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta,
                                                     hidden_states, residual)

        hidden_states = hidden_states + residual

        out = self.norm(hidden_states)

        return out


