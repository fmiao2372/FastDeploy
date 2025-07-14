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

        self.qkv_proj = QKVParallelLinear(fd_config=fd_config,
                                          prefix=f"{prefix}.qkv_proj",
                                          with_bias=True)

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.hidden_size,
        )

        self.attn = Attention_HPU(fd_config=fd_config,
                              layer_id=layer_id,
                              prefix=prefix,
                              use_neox_rotary_style=True)

    def load_state_dict(self, state_dict):
        """
        """
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta_HPU,
        hidden_states: paddle.Tensor,
    ):
        """
        """
        atten_out = self.attn(
            src=hidden_states,
            qkv_proj = self.qkv_proj,
            o_proj = self.o_proj,
            forward_meta=forward_meta,
        )

        return atten_out

