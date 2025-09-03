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

import paddle
import paddle.distributed as dist


@paddle.jit.marker.unified
def tensor_model_parallel_all_reduce(input_: paddle.Tensor) -> paddle.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if paddle.in_dynamic_mode():
        hcg = dist.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        dist.all_reduce(input_, group=mp_group)
    else:
        dist.all_reduce(input_)

from paddle.distributed.communication import stream
from paddle.distributed.communication.reduce import ReduceOp

def all_reduce(
    tensor,
    op,
    group,
    sync_op: bool = True,
):
    return stream.all_reduce(
        tensor, op=op, group=group, sync_op=sync_op, use_calc_stream=True
    )

@paddle.jit.marker.unified
def tensor_model_parallel_all_reduce_custom(input_: paddle.Tensor) -> paddle.Tensor:
    """All-reduce the input tensor across model parallel group on calc stream."""
    if paddle.in_dynamic_mode():
        hcg = dist.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        all_reduce(input_, op=ReduceOp.SUM, group=mp_group)
    else:
        dist.all_reduce(input_)