"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "/data/disk2/ckpt/Qwen/Qwen2.5-7B-Instruct"
# model_name_or_path = "/data/disk2/ERNIE-4.5-21B-A3B-Paddle"
# model_name_or_path = "/data/ernie_opensource/ERNIE-4.5-300B-A47B-Paddle/"

# 超参设置
sampling_params = SamplingParams(temperature=1.0, max_tokens=32)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, engine_worker_queue_port=8888, num_gpu_blocks_override=1000)
output = llm.generate(prompts="who are you?",
                      use_tqdm=True,
                      sampling_params=sampling_params)

print(output)
