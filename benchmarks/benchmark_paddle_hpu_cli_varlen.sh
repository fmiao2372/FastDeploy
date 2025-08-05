#!/bin/bash

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

# set -x

model="/data/disk3/ernie_opensource/ERNIE-4.5-21B-A3B-Paddle"
model_log_name="ERNIE-4.5-21B-A3B-Paddle"
model_yaml="yaml/eb45-21b-a3b-32k-bf16.yaml"
# model="/data/disk3/ernie_opensource/ERNIE-4.5-300B-A47B-Paddle"
# model_log_name="ERNIE-4.5-300B-A47B-Paddle"
# model_yaml="yaml/eb45-300b-a47b-32k-bf16.yaml"

num_prompts_list=(3 6 12 24)
max_concurrency_list=(1 2 4 8)

workspace=$(pwd)
cd $workspace
log_home=$workspace/benchmark_fastdeploy_logs/$(TZ='Asia/Shanghai' date '+WW%V')_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)_${model_log_name}_VarLen

mkdir -p ${log_home}

for ((i=0; i<${#num_prompts_list[@]}; i++)); do
    num_prompts=${num_prompts_list[$i]}
    max_concurrency=${max_concurrency_list[$i]}
    > log/hpu_model_runner_profile.log
    log_name_prefix="benchmarkdata_${model_log_name}_sharegpt_prompts_${num_prompts}_concurrency_${max_concurrency}"
    log_name=${log_name_prefix}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    echo "running benchmark with num_prompts ${num_prompts}, max_concurrency ${max_concurrency} log name ${log_name}"
    cmd="python benchmark_serving.py \
        --backend openai-chat \
        --model $model \
        --endpoint /v1/chat/completions \
        --host 0.0.0.0 \
        --port 8188 \
        --dataset-name EBChat \
        --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
        --hyperparameter-path ${model_yaml} \
        --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
        --metric-percentiles 80,95,99,99.9,99.95,99.99 \
        --num-prompts ${num_prompts} \
        --max-concurrency ${max_concurrency}"
    echo $cmd | tee -a ${log_home}/${log_name}.log
    eval $cmd >> ${log_home}/${log_name}.log 2>&1

    cp log/hpu_model_runner_profile.log ${log_home}/${log_name}_profile.log
done
