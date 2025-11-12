#!/bin/bash
# set -x
#--------------------------------------------------------------------------------------------------------#
# Setting for HPU on FastDeploy: users can edit this section to suit their benchmarking needs.
#--------------------------------------------------------------------------------------------------------#
MODEL="ERNIE-4.5-21B-A3B-Paddle" # ERNIE-4.5-21B-A3B-Paddle, ERNIE-4.5-300B-A47B-Paddle
TP_SIZE=1                        # 1, 8
SELECTED_CARD=0                  # 0 for TP_SIZE=1, 0,1,2,3,4,5,6,7 for TP_SIZE=8
MAX_NUM_SEQS=128                 # server batch
BLOCK_SIZE=128                   # block size for kv cache
MAX_MODEL_LEN_DEFAULT=32768      # default max model length
NUM_GPU_BLOCKS_OVERRIDE=5000     # total HPU blocks
ENABLE_PREFIX_CACHING=false      # true/false
FD_ENC_DEC_BLOCK_NUM_DEFAULT=2   # default decode block numbers for each request
KV_CACHE_RATIO_DEFAULT=0.75      # default KV Cache ratio
METRICS_PORT=8001
ENGINE_WORKER_QUEUE_PORT=8002
CACHE_QUEUE_PORT=8003
SERVER_PORT=8188


# Fixed/Variable input/output lengths for benchmarking
MAX_INPUT_LENGTH=0
MAX_OUTPUT_LENGTH=0
AVG_INPUT_LENGTH=0
AVG_OUTPUT_LENGTH=0

export HPU_WARMUP_BUCKET=1                  # enable warmup
export HPU_WARMUP_MODEL_LEN=4096
export MAX_PREFILL_NUM=3
export BATCH_STEP_PREFILL=1
export SEQUENCE_STEP_PREFILL=128
export CONTEXT_BLOCK_STEP_PREFILL=1
export BATCH_STEP_DECODE=4
export BLOCK_STEP_DECODE=16
export FLAGS_intel_hpu_recipe_cache_num=20480
# export FLAGS_intel_hpu_recipe_cache_config=/tmp/recipe,false,20480

#--------------------------------------------------------------------------------------------------------#
# Tip: The following section involves important environment variable settings and automatic parameter calculation 
#      for HPU and FastDeploy. It is recommended not to modify unless you fully understand the parameter meanings.
#--------------------------------------------------------------------------------------------------------#
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
export HABANA_PROFILE=0
export HPU_PERF_BREAKDOWN_SYNC_MODE=1
export FD_ATTENTION_BACKEND=HPU_ATTN

MAX_INPUT_LENGTH=$(((MAX_INPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE))
MAX_OUTPUT_LENGTH=$(((MAX_OUTPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE))
AVG_INPUT_LENGTH=$(((AVG_INPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE))
AVG_OUTPUT_LENGTH=$(((AVG_OUTPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE))

export HPU_VISIBLE_DEVICES=${SELECTED_CARD}

if [ "$AVG_OUTPUT_LENGTH" -ne 0 ]; then
    export FD_ENC_DEC_BLOCK_NUM=$(( (AVG_OUTPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE ))
else
    export FD_ENC_DEC_BLOCK_NUM=$FD_ENC_DEC_BLOCK_NUM_DEFAULT
fi

if [ "$AVG_INPUT_LENGTH" -ne 0 ] && [ "$AVG_OUTPUT_LENGTH" -ne 0 ]; then
    KV_CACHE_RATIO=$(printf "%.3f" "$(echo "scale=5; ($AVG_INPUT_LENGTH + $BLOCK_SIZE) / ($AVG_INPUT_LENGTH + $AVG_OUTPUT_LENGTH)" | bc)")
else    
    KV_CACHE_RATIO=$KV_CACHE_RATIO_DEFAULT
fi

if [ "$NUM_GPU_BLOCKS_OVERRIDE" -ne 0 ] && [ "$MAX_OUTPUT_LENGTH" -ne 0 ]; then
    FREE_BLOCK=$(printf "%.0f" "$(echo "$NUM_GPU_BLOCKS_OVERRIDE * (1 - $KV_CACHE_RATIO)" | bc)")
    FREE_BLOCK_LENGTH=$(( (FREE_BLOCK - 1) * BLOCK_SIZE))
    if [ "$FREE_BLOCK_LENGTH" -lt "$MAX_OUTPUT_LENGTH" ]; then
        echo "Error: NUM_GPU_BLOCKS_OVERRIDE is too small for the given MAX_OUTPUT_LENGTH and KV_CACHE_RATIO."
        exit 1
    fi
fi

if [ "$NUM_GPU_BLOCKS_OVERRIDE" -ne 0 ] && [ "$AVG_INPUT_LENGTH" -ne 0 ]; then
    NEED_BLOCKS=$(( ((AVG_INPUT_LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE  + FD_ENC_DEC_BLOCK_NUM) * MAX_NUM_SEQS ))
    ACTUAL_BLOCKS=$(printf "%.0f" "$(echo "$NUM_GPU_BLOCKS_OVERRIDE * $KV_CACHE_RATIO" | bc)")
    if [ "$NEED_BLOCKS" -gt "$ACTUAL_BLOCKS" ]; then
        echo "Error: NUM_GPU_BLOCKS_OVERRIDE is a bit small for the given AVG_INPUT_LENGTH and MAX_NUM_SEQS."
        exit 1
    fi
fi
#--------------------------------------------------------------------------------------------------------#
rm -rf log 2>/dev/null

CMD="python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL} \
    --port ${SERVER_PORT} \
    --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} \
    --metrics-port ${METRICS_PORT} \
    --cache-queue-port ${CACHE_QUEUE_PORT} \
    --tensor-parallel-size ${TP_SIZE} \
    --kv-cache-ratio ${KV_CACHE_RATIO} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --block-size ${BLOCK_SIZE} \
    --graph-optimization-config '{\"use_cudagraph\":false}'"

if [ "$NUM_GPU_BLOCKS_OVERRIDE" -ne 0 ]; then
    CMD="$CMD --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
fi

if [ "$MAX_INPUT_LENGTH" -ne 0 ] && [ "$MAX_OUTPUT_LENGTH" -ne 0 ]; then
    ACTUAL_MODEL_LEN=$((MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH))
    export HPU_WARMUP_MODEL_LEN=${ACTUAL_MODEL_LEN}
    if [ "$ACTUAL_MODEL_LEN" -gt "$MAX_MODEL_LEN" ]; then
        CMD="$CMD --max-model-len ${ACTUAL_MODEL_LEN}"
    else
        CMD="$CMD --max-model-len ${MAX_MODEL_LEN_DEFAULT}"
    fi
else
    CMD="$CMD --max-model-len ${MAX_MODEL_LEN_DEFAULT}"
fi

if [ "$ENABLE_PREFIX_CACHING" = false ]; then
    CMD="$CMD --no-enable-prefix-caching"
fi

eval $CMD
