export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# export FLAGS_intel_hpu_recipe_cache_config=/tmp/recipe,false,10240
export SERVER_PORT=8188
export ENGINE_WORKER_QUEUE_PORT=8002
export METRICS_PORT=8001

export FLAGS_selected_intel_hpus=0
rm -rf log 2>/dev/null
HPU_WARMUP_BUCKET=1 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=BLOCK_ATTN python -m fastdeploy.entrypoints.openai.api_server --model /data/disk3/ernie_opensource/ERNIE-4.5-21B-A3B-Paddle --port ${SERVER_PORT} --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} --metrics-port ${METRICS_PORT} --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 --block-size 128

# (2k + 1k) / 128(block_size) * 128(batch) = 3072
# export FLAGS_selected_intel_hpus=0,1,2,3,4,5,6,7
# rm -rf log 2>/dev/null
# HPU_WARMUP_BUCKET=1 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=BLOCK_ATTN python -m fastdeploy.entrypoints.openai.api_server --model /data/disk3/ernie_opensource/ERNIE-4.5-300B-A47B-Paddle --port ${SERVER_PORT} --engine-worker-queue-port ${ENGINE_WORKER_QUEUE_PORT} --metrics-port ${METRICS_PORT} --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 128 --block-size 128 --static-decode-blocks 1 --num-gpu-blocks-override 3100
