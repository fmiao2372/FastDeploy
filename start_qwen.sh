export HF_ENDPOINT=https://hf-mirror.com
export FD_MODEL_SOURCE=HUGGINGFACE

export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
export HABANA_PROFILE=0
export HPU_VISIBLE_DEVICES=6

/workspace/kill_python.sh
rm -rf log

HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=HPU_ATTN python -m fastdeploy.entrypoints.openai.api_server --model /workspace/models/Qwen3-30B-A3B --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 --load-choices 'default_v1'
#HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=HPU_ATTN python -m fastdeploy.entrypoints.openai.api_server --model Qwen/Qwen3-8B --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 --load-choices 'default_v1'
