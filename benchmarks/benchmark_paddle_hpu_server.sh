export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
export FLAGS_selected_intel_hpus=0
rm -rf log 2>/dev/null
FD_ATTENTION_BACKEND=BLOCK_ATTN python -m fastdeploy.entrypoints.openai.api_server --model /data/disk2/ERNIE-4.5-21B-A3B-Paddle --port 8188 --tensor-parallel-size 1 --max-model-len 32786 --max-num-seqs 128
