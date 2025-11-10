# benchmark_paddle_hpu_server_new.sh 使用说明

本脚本用于在 Intel HPU 上启动 FastDeploy Paddle 大模型推理服务，支持灵活配置 KV Cache Ratio、Block 数量、Warmup等参数，适用于性能/压力测试。

## 主要参数说明
- `MODEL`：模型名称（如 ERNIE-4.5-21B-A3B-Paddle）
- `TP_SIZE`：张量并行卡数（1 或 8）
- `MAX_NUM_SEQS`：最大批量, 默认值128
- `BLOCK_SIZE`：KV Cache Block 大小，默认值128
- `MAX_MODEL_LEN`：默认最大模型长度，默认值32K
- `NUM_GPU_BLOCKS_OVERRIDE`：HPU Block 总数，第一次可先设置为0，FastDeploy会运行profile run，在worker_process.log可获取其值，设置后再启动
- `ENABLE_PREFIX_CACHING`：是否启用前缀缓存（true/false），默认值false
- `FD_ENC_DEC_BLOCK_NUM_DEFAULT`：每个请求解码block的数目的默认值2
- `KV_CACHE_RATIO_DEFAULT`：KV Cache 比例的默认值0.75

- `METRICS_PORT`：服务监控端口
- `ENGINE_WORKER_QUEUE_PORT`：Engine Worker 队列端口
- `CACHE_QUEUE_PORT`：Cache 队列端口
- `SERVER_PORT`：服务端口

- `MAX_INPUT_LENGTH`/`MAX_OUTPUT_LENGTH`/`AVG_INPUT_LENGTH`/`AVG_OUTPUT_LENGTH`：输入/输出长度配置，支持定长/变长，如果知道测试数据的分布，可以进行相应设置，以获取更好性能

- `HPU_WARMUP_BUCKET`：是否启用 Warmup（1 表示启用）
- `HPU_WARMUP_MODEL_LEN`：Warmup的模型长度（含输入和输出）
- `MAX_PREFILL_NUM`：prefill阶段最大的batch, 默认值是3
- `BATCH_STEP_PREFILL`：prefill 阶段的 batch 步长，默认值是1
- `SEQUENCE_STEP_PREFILL`：prefill 阶段的 sequence 步长，默认是128，与block size保持一致
- `CONTEXT_BLOCK_STEP_PREFILL`：prefill 阶段开启prefill caching时，命中block数目的步长，默认值是1
- `BATCH_STEP_DECODE`：decode 阶段的 batch 步长，默认值是4
- `BLOCK_STEP_DECODE`：decode 阶段的 block 步长，默认值是16
- `FLAGS_intel_hpu_recipe_cache_num`: HPU recipe cache数目的限制
- `FLAGS_intel_hpu_recipe_cache_config`: HPU recipe cache的Config，可用于Warmup阶段的优化

## 主要功能和逻辑
1. 对输入/输出长度进行 Block 对齐。
2. 如果设置AVG_OUTPUT_LENGTH，则FD_ENC_DEC_BLOCK_NUM的值为AVG_OUTPUT_LENGTH/BLOCK_SIZE。
3. 如果设置AVG_INPUT_LENGTH和AVG_OUTPUT_LENGTH, 则KV_CACHE_RATIO的值为(AVG_INPUT_LENGTH + BLOCK_SIZE) / (AVG_INPUT_LENGTH + AVG_OUTPUT_LENGTH)。
4. 如果设置NUM_GPU_BLOCKS_OVERRIDE和MAX_OUTPUT_LENGTH，确保FreeList里Block的数目足够其使用，防止OOM。
5. 如果设置NUM_GPU_BLOCKS_OVERRIDE和AVG_INPUT_LENGTH，根据MAX_NUM_SEQS和FD_ENC_DEC_BLOCK_NUM计算实际可能需要的Block数目，与NUM_GPU_BLOCKS_OVERRIDE*KV_CACHE_RATIO比较，判断Block数目是否足够。
6. 如果设置MAX_INPUT_LENGTH和MAX_OUTPUT_LENGTH，修改实际MAX_MODEL_LEN和HPU_WARMUP_MODEL_LEN
7. 拼接命令，执行服务启动。

## 使用方法
1. 根据实际测试需求修改脚本顶部参数
2. 运行脚本：
   ```bash
   bash benchmark_paddle_hpu_server_new.sh
   ```
3. 查看 log/ 目录下日志，或通过端口访问服务
