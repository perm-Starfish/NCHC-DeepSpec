#!/usr/bin/env bash
set -uo pipefail

cd /workspace/DeepSpec
export PYTHONPATH=/workspace/DeepSpec

target_name_or_path="Qwen/Qwen3-4B"
draft_name_or_path="/data/checkpoints/deepspec/dspark_block7_qwen3_4b_v100/step_170"
log_dir="/workspace/DeepSpec/logs"

# GPU 0
# 1. aime25 (small task test)
# 2. gsm8k, math500, alpaca
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}" \
    --max-new-tokens 2048 \
    --tasks gsm8k math500 alpaca \
    > "${log_dir}/dspark_step170_gpu0_remaining.log" 2>&1 &

pid_gpu0=$!

# GPU 1
# 1. mt-bench (small task test)
# 2. humaneval, mbpp, livecodebench (adding arena-hard-v2 causes OOM, so we run it separately)
# 3. arena-hard-v2
MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}" \
    --max-new-tokens 2048 \
    --tasks humaneval mbpp livecodebench arena-hard-v2 \
    > "${log_dir}/dspark_step170_gpu1_remaining.log" 2>&1 &

pid_gpu1=$!

echo "GPU 0 process PID: ${pid_gpu0}"
echo "GPU 1 process PID: ${pid_gpu1}"
echo "GPU 0 log: ${log_dir}/dspark_step170_gpu0_remaining.log"
echo "GPU 1 log: ${log_dir}/dspark_step170_gpu1_remaining.log"

status_gpu0=0
status_gpu1=0

wait "${pid_gpu0}" || status_gpu0=$?
wait "${pid_gpu1}" || status_gpu1=$?

echo "GPU 0 exit status: ${status_gpu0}"
echo "GPU 1 exit status: ${status_gpu1}"

if [[ "${status_gpu0}" -ne 0 || "${status_gpu1}" -ne 0 ]]; then
    exit 1
fi