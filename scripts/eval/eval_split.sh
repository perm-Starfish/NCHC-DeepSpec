#!/usr/bin/env bash
set -uo pipefail

cd /workspace/DeepSpec
export PYTHONPATH=/workspace/DeepSpec

target_name_or_path="Qwen/Qwen3-4B"
draft_name_or_path="/data/checkpoints/deepspec/eagle3_ttt7_qwen3_4b_v100/step_170"
log_dir="/workspace/DeepSpec/logs"

# GPU 0
# aime25, mt-bench, gsm8k, math500
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}" \
    --max-new-tokens 2048 \
    --tasks aime25 mt-bench gsm8k math500 \
    > "${log_dir}/eagle3_step170_gpu0.log" 2>&1 &

pid_gpu0=$!

# GPU 1
# humaneval, mbpp, livecodebench, alpaca
# arena-hard-v2 (OOM)
MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
CUDA_VISIBLE_DEVICES=1 python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}" \
    --max-new-tokens 2048 \
    --tasks humaneval mbpp livecodebench alpaca \
    > "${log_dir}/eagle3_step170_gpu1.log" 2>&1 &

pid_gpu1=$!

echo "GPU 0 process PID: ${pid_gpu0}"
echo "GPU 1 process PID: ${pid_gpu1}"
echo "GPU 0 log: ${log_dir}/eagle3_step170_gpu0.log"
echo "GPU 1 log: ${log_dir}/eagle3_step170_gpu1.log"

status_gpu0=0
status_gpu1=0

wait "${pid_gpu0}" || status_gpu0=$?
wait "${pid_gpu1}" || status_gpu1=$?

echo "GPU 0 exit status: ${status_gpu0}"
echo "GPU 1 exit status: ${status_gpu1}"

if [[ "${status_gpu0}" -ne 0 || "${status_gpu1}" -ne 0 ]]; then
    exit 1
fi