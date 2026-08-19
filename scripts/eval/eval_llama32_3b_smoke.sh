#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DeepSpec
export PYTHONPATH=/workspace/DeepSpec

target_model="${TARGET_MODEL:-/data/models/meta-llama/Llama-3.2-3B-Instruct}"
checkpoint_root="${CHECKPOINT_ROOT:-/data/checkpoints/deepspec}"
log_root="${LOG_ROOT:-/data/logs}"
gpu="${GPU:-0}"

mkdir -p "${log_root}"

methods=(dspark dflash eagle3)
checkpoints=(
    "dspark_block7_llama32_3b_instruct_v100/step_420"
    "dflash_block7_llama32_3b_instruct_v100/step_420"
    "eagle3_ttt7_llama32_3b_instruct_v100/step_420"
)

for index in "${!methods[@]}"; do
    method="${methods[$index]}"
    draft_model="${checkpoint_root}/${checkpoints[$index]}"
    log_file="${log_root}/llama_${method}_eval_smoke.log"

    echo "===== ${method} evaluator smoke test ====="
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python eval.py \
        --target_name_or_path "${target_model}" \
        --draft_name_or_path "${draft_model}" \
        --precision fp16 \
        --local-files-only \
        --tasks gsm8k \
        --max-samples-per-task 4 \
        --max-new-tokens 64 \
        2>&1 | tee "${log_file}"
done
