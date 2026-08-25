#!/usr/bin/env bash
set -euo pipefail

# Place this file under scripts/train/ in the DeepSpec repository.
# train.py spawns one worker per GPU listed in CUDA_VISIBLE_DEVICES.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/best/dflash_llama32_3b_lr6e5.py}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/data/deepspec_cache/llama32_3b_instruct_target_cache}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-/data/tensorboard}"
LOG_DIR="${LOG_DIR:-/data/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_NAME="deepspec"
EXP_NAME="dflash_block7_llama32_3b_instruct_v100_lr6e5"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXP_NAME}"
TENSORBOARD_DIR="${TENSORBOARD_ROOT}/${PROJECT_NAME}/${EXP_NAME}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_train.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -d "${TARGET_CACHE_DIR}" ]] || { echo "Target cache not found: ${TARGET_CACHE_DIR}" >&2; exit 1; }

if [[ -e "${CHECKPOINT_DIR}/step_latest" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Checkpoint already exists: ${CHECKPOINT_DIR}" >&2
    echo "Set ALLOW_EXISTING=1 only if resuming/reusing this directory is intentional." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"
cd "${REPO_ROOT}"

echo "Training DFlash: lr=6e-5, total_steps=420, selected checkpoint=step_420"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${CHECKPOINT_DIR}"

"${PYTHON_BIN}" train.py \
    --config "${CONFIG_PATH}" \
    --opts "exp_name=${EXP_NAME}" \
    --opts "train.lr=6e-5" \
    --opts "train.max_train_steps=420" \
    --opts "logging.checkpointing_steps=50" \
    --opts "logging.checkpoint_dir=${CHECKPOINT_DIR}" \
    --opts "logging.tensorboard_dir=${TENSORBOARD_DIR}" \
    --opts "data.target_cache_path=${TARGET_CACHE_DIR}" \
    2>&1 | tee "${LOG_FILE}"

