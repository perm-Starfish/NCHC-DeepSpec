#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Llama-3.2-3B-Instruct full custom-dataset evaluation
#
# Custom evaluation dataset:
#   data_v2/prompts/eval/all_eval.jsonl
#   3,249 samples
#
# Best checkpoints:
#   The values below are defaults only. Override *_STEP, *_DRAFT,
#   *_PATTERN, CHECKPOINT_ROOT, or other path variables at runtime.
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-$HOME/Llama-DeepSpec}"
CONTAINER="${CONTAINER:-deepspec-dev}"

TARGET_MODEL="${TARGET_MODEL:-/data/models/meta-llama/Llama-3.2-3B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/deepspec}"

CUSTOM_DATASET="${CUSTOM_DATASET:-data_v2/prompts/eval/all_eval.jsonl}"
CUSTOM_REFERENCE="${CUSTOM_REFERENCE:-data_v2/references/all_eval.jsonl}"
CUSTOM_EVAL_PATH_CONTAINER="${CUSTOM_EVAL_PATH_CONTAINER:-/workspace/DeepSpec/data_v2/prompts/eval/all_eval.jsonl}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-3249}"

CUSTOM_TASKS=(
    "llama32_gsm8k"
    "llama32_math"
    "llama32_mmlu_stem"
    "llama32_ocw"
    "llama32_tw_function_call"
    "llama32_tw_instruct"
    "llama32_bailong"
)

CUSTOM_DATASET_NAMES=(
    "gsm8k"
    "math"
    "mmlu_stem"
    "ocw"
    "tw_function_call_reasoning_10k"
    "tw_instruct_500k"
    "bailong_bench"
)

EXPECTED_DATASET_COUNTS=(
    1319
    500
    318
    272
    200
    500
    140
)

LOG_ROOT="${LOG_ROOT:-/data/logs/llama32_3b_full_eval}"
TB_ROOT="${TB_ROOT:-/data/tensorboard/deepspec/llama32_3b_full_eval}"

EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
PRECISION="${PRECISION:-fp16}"
SEED="${SEED:-980406}"

# Default best steps. If *_DRAFT is explicitly set and ends in step_<N>,
# the step is inferred from that directory unless *_STEP is also set.
EAGLE3_DEFAULT_STEP="200"
DFLASH_DEFAULT_STEP="420"
DSPARK_DEFAULT_STEP="75"

# Optional regular expressions used only when automatically discovering a
# checkpoint. Set a pattern to an empty string to disable preferred matching.
EAGLE3_PATTERN="${EAGLE3_PATTERN-eagle3.*(lr)?5e-?5}"
DFLASH_PATTERN="${DFLASH_PATTERN-dflash.*(lr)?6e-?5}"
DSPARK_PATTERN="${DSPARK_PATTERN-dspark.*((ce[-_]?only.*(lr)?6e-?5)|((lr)?6e-?5.*ce[-_]?only))}"

# Override steps and let the script discover checkpoints under CHECKPOINT_ROOT:
#
# EAGLE3_STEP=240 DFLASH_STEP=500 DSPARK_STEP=100 \
# CHECKPOINT_ROOT=/data/checkpoints/another_run \
# ./eval_llama32_3b.sh
#
# Or specify exact checkpoint paths. The step is inferred from a final
# step_<N> directory name, or can be set explicitly with *_STEP:
#
# EAGLE3_DRAFT=/data/checkpoints/run_a/step_240 \
# DFLASH_DRAFT=/data/checkpoints/run_b/step_500 \
# DSPARK_DRAFT=/data/checkpoints/run_c/step_100 \
# ./eval_llama32_3b.sh

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"


# Automatically discover checkpoint

discover_checkpoint() {
    local method="$1"
    local step="$2"
    local preferred_pattern="$3"

    local -a all_candidates=()
    local -a preferred_candidates=()

    mapfile -t all_candidates < <(
        find "$CHECKPOINT_ROOT" \
            -mindepth 1 \
            -type d \
            -name "step_${step}" \
            2>/dev/null |
        grep -Ei "/${method}[^/]*/step_${step}$" |
        sort || true
    )

    if (( ${#all_candidates[@]} == 0 )); then
        echo "ERROR: Could not find ${method} step_${step} checkpoint." >&2
        echo "Search location: ${CHECKPOINT_ROOT}" >&2
        return 1
    fi

    if [[ -n "$preferred_pattern" ]]; then
        mapfile -t preferred_candidates < <(
            printf '%s\n' "${all_candidates[@]}" |
            grep -Ei "$preferred_pattern" || true
        )
    fi

    if (( ${#preferred_candidates[@]} == 1 )); then
        printf '%s\n' "${preferred_candidates[0]}"
        return 0
    fi

    if (( ${#preferred_candidates[@]} > 1 )); then
        echo "ERROR: ${method} found multiple checkpoints matching the preferred pattern:" >&2
        printf '  %s\n' "${preferred_candidates[@]}" >&2
        echo "Please use ${method^^}_DRAFT to explicitly specify the path." >&2
        return 1
    fi

    if (( ${#all_candidates[@]} == 1 )); then
        printf '%s\n' "${all_candidates[0]}"
        return 0
    fi

    echo "ERROR: ${method} step_${step} has multiple candidate checkpoints:" >&2
    printf '  %s\n' "${all_candidates[@]}" >&2
    echo "Please specify the exact checkpoint path at the top of the script or during execution." >&2
    return 1
}


resolve_step() {
    local configured_step="$1"
    local draft_path="$2"
    local default_step="$3"
    local method="$4"
    local resolved_step=""

    if [[ -n "$configured_step" ]]; then
        resolved_step="$configured_step"
    elif [[ -n "$draft_path" && "$(basename "$draft_path")" =~ ^step_([0-9]+)$ ]]; then
        resolved_step="${BASH_REMATCH[1]}"
    else
        resolved_step="$default_step"
    fi

    if [[ ! "$resolved_step" =~ ^[0-9]+$ ]] || (( resolved_step <= 0 )); then
        echo "ERROR: ${method} step must be a positive integer: $resolved_step" >&2
        return 1
    fi

    printf '%s\n' "$resolved_step"
}


# Project and dataset validation

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi

cd "$PROJECT_DIR"

if [[ ! -f "eval.py" ]]; then
    echo "ERROR: eval.py not found in project directory: $PROJECT_DIR" >&2
    exit 1
fi

if [[ ! -f "$CUSTOM_DATASET" ]]; then
    echo "ERROR: Custom evaluation dataset not found:" >&2
    echo "  $PROJECT_DIR/$CUSTOM_DATASET" >&2
    exit 1
fi

if [[ ! -f "$CUSTOM_REFERENCE" ]]; then
    echo "ERROR: Custom evaluation reference file not found:" >&2
    echo "  $PROJECT_DIR/$CUSTOM_REFERENCE" >&2
    exit 1
fi

PROMPT_COUNT="$(wc -l < "$CUSTOM_DATASET")"
REFERENCE_COUNT="$(wc -l < "$CUSTOM_REFERENCE")"

if [[ "$PROMPT_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Expected $EXPECTED_SAMPLES evaluation prompts, found $PROMPT_COUNT." >&2
    echo "Dataset: $CUSTOM_DATASET" >&2
    exit 1
fi

if [[ "$REFERENCE_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Expected $EXPECTED_SAMPLES evaluation references, found $REFERENCE_COUNT." >&2
    echo "Reference file: $CUSTOM_REFERENCE" >&2
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required to validate custom dataset counts." >&2
    exit 1
fi

for task_name in "${CUSTOM_TASKS[@]}"; do
    if ! grep -Fq "(\"${task_name}\"," eval.py; then
        echo "ERROR: Custom task is not registered in eval.py: $task_name" >&2
        exit 1
    fi
done

if ! grep -Fq 'CUSTOM_TASK_DATASETS' deepspec/eval/base_evaluator.py; then
    echo "ERROR: base_evaluator.py does not contain custom dataset filtering support." >&2
    exit 1
fi

VALIDATED_TOTAL=0
for index in "${!CUSTOM_TASKS[@]}"; do
    dataset_name="${CUSTOM_DATASET_NAMES[$index]}"
    expected_count="${EXPECTED_DATASET_COUNTS[$index]}"
    actual_count="$(jq -c --arg dataset "$dataset_name" \
        'select((.dataset | ascii_downcase | gsub("-"; "_")) == $dataset)' \
        "$CUSTOM_DATASET" | wc -l)"
    actual_count="${actual_count//[[:space:]]/}"

    if [[ "$actual_count" -ne "$expected_count" ]]; then
        echo "ERROR: Unexpected sample count for $dataset_name." >&2
        echo "Expected: $expected_count" >&2
        echo "Found   : $actual_count" >&2
        exit 1
    fi

    VALIDATED_TOTAL=$((VALIDATED_TOTAL + actual_count))
done

if [[ "$VALIDATED_TOTAL" -ne "$EXPECTED_SAMPLES" ]]; then
    echo "ERROR: Per-dataset counts sum to $VALIDATED_TOTAL, expected $EXPECTED_SAMPLES." >&2
    exit 1
fi


# Discover selected checkpoints

EAGLE3_STEP="$(resolve_step \
    "${EAGLE3_STEP:-}" \
    "${EAGLE3_DRAFT:-}" \
    "$EAGLE3_DEFAULT_STEP" \
    "EAGLE-3")"

DFLASH_STEP="$(resolve_step \
    "${DFLASH_STEP:-}" \
    "${DFLASH_DRAFT:-}" \
    "$DFLASH_DEFAULT_STEP" \
    "DFlash")"

DSPARK_STEP="$(resolve_step \
    "${DSPARK_STEP:-}" \
    "${DSPARK_DRAFT:-}" \
    "$DSPARK_DEFAULT_STEP" \
    "DSpark")"

EAGLE3_DRAFT="${EAGLE3_DRAFT:-$(discover_checkpoint \
    "eagle3" \
    "$EAGLE3_STEP" \
    "$EAGLE3_PATTERN")}"

DFLASH_DRAFT="${DFLASH_DRAFT:-$(discover_checkpoint \
    "dflash" \
    "$DFLASH_STEP" \
    "$DFLASH_PATTERN")}"

DSPARK_DRAFT="${DSPARK_DRAFT:-$(discover_checkpoint \
    "dspark" \
    "$DSPARK_STEP" \
    "$DSPARK_PATTERN")}"


# Model, container, and checkpoint validation

if [[ ! -d "$TARGET_MODEL" ]]; then
    echo "ERROR: Target model not found: $TARGET_MODEL" >&2
    exit 1
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ERROR: Container not found: $CONTAINER" >&2
    echo "Please run: docker-compose up -d deepspec" >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
    echo "ERROR: Container is not running: $CONTAINER" >&2
    exit 1
fi

for checkpoint in \
    "$EAGLE3_DRAFT" \
    "$DFLASH_DRAFT" \
    "$DSPARK_DRAFT"
do
    if [[ ! -f "$checkpoint/config.json" ]]; then
        echo "ERROR: Checkpoint lacks config.json: $checkpoint" >&2
        exit 1
    fi

    if ! find "$checkpoint" \
        -maxdepth 1 \
        -type f \
        -name '*.safetensors' \
        -print -quit |
        grep -q .
    then
        echo "ERROR: Checkpoint lacks safetensors weights: $checkpoint" >&2
        exit 1
    fi
done

mkdir -p "$LOG_ROOT" "$TB_ROOT"

echo "============================================================"
echo "Llama full custom-dataset evaluation"
echo "============================================================"
echo "Target model : $TARGET_MODEL"
echo "Dataset      : $CUSTOM_DATASET"
echo "Samples      : $PROMPT_COUNT"
echo "GPUs         : $EVAL_GPUS"
echo "Max tokens   : $MAX_NEW_TOKENS"
echo "Temperature  : $TEMPERATURE"
echo "Precision    : $PRECISION"
echo "Seed         : $SEED"
echo
echo "EAGLE-3      : step=$EAGLE3_STEP, checkpoint=$EAGLE3_DRAFT"
echo "DFlash       : step=$DFLASH_STEP, checkpoint=$DFLASH_DRAFT"
echo "DSpark       : step=$DSPARK_STEP, checkpoint=$DSPARK_DRAFT"
echo "============================================================"
echo
echo "Only the following custom datasets will be evaluated:"
for index in "${!CUSTOM_TASKS[@]}"; do
    printf '  %-34s %4d samples\n' \
        "${CUSTOM_DATASET_NAMES[$index]}" \
        "${EXPECTED_DATASET_COUNTS[$index]}"
done
echo


# Evaluation function

run_eval() {
    local method="$1"
    local step="$2"
    local draft="$3"

    local run_name="${method}_custom${EXPECTED_SAMPLES}_step${step}_${RUN_TAG}"
    local log_file="${LOG_ROOT}/${run_name}.log"
    local tensorboard_dir="${TB_ROOT}/${run_name}"

    echo
    echo "============================================================"
    echo "Starting: $method"
    echo "Tasks: ${CUSTOM_TASKS[*]}"
    echo "Samples: $EXPECTED_SAMPLES"
    echo "Checkpoint: $draft"
    echo "Log: $log_file"
    echo "============================================================"

    set +e

    docker exec \
        -e "CUDA_VISIBLE_DEVICES=${EVAL_GPUS}" \
        -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
        -w /workspace/DeepSpec \
        "$CONTAINER" \
        python eval.py \
            --target_name_or_path "$TARGET_MODEL" \
            --draft_name_or_path "$draft" \
            --custom-eval-path "$CUSTOM_EVAL_PATH_CONTAINER" \
            --tasks "${CUSTOM_TASKS[@]}" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --temperature "$TEMPERATURE" \
            --precision "$PRECISION" \
            --local-files-only \
            --seed "$SEED" \
            --tensorboard-dir "$tensorboard_dir" \
            --step "$step" \
        2>&1 | tee "$log_file"

    local eval_status=${PIPESTATUS[0]}

    set -e

    {
        echo
        echo "Evaluation exit status: $eval_status"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    } | tee -a "$log_file"

    if (( eval_status != 0 )); then
        echo "ERROR: $method evaluation failed." >&2
        echo "Please check: $log_file" >&2
        exit "$eval_status"
    fi

    echo "$method evaluation completed."
}


run_eval "eagle3" "$EAGLE3_STEP" "$EAGLE3_DRAFT"
run_eval "dflash" "$DFLASH_STEP" "$DFLASH_DRAFT"
run_eval "dspark" "$DSPARK_STEP" "$DSPARK_DRAFT"

echo
echo "============================================================"
echo "All custom-dataset evaluations completed"
echo "Dataset: $CUSTOM_DATASET"
echo "Samples: $PROMPT_COUNT"
echo "Logs: $LOG_ROOT"
echo "TensorBoard: $TB_ROOT"
echo "============================================================"
