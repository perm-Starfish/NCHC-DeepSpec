# Llama 3.2 evaluator support

This overlay adds evaluator support for the locally stored
`Llama-3.2-3B-Instruct` target and the trained DSpark, DFlash, and EAGLE-3
draft checkpoints.

## Changes

- Register `Llama3DSparkModel` and `Llama3Eagle3Model` in `eval.py`.
- Add Llama evaluator subclasses for DSpark/DFlash and EAGLE-3.
- Add `--precision`; use `fp16` on V100.
- Add `--local-files-only` for the internal-S3/local model layout.
- Add `--max-samples-per-task` for bounded smoke tests.
- Do not pass the Qwen-only `enable_thinking` template option to Llama.

## Prerequisite

The selected task must exist under `/workspace/DeepSpec/eval_datasets`, for
example:

```bash
test -s /workspace/DeepSpec/eval_datasets/gsm8k.jsonl
```

Evaluation does not use the 297 GB target cache.

## Smoke test

Run inside the existing Compose container:

```bash
docker compose exec deepspec bash -lc \
  'GPU=0 bash scripts/eval/eval_llama32_3b_smoke.sh'
```

The script evaluates the three final `step_420` checkpoints sequentially and
writes these logs:

```text
/data/logs/llama_dspark_eval_smoke.log
/data/logs/llama_dflash_eval_smoke.log
/data/logs/llama_eagle3_eval_smoke.log
```

Check the results with:

```bash
for method in dspark dflash eagle3; do
  log="/data/logs/llama_${method}_eval_smoke.log"
  echo "===== ${method} ====="
  tail -n 30 "${log}"
done
```

Only proceed to the full benchmark after all three smoke tests exit normally
and print a result table with four samples.
