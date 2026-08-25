# Best Draft Models

This document describes the best Llama-3.2-3B-Instruct draft-model checkpoints selected from the experiments conducted in this project.

Machine-readable paths and parameters are recorded in `release/best_models.json`.

## Target Model

* Target model: `meta-llama/Llama-3.2-3B-Instruct`
* Target cache: `/data/deepspec_cache/llama32_3b_instruct_target_cache`
* Precision: FP16
* Target feature layers: `[1, 7, 13, 19, 25]`
* Training GPUs: 4
* Training samples: 21,535

## Selected Checkpoints

| Method  |     LR | Final step | Selected step | Macro accept length | Macro verify rate |
| ------- | -----: | ---------: | ------------: | ------------------: | ----------------: |
| EAGLE-3 | `5e-5` |        420 |           200 |           **1.177** |        **0.1469** |
| DFlash  | `6e-5` |        420 |           420 |               1.101 |            0.1375 |
| DSpark  | `6e-5` |        100 |            75 |               1.113 |            0.1391 |

> Macro averages are unweighted averages across the seven evaluation datasets. They are not recalculated directly from all decoding cycles.

## Results by Dataset

| Dataset          |   Samples |   EAGLE-3 AL / VR | DFlash AL / VR |    DSpark AL / VR |
| ---------------- | --------: | ----------------: | -------------: | ----------------: |
| GSM8K            |     1,319 | **1.17 / 0.1457** |  1.11 / 0.1382 |     1.08 / 0.1347 |
| MATH             |       500 | **1.28 / 0.1598** |  1.08 / 0.1350 |     1.11 / 0.1391 |
| MMLU-STEM        |       318 | **1.15 / 0.1432** |  1.09 / 0.1358 |     1.06 / 0.1327 |
| OCW              |       272 | **1.19 / 0.1485** |  1.06 / 0.1319 |     1.08 / 0.1348 |
| TW Function Call |       200 |     1.33 / 0.1667 |  1.29 / 0.1618 | **1.42 / 0.1770** |
| TW Instruct      |       500 | **1.06 / 0.1323** |  1.04 / 0.1300 |     1.02 / 0.1276 |
| Bailong Bench    |       140 | **1.06 / 0.1320** |  1.04 / 0.1297 |     1.02 / 0.1279 |
| **Total**        | **3,249** |                 — |              — |                 — |

AL = Accept Length, VR = Verify Rate. 

## Result Analysis

* **EAGLE-3 achieved the strongest overall performance**, obtaining the highest acceptance length on six of the seven evaluation datasets.
* **DSpark performed best on the TW Function Call dataset**, reaching the highest acceptance length of `1.42`, suggesting particular strength on structured function-calling tasks.
* **DFlash showed relatively consistent performance across datasets**, although it did not achieve the highest acceptance length on any individual dataset in this evaluation.


## Evaluation Settings

```text
Target model: meta-llama/Llama-3.2-3B-Instruct
Total samples: 3,249
Datasets: 7
Proposals: 7 draft tokens + 1 target bonus token
Temperature: 0.0
Precision: FP16
Max new tokens: 2,048
Confidence threshold: 0.0
Seed: 980406
```

## Checkpoint Paths

### EAGLE-3

```text
/data/checkpoints/deepspec/eagle3_ttt7_llama32_3b_instruct_v100_lr5e5/step_200
```

Config:

```text
config/best/eagle3_llama32_3b_lr5e5.py
```

Training script:

```text
scripts/train/train_best_eagle3.sh
```

### DFlash

```text
/data/checkpoints/deepspec/dflash_block7_llama32_3b_instruct_v100_lr6e5/step_420
```

Config:

```text
config/best/dflash_llama32_3b_lr6e5.py
```

Training script:

```text
scripts/train/train_best_dflash.sh
```

### DSpark

```text
/data/checkpoints/deepspec/dspark_block7_llama32_3b_instruct_v100_lr6e5_ceonly_markov_pilot100/step_75
```

Config:

```text
config/best/dspark_llama32_3b_lr6e5_ceonly.py
```

Training script:

```text
scripts/train/train_best_dspark.sh
```

For DSpark, “CE-only” means `ce_loss_alpha=1.0`, `l1_loss_alpha=0.0`, and `confidence_head_alpha=0.0`. The Markov component remains enabled with `markov_rank=256`; it is part of the model architecture, not an additional loss term.

## Selection Conditions

The checkpoints were selected from the configurations tested in this project. The main screening evaluation used:

* Dataset: GSM8K
* Draft length: 7 tokens
* Temperature: 0
* Precision: FP16
* Small-scale evaluation during parameter tuning

These checkpoints are the best among the tested configurations and should not be interpreted as globally optimal. A complete benchmark under identical evaluation conditions is recommended before making final performance comparisons.

## Reproducing Training

Run from the DeepSpec repository root:

```bash
bash scripts/train/train_best_eagle3.sh
bash scripts/train/train_best_dflash.sh
bash scripts/train/train_best_dspark.sh
```

The scripts default to:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Paths can be overridden through environment variables such as:

```bash
TARGET_CACHE_DIR=/data/deepspec_cache/llama32_3b_instruct_target_cache \
CHECKPOINT_ROOT=/data/checkpoints_repro \
LOG_DIR=/data/logs/repro \
bash scripts/train/train_best_eagle3.sh
```

## Docker Usage

The Docker image contains the source code, configs, manifest, and training scripts. Target-model weights, target cache, and draft checkpoints are mounted separately under `/data`.

For complete build and runtime instructions, see [`release/README.md`](./README.md).

Inspect the machine-readable manifest with:

```bash
python -m json.tool release/best_models.json
```

The learning rate is only used during training. Evaluation loads the selected checkpoint and does not use the training learning rate.
