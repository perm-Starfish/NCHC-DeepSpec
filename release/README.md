# Llama-DeepSpec Release Image

This release records the best tested Llama-3.2-3B-Instruct recipes for EAGLE-3, DFlash, and DSpark.

The image contains:

* Source code
* Pinned Python dependencies
* Best-model configurations
* Training and evaluation scripts
* Release metadata and documentation

The image does **not** contain:

* Training datasets
* Evaluation datasets
* Target-model weights
* Target feature cache
* Draft-model checkpoints
* Training logs or TensorBoard outputs

All datasets and large artifacts are mounted from the host at runtime.

## Selected Recipes

| Method  | Learning rate | Training end | Selected checkpoint | Loss notes                                              |
| ------- | ------------: | -----------: | ------------------: | ------------------------------------------------------- |
| EAGLE-3 |        `5e-5` |          420 |                 200 | EAGLE-3 objective                                       |
| DFlash  |        `6e-5` |          420 |                 420 | CE `1.0`, L1 `0.0`, confidence `0.0`                    |
| DSpark  |        `6e-5` |          100 |                  75 | CE `1.0`, L1 `0.0`, confidence `0.0`, Markov rank `256` |

The machine-readable source of truth is [`best_models.json`](./best_models.json). Benchmark results and absolute checkpoint paths are documented in [`BEST_MODELS.md`](./BEST_MODELS.md).

## Runtime Data Layout

The following host directories are mounted into the same paths expected by the source code:

| Host path               | Container path                       | Purpose                                                          |
| ----------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `<repo>/data_v2`        | `/workspace/DeepSpec/data_v2`        | Custom evaluation prompts and references                         |
| `<repo>/train_datasets` | `/workspace/DeepSpec/train_datasets` | Training datasets                                                |
| `/data`                 | `/data`                              | Models, target cache, checkpoints, logs, and TensorBoard outputs |

The tracked symbolic link:

```text
eval_datasets/llama32_3b_custom.jsonl
    -> ../data_v2/prompts/eval/all_eval.jsonl
```

becomes valid after `data_v2` is mounted.

## Build

Run these commands from the repository root.

`Dockerfile.release` extends the `deepspec:cuda12.6` development image.

If the base image already exists and is current, only build the release image:

```bash
docker build \
  -t llama-deepspec:best-v1 \
  -f Dockerfile.release \
  .
```

If the base image must also be rebuilt:

```bash
docker build \
  --build-arg LOCAL_UID="$(id -u)" \
  --build-arg LOCAL_GID="$(id -g)" \
  -t deepspec:cuda12.6 \
  -f Dockerfile \
  .

docker build \
  -t llama-deepspec:best-v1 \
  -f Dockerfile.release \
  .
```

Building creates Docker image layers and build cache. It does not modify repository files, datasets, checkpoints, models, or `/data` contents.

## Run

Run this command from the repository root:

```bash
PROJECT_DIR="$(pwd)"

docker run --rm -it \
  --name llama-deepspec-release \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v "$PROJECT_DIR/data_v2:/workspace/DeepSpec/data_v2:ro" \
  -v "$PROJECT_DIR/train_datasets:/workspace/DeepSpec/train_datasets:ro" \
  -v /data:/data \
  llama-deepspec:best-v1
```

The dataset mounts use `:ro` so the container cannot modify the host datasets. The `/data` mount remains writable because training and evaluation may create checkpoints, logs, or TensorBoard outputs.

## Override the Target Model Path

The default target-model path is:

```text
/data/models/meta-llama/Llama-3.2-3B-Instruct
```

If the model is stored elsewhere under `/data`, override the variable used by the Llama configuration:

```bash
PROJECT_DIR="$(pwd)"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v "$PROJECT_DIR/data_v2:/workspace/DeepSpec/data_v2:ro" \
  -v "$PROJECT_DIR/train_datasets:/workspace/DeepSpec/train_datasets:ro" \
  -v /data:/data \
  -e LLAMA_3_2_3B_INSTRUCT_PATH=/data/models/your-llama-path \
  llama-deepspec:best-v1
```

## Verify Runtime Mounts

Verify that the custom evaluation symlink resolves after mounting `data_v2`:

```bash
PROJECT_DIR="$(pwd)"

docker run --rm \
  -v "$PROJECT_DIR/data_v2:/workspace/DeepSpec/data_v2:ro" \
  -v "$PROJECT_DIR/train_datasets:/workspace/DeepSpec/train_datasets:ro" \
  llama-deepspec:best-v1 \
  bash -lc '
    test -s eval_datasets/llama32_3b_custom.jsonl &&
    test -d train_datasets &&
    wc -l eval_datasets/llama32_3b_custom.jsonl
  '
```

The evaluation file should report:

```text
3249 eval_datasets/llama32_3b_custom.jsonl
```

## Inspect the Recorded Settings

Inside the container:

```bash
python -m json.tool "$BEST_MODELS_MANIFEST"
sed -n '1,220p' release/BEST_MODELS.md
```

Each produced checkpoint also contains `train_config.py`, which records the source configuration and the `--opts` overrides used during training.

## Reproduce Training

The scripts default to four visible GPUs (`0,1,2,3`) and the `/data` paths recorded in the manifest:

```bash
bash scripts/train/train_best_eagle3.sh
bash scripts/train/train_best_dflash.sh
bash scripts/train/train_best_dspark.sh
```

Paths and visible GPUs can be overridden without modifying the Python source:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TARGET_CACHE_DIR=/data/deepspec_cache/llama32_3b_instruct_target_cache \
CHECKPOINT_ROOT=/data/checkpoints_repro \
LOG_DIR=/data/logs/repro \
bash scripts/train/train_best_dspark.sh
```

Do not change the learning rates directly in the core trainer. The versioned best configurations and scripts preserve the tested recipes while allowing explicit environment-variable or command-line overrides.

## Check the Image Contents

Validate the release manifest and image labels:

```bash
docker run --rm \
  llama-deepspec:best-v1 \
  python -m json.tool \
  /workspace/DeepSpec/release/best_models.json

docker image inspect \
  llama-deepspec:best-v1 \
  --format '{{json .Config.Labels}}'
```

The selected checkpoint directories must exist under `/data/checkpoints` at runtime. They are referenced by the manifest but are intentionally not included in the image.
