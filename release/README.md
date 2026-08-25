# Llama-DeepSpec Release Image

This release records the best tested Llama-3.2-3B-Instruct recipes for
EAGLE-3, DFlash, and DSpark. The image contains the source code, pinned Python
dependencies, best configs, training scripts, release metadata, and the 3,249
sample custom evaluation subset.

It does **not** contain target-model weights, target cache, training data, or
draft checkpoints. Those large artifacts remain under the host `/data` tree and
are mounted at runtime.

## Selected Recipes

| Method | Learning rate | Training end | Selected checkpoint | Loss notes |
| --- | ---: | ---: | ---: | --- |
| EAGLE-3 | `5e-5` | 420 | 200 | EAGLE-3 objective |
| DFlash | `6e-5` | 420 | 420 | CE `1.0`, L1 `0.0`, confidence `0.0` |
| DSpark | `6e-5` | 100 | 75 | CE `1.0`, L1 `0.0`, confidence `0.0`, Markov rank `256` |

The machine-readable source of truth is
[`release/best_models.json`](./best_models.json). Benchmark results and absolute
checkpoint paths are documented in [`release/BEST_MODELS.md`](./BEST_MODELS.md).

## Build

Run these commands from the repository root. `Dockerfile.release` extends the
`deepspec:cuda12.6` development image. Rebuild that base once from the corrected
context so an older image layer cannot retain a previously copied `.venv`:

```bash
docker build \
  --build-arg LOCAL_UID="$(id -u)" \
  --build-arg LOCAL_GID="$(id -g)" \
  -t deepspec:cuda12.6 \
  -f Dockerfile .

docker build \
  -t llama-deepspec:best-v1 \
  -f Dockerfile.release .
```

Building creates Docker image layers and build cache. It does not delete or
modify repository files, checkpoints, models, or `/data` contents.

## Run

Start an interactive GPU container and mount the host artifacts read-write:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v /data:/data \
  --name llama-deepspec-release \
  llama-deepspec:best-v1
```

If your target model is stored elsewhere under the mounted tree, override the
variable that the configs actually read:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v /data:/data \
  -e LLAMA_3_2_3B_INSTRUCT_PATH=/data/models/your-llama-path \
  llama-deepspec:best-v1
```

## Inspect the Recorded Settings

Inside the container:

```bash
python -m json.tool "$BEST_MODELS_MANIFEST"
sed -n '1,220p' release/BEST_MODELS.md
```

Each produced checkpoint also contains `train_config.py`, which records the
source config plus all `--opts` overrides used when it was saved.

## Reproduce Training

The scripts default to four visible GPUs (`0,1,2,3`) and the `/data` paths
recorded in the manifest:

```bash
bash scripts/train/train_best_eagle3.sh
bash scripts/train/train_best_dflash.sh
bash scripts/train/train_best_dspark.sh
```

Override paths or GPUs without editing Python source:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TARGET_CACHE_DIR=/data/deepspec_cache/llama32_3b_instruct_target_cache \
CHECKPOINT_ROOT=/data/checkpoints_repro \
LOG_DIR=/data/logs/repro \
bash scripts/train/train_best_dspark.sh
```

Do not change the learning rates directly in the core trainer. The versioned
best configs and scripts are intended to preserve the tested recipes while
still allowing explicit environment or command-line overrides.

## Check the Image Contents

```bash
docker run --rm llama-deepspec:best-v1 \
  python -m json.tool /workspace/DeepSpec/release/best_models.json

docker image inspect llama-deepspec:best-v1 \
  --format '{{json .Config.Labels}}'
```

The three selected checkpoint directories must exist under `/data/checkpoints`
at runtime. They are referenced by the manifest but are intentionally not baked
into the image.
