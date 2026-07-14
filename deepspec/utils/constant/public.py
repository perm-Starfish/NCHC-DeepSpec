import os

## cache dir
# CACHE_DIR = os.path.expanduser("~/.cache/deepspec")

## my own cache dir for research
CACHE_DIR = os.path.expanduser(
    os.environ.get(
        "DEEPSPEC_CACHE_DIR",
        "~/.cache/deepspec",
    )
)
## model_name_or_path
QWEN_3_4B = "Qwen/Qwen3-4B"
QWEN_3_8B = "Qwen/Qwen3-8B"
QWEN_3_14B = "Qwen/Qwen3-14B"
GEMMA_4_12B = "google/gemma-4-12B-it"
#BASE_TB_DIR = os.path.expanduser("~/tensorboard")

## my own tensorboard dir for research
BASE_TB_DIR = os.path.expanduser(
    os.environ.get(
        "DEEPSPEC_TENSORBOARD_DIR",
        "~/tensorboard",
    )
)

## my own checkpoint dir for research
BASE_CKPT_DIR = os.path.expanduser(
    os.environ.get("DEEPSPEC_CHECKPOINT_DIR", "~/checkpoints")
)

## auto eval
auto_eval_command = None