import os

from deepspec.trainer import Llama3Eagle3Trainer
from deepspec.utils.constant import (
    BASE_CKPT_DIR,
    BASE_TB_DIR,
    CACHE_DIR,
    LLAMA_3_2_3B_INSTRUCT,
)


project_name = "deepspec"
exp_name = "eagle3_ttt7_llama32_3b_instruct_v100_lr5e5"
seed = 0

model = dict(
    target_model_name_or_path=LLAMA_3_2_3B_INSTRUCT,
    local_files_only=True,
    target_layer_ids=[1, 7, 13, 19, 25],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=Llama3Eagle3Trainer,
    lr=5.0e-5,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="fp16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=50,
)

data = dict(
    target_cache_path=os.path.join(
        CACHE_DIR,
        "llama32_3b_instruct_target_cache",
    ),
    chat_template="llama3",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    resolved_project_name = str(cfg["project_name"])
    resolved_exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        resolved_project_name,
        resolved_exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        resolved_project_name,
        resolved_exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
