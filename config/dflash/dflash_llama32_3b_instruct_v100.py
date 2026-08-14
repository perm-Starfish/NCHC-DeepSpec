import os

from deepspec.trainer import Llama3DSparkTrainer
from deepspec.utils.constant import (
    BASE_CKPT_DIR,
    BASE_TB_DIR,
    CACHE_DIR,
    LLAMA_3_2_3B_INSTRUCT,
)


project_name = "deepspec"
exp_name = "dflash_block7_llama32_3b_instruct_v100"
seed = 42

model = dict(
    target_model_name_or_path=LLAMA_3_2_3B_INSTRUCT,
    local_files_only=True,
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 7, 13, 19, 25],
    mask_token_id=128002,
    num_anchors=64,
    markov_rank=0,
    confidence_head_alpha=0.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Llama3DSparkTrainer,
    lr=6.0e-4,
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
