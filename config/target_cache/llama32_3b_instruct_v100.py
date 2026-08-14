from deepspec.utils.constant import LLAMA_3_2_3B_INSTRUCT


project_name = "deepspec"
exp_name = "llama32_3b_instruct_target_cache"
seed = 42

model = dict(
    target_model_name_or_path=LLAMA_3_2_3B_INSTRUCT,
    # Llama-3.2-3B has 28 decoder layers (indices 0..27).  These five
    # positions mirror DeepSpec's evenly spaced Qwen feature selection; the
    # final backbone output is cached separately as target_last_hidden_states.
    target_layer_ids=[1, 7, 13, 19, 25],
    # Tesla V100 has no native BF16 compute.  Cache serialization remains
    # DeepSpec's canonical BF16 format after the FP16 target forward pass.
    target_dtype="float16",
    local_files_only=True,
)

data = dict(
    chat_template="llama3",
    max_length=4096,
)
