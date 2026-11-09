# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import ModelRegistry


def register():
    # Test directly passing the model
    from .my_qwen3 import MyQwen3ForCausalLM

    ModelRegistry.register_model("Qwen3ForCausalLM", MyQwen3ForCausalLM)

    # Test passing lazy model
    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_add_dummy_model.my_qwen_embedding:MyQwen2Embedding",
    )

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_add_dummy_model.my_qwen_vl:MyQwen2VL",
    )
