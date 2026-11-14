# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mixed checkpoint must retain FP8 settings and exact fractional EXL3 plans."""

from pathlib import Path

import pytest

from vllm.config import ModelConfig
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.deepseek_v41_exl3.plan import (
    Exl3MoEPlan,
    PlanError,
    tp_shard,
)


def test_model_config_accepts_mixed_checkpoint_override_order(tmp_path):
    fixture = Path(__file__).with_name("fixtures") / "deepseek_v41_exl3_config.json"
    (tmp_path / "config.json").write_bytes(fixture.read_bytes())
    config = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True)
    assert config.quantization == "deepseek_v41_exl3"
    assert config.architecture == "DeepseekV41ForCausalLM"
    assert config.hf_text_config.num_hidden_layers == 40


def test_mixed_config_preserves_fp8_base_and_exl3_head():
    config = get_quantization_config("deepseek_v41_exl3").from_config(
        {
            "quant_method": "deepseek_v41_exl3",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
            "exl3_moe": {"head": "exl3", "head_k": 5, "tp_layout": "channels"},
        }
    )
    assert config.get_name() == "deepseek_v41_exl3"
    assert config.is_checkpoint_fp8_serialized
    assert config.weight_block_size == [32, 32]
    assert config.exl3_settings.head == "exl3"
    assert config.exl3_settings.head_k == 5


def test_fractional_expert_bits_survive_plan_roundtrip():
    plan = Exl3MoEPlan.from_dict(
        {
            "num_layers": 1,
            "num_experts": 4,
            "hidden_size": 256,
            "moe_intermediate_size": 256,
            "layers": {"0": "ab34"},
        }
    )
    assert [b.k_gu for b in plan.layers[0].bits] == [1.5, 2.5, 3, 4]
    restored = Exl3MoEPlan.from_dict(plan.to_dict())
    assert restored.layers[0].bits == plan.layers[0].bits
    assert restored.bytes_per_rank(2) == plan.bytes_per_rank(2)
    assert tp_shard(256, 2, 0) == (0, 128)
    assert tp_shard(256, 2, 1) == (128, 256)


def test_tp_partition_rejects_partial_hadamard_blocks():
    with pytest.raises(PlanError, match="Hadamard"):
        tp_shard(2304, 4, 0)


def test_plan_rejects_missing_expert_entries():
    with pytest.raises(PlanError, match="expected 4"):
        Exl3MoEPlan.from_dict(
            {
                "num_layers": 1,
                "num_experts": 4,
                "hidden_size": 256,
                "moe_intermediate_size": 256,
                "layers": {"0": "233"},
            }
        )
