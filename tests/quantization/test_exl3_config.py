# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.config import LoadConfig
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.quantization.exl3.config import (
    should_skip_exl3_linear,
)
from vllm.model_executor.model_loader.weight_utils import get_quant_config


def test_exl3_registered() -> None:
    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is Exl3Config


def test_exl3_config_drops_tensor_ledger() -> None:
    config = Exl3Config.from_config(
        {
            "quant_method": "exl3",
            "avg_bits": 2.9,
            "bits": 3,
            "codebook": "mul1",
            "head_bits": 6,
            "mtp_bits": 4,
            "tensor_storage": {
                "layers.0.attn.wq_a": {
                    "quant_format": "exl3",
                    "bits_per_weight": 6,
                },
                "layers.0.attn.wkv": {
                    "quant_format": "exl3",
                    "bits_per_weight": 6,
                },
                "layers.1.ffn.shared_experts.w1": {
                    "quant_format": "exl3",
                    "bits_per_weight": 5,
                },
                "layers.1.ffn.shared_experts.w3": {
                    "quant_format": "exl3",
                    "bits_per_weight": 5,
                },
                "layers.0.attn.wo_a.slice.0": {
                    "quant_format": "exl3",
                    "bits_per_weight": 5,
                },
                "layers.0.attn.wo_a.slice.1": {
                    "quant_format": "exl3",
                    "bits_per_weight": 5,
                },
            },
        }
    )
    assert config.avg_bits == 2.9
    assert config.codebook == "mul1"
    assert "tensor_storage" not in config.raw_config
    assert config.override_quantization_method({"quant_method": "EXL3"}, None) == "exl3"
    assert config.bits_for_prefix("language_model.layers.0.attn.wq_a") == 6
    assert config.bits_for_prefix("model.layers.0.attn.fused_wqa_wkv") == 6
    assert config.bits_for_prefix("model.layers.1.ffn.shared_experts.gate_up_proj") == 5
    assert config.bits_for_prefix("model.layers.0.attn.wo_a") == 5
    assert config.grouped_slices_for_prefix("model.layers.0.attn.wo_a") == (
        "layers.0.attn.wo_a.slice.0",
        "layers.0.attn.wo_a.slice.1",
    )
    assert config.bits_for_prefix("language_model.lm_head") == 6
    assert config.bits_for_prefix("mtp.0.ffn.w1") == 4
    assert config.bits_for_prefix("model.main_proj") == 6
    assert config.bits_for_prefix("model.layers.40.attn.wq_a") == 6
    assert config.bits_for_moe_prefix("model.layers.40.ffn.experts") == 4
    assert len(config.grouped_slices_for_prefix("model.layers.40.attn.wo_a")) == 8


def test_exl3_uses_external_tensor_ledger_when_hf_config_is_summary(
    tmp_path,
) -> None:
    external = {
        "quant_method": "exl3",
        "avg_bits": 2.9,
        "tensor_storage": {
            "layers.0.attn.wq_a": {
                "quant_format": "exl3",
                "bits_per_weight": 6,
            },
            "layers.0.attn.wkv": {
                "quant_format": "exl3",
                "bits_per_weight": 6,
            },
        },
    }
    (tmp_path / "quantization_config.json").write_text(json.dumps(external))
    model_config = SimpleNamespace(
        quantization="exl3",
        quantization_config=None,
        hf_config=SimpleNamespace(
            quantization_config={"quant_method": "exl3", "avg_bits": 2.9},
            text_config=None,
        ),
        hf_overrides={},
        model=str(tmp_path),
        revision=None,
    )

    config = get_quant_config(model_config, LoadConfig())

    assert isinstance(config, Exl3Config)
    assert config.bits_for_prefix(
        "language_model.model.layers.0.attn.fused_wqa_wkv"
    ) == 6


def test_exl3_skip_policy() -> None:
    assert should_skip_exl3_linear("model.layers.0.mlp.gate")
    assert should_skip_exl3_linear("model.vision.proj")
    assert should_skip_exl3_linear("model.embed")
    assert not should_skip_exl3_linear("model.layers.1.engram.wkv")
    assert not should_skip_exl3_linear("model.layers.0.self_attn.o_proj")
