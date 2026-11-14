# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 metadata errors must fail before allocating the large checkpoint."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3.checkpoint import Exl3Checkpoint
from vllm.model_executor.layers.quantization.exl3.config import Exl3Config


def test_exl3_config_resolves_declared_layer_count_and_bits():
    config = get_quantization_config("exl3").from_config(
        {
            "quant_method": "exl3",
            "num_hidden_layers": 40,
            "tensor_storage": {
                "layers.0.ffn.w1": {"quant_format": "exl3", "bits_per_weight": 3},
                "layers.0.ffn.w3": {"quant_format": "exl3", "bits_per_weight": 3},
            },
        }
    )
    assert isinstance(config, Exl3Config)
    assert config.num_hidden_layers == 40
    assert config.bits_for_prefix("model.layers.0.ffn.gate_up_proj") == 3
    with pytest.raises(ValueError, match="no EXL3 bit-width metadata"):
        config.bits_for_prefix("model.layers.2.attn.wq_b")


@pytest.mark.parametrize("omit_scale", [False, True])
def test_exl3_checkpoint_requires_complete_packed_components(tmp_path, omit_scale):
    prefix = "layers.0.ffn.w1"
    tensors = {
        prefix + ".trellis": torch.zeros((2, 2, 48), dtype=torch.int16),
        prefix + ".suh": torch.ones(32, dtype=torch.float16),
        prefix + ".svh": torch.ones(32, dtype=torch.float16),
        prefix + ".mul1": torch.tensor([-2082680531], dtype=torch.int32),
    }
    if omit_scale:
        del tensors[prefix + ".suh"]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        )
    )
    (tmp_path / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "codebook": "mul1",
            }
        )
    )
    if omit_scale:
        with pytest.raises(ValueError, match="incomplete EXL3 component"):
            Exl3Checkpoint.parse(tmp_path)
    else:
        spec = Exl3Checkpoint.parse(tmp_path).specs[prefix]
        assert spec.bits == 3
        assert spec.bits_source == "trellis_shape"
        assert spec.tp_axis == "column"
        assert spec.codebook == "mul1"
