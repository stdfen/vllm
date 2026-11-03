# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.engram import EngramConfig
from vllm.models.deepseek_v41.common.engram_file import SafetensorsEngramStore


def test_file_backed_engram_config_requires_source_directory() -> None:
    with pytest.raises(ValueError, match="requires table_dir"):
        EngramConfig(file_backed=True)
    config = EngramConfig(file_backed=True, table_dir="/engram-src")
    assert config.cpu_offload
    assert not config.dp_shared_memory


def test_safetensors_engram_store_reads_only_selected_rows(tmp_path) -> None:
    values = torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32) / 32
    values = values.to(torch.float8_e4m3fn)
    scales = torch.tensor([[127], [128], [126], [129]], dtype=torch.uint8)
    shard = tmp_path / "model.safetensors"
    save_file(
        {
            "layers.1.engram.embed.weight": values,
            "layers.1.engram.embed.scale": scales,
        },
        shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.1.engram.embed.weight": shard.name,
                    "layers.1.engram.embed.scale": shard.name,
                }
            }
        )
    )
    store = SafetensorsEngramStore(
        tmp_path,
        "language_model.model.layers.1.engram.embed",
        1,
        4,
        32,
        32,
    )
    indices = torch.tensor([[1], [3]])
    output = torch.empty((2, 1, 32), dtype=torch.bfloat16)

    store.lookup(indices, 0, 1, output)

    scale_values = (scales[[1, 3]].to(torch.int32) << 23).view(torch.float32)
    expected = (values[[1, 3]].float() * scale_values).to(torch.bfloat16)
    torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)
    assert store.mapped_bytes == 3 * (32 + 1)


def test_safetensors_engram_store_rejects_ambiguous_wrapper_alias(tmp_path) -> None:
    values = torch.zeros((2, 32), dtype=torch.float8_e4m3fn)
    scales = torch.zeros((2, 1), dtype=torch.uint8)
    shard = tmp_path / "model.safetensors"
    save_file(
        {
            "language_model.model.layers.1.engram.embed.weight": values.clone(),
            "language_model.model.layers.1.engram.embed.scale": scales.clone(),
            "layers.1.engram.embed.weight": values,
            "layers.1.engram.embed.scale": scales,
        },
        shard,
    )
    weight_map = {
        name: shard.name
        for name in (
            "language_model.model.layers.1.engram.embed.weight",
            "language_model.model.layers.1.engram.embed.scale",
            "layers.1.engram.embed.weight",
            "layers.1.engram.embed.scale",
        )
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    with pytest.raises(ValueError, match="expected one Engram"):
        SafetensorsEngramStore(
            tmp_path,
            "language_model.model.layers.1.engram.embed",
            0,
            2,
            32,
            32,
        )
