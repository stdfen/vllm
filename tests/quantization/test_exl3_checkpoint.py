# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from pathlib import Path

import pytest

from vllm.model_executor.layers.quantization.exl3.checkpoint import Exl3Checkpoint


def _write_shard(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {
        name: {"dtype": "I16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def _checkpoint(tmp_path: Path, *, marker: str = "mul1", ledger: bool = True) -> Path:
    tensors = {
        "layers.0.mlp.down_proj.trellis": [320, 128, 48],
        "layers.0.mlp.down_proj.suh": [5120],
        "layers.0.mlp.down_proj.svh": [2048],
        f"layers.0.mlp.down_proj.{marker}": [],
        "embed.weight": [128, 64],
    }
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", tensors)
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    storage = {}
    if ledger:
        storage["layers.0.mlp.down_proj"] = {"bits_per_weight": 3}
    (tmp_path / "quantization_config.json").write_text(
        json.dumps(
            {"quant_method": "exl3", "codebook": marker, "tensor_storage": storage}
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize(
    "ledger,source", [(True, "metadata"), (False, "trellis_shape")]
)
def test_parse_packed_and_native(tmp_path: Path, ledger: bool, source: str) -> None:
    checkpoint = Exl3Checkpoint.parse(_checkpoint(tmp_path, ledger=ledger))
    packed = checkpoint.specs["layers.0.mlp.down_proj"]
    assert (packed.role, packed.bits, packed.bits_source) == ("packed", 3, source)
    assert packed.codebook == "mul1"
    assert packed.tp_axis == "row"
    assert checkpoint.specs["embed"].role == "native"


def test_rejects_ambiguous_marker(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = "layers.0.mlp.down_proj.mcg"
    index["weight_map"][name] = "model-00001-of-00001.safetensors"
    index_path.write_text(json.dumps(index))
    tensors = {tensor: [1] for tensor in index["weight_map"]}
    tensors.update(
        {
            "layers.0.mlp.down_proj.trellis": [320, 128, 48],
            "layers.0.mlp.down_proj.suh": [5120],
            "layers.0.mlp.down_proj.svh": [2048],
            "layers.0.mlp.down_proj.mul1": [],
            "layers.0.mlp.down_proj.mcg": [],
            "embed.weight": [128, 64],
        }
    )
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    with pytest.raises(ValueError, match="incomplete EXL3 component"):
        Exl3Checkpoint.parse(root)
