import json
import struct
from pathlib import Path

import pytest

from tools.exl3.manifest import ManifestError, build_manifest


def _write_safetensors(path: Path, tensors: dict[str, dict]) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)


def _checkpoint(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "model_type": "deepseek_v41",
                "text_config": {"num_hidden_layers": 1, "engram_layer_ids": []},
                "quantization_config": {"quant_method": "exl3"},
            }
        )
    )
    storage = {
        "layers.0.attn.wq_a": {
            "quant_format": "exl3",
            "bits_per_weight": 5,
            "stored_tensors": {
                "layers.0.attn.wq_a.trellis": {
                    "shape": [1, 1, 1],
                    "n_bytes": 2,
                    "dtype": "torch.int16",
                },
                "layers.0.attn.wq_a.mul1": {
                    "shape": [],
                    "n_bytes": 4,
                    "dtype": "torch.int32",
                },
            },
        },
        "embed": {
            "stored_tensors": {
                "embed.weight": {
                    "shape": [2, 2],
                    "n_bytes": 8,
                    "dtype": "torch.bfloat16",
                }
            }
        },
    }
    (tmp_path / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "version": "1.6.2",
                "codebook": "mul1",
                "tensor_storage": storage,
            }
        )
    )
    weight_map = {
        "layers.0.attn.wq_a.trellis": "model-00001-of-00001.safetensors",
        "layers.0.attn.wq_a.mul1": "model-00001-of-00001.safetensors",
        "embed.weight": "model-00001-of-00001.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 14}, "weight_map": weight_map})
    )
    _write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "layers.0.attn.wq_a.trellis": {
                "dtype": "I16",
                "shape": [1, 1, 1],
                "data_offsets": [0, 2],
            },
            "layers.0.attn.wq_a.mul1": {
                "dtype": "I32",
                "shape": [],
                "data_offsets": [2, 6],
            },
            "embed.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [6, 14],
            },
        },
    )
    return tmp_path


def test_build_manifest_classifies_and_hashes_checkpoint(tmp_path: Path) -> None:
    manifest = build_manifest(_checkpoint(tmp_path))

    assert manifest["index"] == {
        "declared_total_size": 14,
        "tensor_count": 3,
        "shard_count": 1,
    }
    assert manifest["storage"]["packed_component_count"] == 1
    assert manifest["storage"]["native_component_count"] == 1
    assert manifest["storage"]["bits_per_weight_counts"] == {"5": 1}
    assert manifest["storage"]["role_counts"] == {"attention": 1, "embedding": 1}
    assert len(manifest["manifest_sha256"]) == 64


def test_build_manifest_rejects_index_header_mismatch(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["unexpected.weight"] = "model-00001-of-00001.safetensors"
    index_path.write_text(json.dumps(index))

    with pytest.raises(ManifestError, match="index/header mismatch"):
        build_manifest(model_dir)


def test_build_manifest_accepts_index_only_native_tensor(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path)
    storage_path = model_dir / "quantization_config.json"
    storage = json.loads(storage_path.read_text())
    del storage["tensor_storage"]["embed"]
    storage_path.write_text(json.dumps(storage))

    manifest = build_manifest(model_dir)

    assert manifest["index_only"]["native_tensor_count"] == 1
    assert manifest["index_only"]["packed_tensor_count"] == 0
    assert manifest["index_only"]["role_counts"] == {"embedding": 1}


def test_build_manifest_rejects_shape_mismatch(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path)
    shard = model_dir / "model-00001-of-00001.safetensors"
    _write_safetensors(
        shard,
        {
            "layers.0.attn.wq_a.trellis": {
                "dtype": "I16",
                "shape": [2, 1, 1],
                "data_offsets": [0, 4],
            },
            "layers.0.attn.wq_a.mul1": {
                "dtype": "I32",
                "shape": [],
                "data_offsets": [4, 8],
            },
            "embed.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [8, 16],
            },
        },
    )

    with pytest.raises(ManifestError, match="shape mismatch"):
        build_manifest(model_dir)
