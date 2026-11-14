# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real file-format and admission tests; no CUDA initialization required."""

import copy
import hashlib
import json

import pytest

from vllm.models.deepseek_v41.common.engram_pack import (
    HEADER_BYTES,
    atomic_json,
    checkpoint_identity,
    select_pack,
    tensor_source,
    validate_record,
    write_pack,
)
from vllm.models.deepseek_v41.common.engram_preflight import (
    calibrate,
    estimate,
)


@pytest.fixture
def packed_checkpoint(tmp_path):
    model, output = tmp_path / "model", tmp_path / "pack"
    model.mkdir()
    output.mkdir()
    tensors = {"weight": bytes(range(256)) * 6, "scale": bytes([0, 255]) * 24}
    header, offset = {}, 0
    names = {}
    for suffix, raw in tensors.items():
        name = f"layers.1.engram.embed.{suffix}"
        names[name] = "model.safetensors"
        header[name] = dict(
            dtype="F8_E4M3" if suffix == "weight" else "U8",
            shape=[6, 256 if suffix == "weight" else 8],
            data_offsets=[offset, offset + len(raw)],
        )
        offset += len(raw)
    raw = json.dumps(header).encode()
    (model / "model.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + b"".join(tensors.values())
    )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=names))
    )
    identity = checkpoint_identity(model)
    record = dict(
        version=1,
        header_bytes=HEADER_BYTES,
        layer=1,
        tp_size=2,
        tp_rank=1,
        row_start=3,
        rows=3,
        global_rows=6,
        dim=256,
        scale_dim=8,
        stride=512,
        file="rank1.bin",
        sources={
            s: tensor_source(model, f"layers.1.engram.embed.{s}") for s in tensors
        },
    )
    record = write_pack(model, output / record["file"], record, identity, chunk_rows=2)
    manifest = dict(schema=1, identity=identity, packs=[record])
    atomic_json(output / "manifest.json", manifest)
    return model, output, manifest, record, tensors


def test_pack_preserves_raw_rows_and_validates_runtime_rank(packed_checkpoint):
    model, output, manifest, record, tensors = packed_checkpoint
    path, selected = select_pack(
        output / "manifest.json", model, 1, 2, 1, 3, 3, 256, 8, "full"
    )
    raw = path.read_bytes()
    for row in range(3):
        source_row = row + 3
        packed = raw[HEADER_BYTES + row * 512 : HEADER_BYTES + (row + 1) * 512]
        assert (
            packed[:256] == tensors["weight"][source_row * 256 : (source_row + 1) * 256]
        )
        assert (
            packed[256:264] == tensors["scale"][source_row * 8 : (source_row + 1) * 8]
        )
        assert packed[264:] == bytes(248)
    assert selected["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="missing or duplicate"):
        select_pack(output / "manifest.json", model, 1, 2, 0, 0, 3, 256, 8, "full")
    with pytest.raises(ValueError, match="TP layout"):
        select_pack(output / "manifest.json", model, 1, 2, 1, 2, 3, 256, 8, "full")
    (model / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="different checkpoint"):
        select_pack(output / "manifest.json", model, 1, 2, 1, 3, 3, 256, 8, "full")


@pytest.mark.parametrize("damage", ["header", "truncated", "sample", "full"])
def test_pack_rejects_damage_before_cuda(packed_checkpoint, damage):
    _, output, manifest, record, _ = packed_checkpoint
    path = output / record["file"]
    raw = bytearray(path.read_bytes())
    if damage == "truncated":
        raw.pop()
    else:
        position = 0 if damage == "header" else HEADER_BYTES + 1
        raw[position] ^= 1
    path.write_bytes(raw)
    if damage == "full":
        record = copy.deepcopy(record)
        # Full integrity must reject damage even if supplied samples miss it.
        record["samples"] = [record["samples"][-1]]
    with pytest.raises(ValueError, match="mismatch|truncated"):
        validate_record(output, manifest, record, verification="full")


def test_uma_budget_does_not_double_count_and_rejects_insufficient_memory(tmp_path):
    recipe = dict(
        engine_argv=[
            "vllm",
            "serve",
            "model",
            "--engram-config",
            '{"disk_threads":32}',
            "--max-num-batched-tokens",
            "2048",
            "--max-model-len",
            "524288",
        ]
    )
    resources = tmp_path / "resources.jsonl"
    resources.write_text(
        "\n".join(
            json.dumps(dict(memory_bytes=s))
            for s in [
                dict(MemTotal=128 << 30, MemAvailable=120 << 30),
                dict(MemTotal=128 << 30, MemAvailable=20 << 30),
            ]
        )
    )
    profile = calibrate(resources, "model-identity", recipe, 64890)
    assert profile["measured_increment_bytes"] == 100 << 30
    result = estimate(
        profile,
        recipe,
        "model-identity",
        dict(MemTotal=128 << 30, MemAvailable=120 << 30),
        cache_bytes=64 << 20,
        reserve_bytes=2 << 30,
    )
    assert result["status"] == "WARN_UNVALIDATED_CONTEXT"
    result = estimate(
        profile,
        recipe,
        "model-identity",
        dict(MemTotal=128 << 30, MemAvailable=90 << 30),
        cache_bytes=64 << 20,
        reserve_bytes=2 << 30,
    )
    assert result["status"] == "FAIL_CAPACITY"
    with pytest.raises(ValueError, match="does not match"):
        estimate(
            profile,
            recipe,
            "other-model",
            dict(MemTotal=128 << 30),
            cache_bytes=0,
            reserve_bytes=0,
        )
