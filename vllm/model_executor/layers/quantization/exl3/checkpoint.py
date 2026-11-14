# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_PACKED_ROLES = ("trellis", "suh", "svh", "mul1", "mcg")


@dataclass(frozen=True)
class Exl3TensorSpec:
    logical_prefix: str
    role: Literal["packed", "native"]
    bits: int | None
    bits_source: Literal["metadata", "trellis_shape", "native"]
    codebook: Literal["mul1", "mcg"] | None
    shapes: dict[str, tuple[int, ...]]
    tp_axis: Literal["row", "column", "replicated"]
    source_shards: tuple[str, ...]


@dataclass(frozen=True)
class Exl3Checkpoint:
    model_dir: Path
    specs: dict[str, Exl3TensorSpec]

    @classmethod
    def parse(cls, model_dir: str | Path) -> Exl3Checkpoint:
        root = Path(model_dir)
        quant = _read_json(root / "quantization_config.json")
        if str(quant.get("quant_method", "")).lower() != "exl3":
            raise ValueError("quantization_config.json does not declare EXL3")
        index = _read_json(root / "model.safetensors.index.json")
        weight_map: dict[str, str] = index.get("weight_map", {})
        if not weight_map:
            raise ValueError("checkpoint index has no weight_map")

        headers: dict[str, dict[str, Any]] = {}
        for shard in sorted(set(weight_map.values())):
            headers[shard] = _read_safetensors_header(root / shard)

        groups: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
        for name, shard in weight_map.items():
            if name not in headers[shard]:
                raise ValueError(f"{name} is missing from {shard} header")
            shape = tuple(headers[shard][name].get("shape", ()))
            prefix, suffix = _split_name(name)
            groups.setdefault(prefix, {})[suffix] = (shape, shard)

        ledger = quant.get("tensor_storage") or {}
        specs = {
            prefix: _make_spec(prefix, tensors, ledger.get(prefix))
            for prefix, tensors in groups.items()
        }
        return cls(root, specs)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def _split_name(name: str) -> tuple[str, str]:
    prefix, separator, suffix = name.rpartition(".")
    if separator and suffix in _PACKED_ROLES:
        return prefix, suffix
    if separator and suffix == "weight":
        return prefix, "weight"
    return name, "native"


def _make_spec(
    prefix: str,
    tensors: dict[str, tuple[tuple[int, ...], str]],
    ledger_entry: dict[str, Any] | None,
) -> Exl3TensorSpec:
    packed = set(tensors).intersection(_PACKED_ROLES)
    if not packed:
        return Exl3TensorSpec(
            prefix,
            "native",
            None,
            "native",
            None,
            {key: value[0] for key, value in tensors.items()},
            "replicated",
            tuple(sorted({value[1] for value in tensors.values()})),
        )
    required = {"trellis", "suh", "svh"}
    markers = packed.intersection(("mul1", "mcg"))
    if not required.issubset(packed) or len(markers) != 1:
        raise ValueError(f"incomplete EXL3 component {prefix}: {sorted(packed)}")
    marker = next(iter(markers))
    metadata_bits = (ledger_entry or {}).get("bits_per_weight")
    if metadata_bits is not None:
        bits, source = int(metadata_bits), "metadata"
    else:
        trellis_shape = tensors["trellis"][0]
        if len(trellis_shape) != 3 or trellis_shape[-1] % 16:
            raise ValueError(f"cannot infer EXL3 bits for {prefix}: {trellis_shape}")
        bits, source = trellis_shape[-1] // 16, "trellis_shape"
    if not 2 <= bits <= 8:
        raise ValueError(f"unsupported EXL3 bits for {prefix}: {bits}")
    return Exl3TensorSpec(
        prefix,
        "packed",
        bits,
        source,
        marker,
        {key: value[0] for key, value in tensors.items()},
        _tp_axis(prefix),
        tuple(sorted({value[1] for value in tensors.values()})),
    )


def _tp_axis(prefix: str) -> Literal["row", "column", "replicated"]:
    name = prefix.lower()
    if any(token in name for token in ("down_proj", ".w2", ".o_proj")):
        return "row"
    if any(
        token in name
        for token in ("gate_proj", "up_proj", ".w1", ".w3", "lm_head", "head")
    ):
        return "column"
    return "replicated"
