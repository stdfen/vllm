"""Build a deterministic manifest for an EXL3 safetensors checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

_METADATA_FILES = (
    "config.json",
    "quantization_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)
_PACKED_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")


class ManifestError(ValueError):
    """Raised when checkpoint metadata is incomplete or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc


def _sha256_file(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = limit
        while remaining is None or remaining > 0:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ManifestError(f"truncated safetensors prefix: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 1 or header_size > path.stat().st_size - 8:
                raise ManifestError(
                    f"invalid safetensors header size {header_size}: {path}"
                )
            raw_header = stream.read(header_size)
            if len(raw_header) != header_size:
                raise ManifestError(f"truncated safetensors header: {path}")
    except OSError as exc:
        raise ManifestError(f"cannot read safetensors header {path}: {exc}") from exc
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid safetensors JSON header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ManifestError(f"safetensors header is not an object: {path}")
    return header_size, header


def _tensor_role(name: str) -> str:
    if name.startswith("vision."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if ".engram." in name:
        return "engram"
    if ".ffn.experts." in name:
        return "routed_expert"
    if ".ffn.shared_experts." in name:
        return "shared_expert"
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn.compressor." in name:
        return "compressor"
    if ".attn.wo_a" in name or ".attn.wo_b" in name:
        return "o_proj"
    if ".attn." in name:
        return "attention"
    if name in {"head", "lm_head"} or name.startswith("head."):
        return "lm_head"
    if name == "embed" or name.startswith("embed."):
        return "embedding"
    return "other"


def _storage_summary(storage: dict[str, Any]) -> dict[str, Any]:
    format_counts: Counter[str] = Counter()
    bits_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    stored_tensor_count = 0
    packed_component_count = 0
    native_component_count = 0

    for component, spec in storage.items():
        if not isinstance(spec, dict):
            raise ManifestError(f"invalid tensor_storage entry: {component}")
        tensors = spec.get("stored_tensors", {})
        if not isinstance(tensors, dict):
            raise ManifestError(f"invalid stored_tensors entry: {component}")
        stored_tensor_count += len(tensors)
        role_counts[_tensor_role(component)] += 1
        quant_format = spec.get("quant_format")
        if quant_format is None:
            native_component_count += 1
            format_counts["native"] += 1
        else:
            packed_component_count += 1
            format_counts[str(quant_format)] += 1
            bits_counts[str(spec.get("bits_per_weight", "unknown"))] += 1

    return {
        "component_count": len(storage),
        "stored_tensor_count": stored_tensor_count,
        "packed_component_count": packed_component_count,
        "native_component_count": native_component_count,
        "format_counts": dict(sorted(format_counts.items())),
        "bits_per_weight_counts": dict(sorted(bits_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "tensor_storage_sha256": _canonical_sha256(storage),
    }


def build_manifest(
    model_dir: Path,
    *,
    hash_shards: bool = False,
    include_components: bool = False,
) -> dict[str, Any]:
    """Validate a checkpoint and return a deterministic manifest."""
    model_dir = model_dir.resolve()
    required = (
        model_dir / "config.json",
        model_dir / "quantization_config.json",
        model_dir / "model.safetensors.index.json",
    )
    missing_metadata = [path.name for path in required if not path.is_file()]
    if missing_metadata:
        raise ManifestError(f"missing metadata files: {', '.join(missing_metadata)}")

    config = _read_json(required[0])
    quant_config = _read_json(required[1])
    index = _read_json(required[2])
    if config.get("quantization_config", {}).get("quant_method") != "exl3":
        raise ManifestError("config.json does not declare quant_method=exl3")
    if quant_config.get("quant_method") != "exl3":
        raise ManifestError("quantization_config.json does not declare exl3")

    weight_map = index.get("weight_map")
    storage = quant_config.get("tensor_storage")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ManifestError("index weight_map is missing or empty")
    if not isinstance(storage, dict) or not storage:
        raise ManifestError("quantization tensor_storage is missing or empty")

    indexed_names = set(weight_map)
    stored_names: set[str] = set()
    stored_specs: dict[str, dict[str, Any]] = {}
    for component, spec in storage.items():
        tensors = spec.get("stored_tensors", {}) if isinstance(spec, dict) else {}
        if not isinstance(tensors, dict):
            raise ManifestError(f"invalid stored_tensors entry: {component}")
        overlap = stored_names.intersection(tensors)
        if overlap:
            raise ManifestError(f"duplicate stored tensor: {min(overlap)}")
        stored_names.update(tensors)
        stored_specs.update(tensors)
    if not stored_names.issubset(indexed_names):
        missing_from_index = sorted(stored_names - indexed_names)[:5]
        raise ManifestError(
            "tensor_storage contains tensors absent from index: "
            f"storage_only={missing_from_index}"
        )
    index_only_names = indexed_names - stored_names
    packed_index_only = sorted(
        name for name in index_only_names if name.rsplit(".", 1)[-1] in _PACKED_SUFFIXES
    )
    packed_index_only_set = set(packed_index_only)
    native_index_only = index_only_names - packed_index_only_set
    packed_groups: dict[str, set[str]] = {}
    for name in packed_index_only:
        prefix, suffix = name.rsplit(".", 1)
        packed_groups.setdefault(prefix, set()).add(suffix)
    for prefix, suffixes in packed_groups.items():
        required = {"trellis", "suh", "svh"}
        markers = suffixes.intersection({"mul1", "mcg"})
        if not required.issubset(suffixes) or len(markers) != 1:
            raise ManifestError(
                f"incomplete index-only packed component {prefix}: {sorted(suffixes)}"
            )

    shard_names = sorted(set(weight_map.values()))
    shard_records: list[dict[str, Any]] = []
    header_names: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise ManifestError(f"missing shard: {shard_name}")
        header_size, header = _read_safetensors_header(shard_path)
        tensors = {
            name: value for name, value in header.items() if name != "__metadata__"
        }
        overlap = header_names.intersection(tensors)
        if overlap:
            raise ManifestError(
                f"tensor occurs in multiple shard headers: {min(overlap)}"
            )
        header_names.update(tensors)
        expected = {name for name, value in weight_map.items() if value == shard_name}
        if set(tensors) != expected:
            raise ManifestError(
                f"index/header mismatch in {shard_name}: "
                f"index_only={sorted(expected - set(tensors))[:5]}, "
                f"header_only={sorted(set(tensors) - expected)[:5]}"
            )
        for tensor_name, tensor_spec in tensors.items():
            dtype_counts[str(tensor_spec.get("dtype", "unknown"))] += 1
            storage_spec = stored_specs.get(tensor_name)
            if storage_spec is not None and (
                list(tensor_spec.get("shape", [])) != storage_spec.get("shape")
            ):
                raise ManifestError(f"shape mismatch for {tensor_name}")
        record = {
            "file": shard_name,
            "size": shard_path.stat().st_size,
            "header_size": header_size,
            "header_sha256": _sha256_file(shard_path, limit=8 + header_size),
            "tensor_count": len(tensors),
        }
        if hash_shards:
            record["sha256"] = _sha256_file(shard_path)
        shard_records.append(record)

    if header_names != indexed_names:
        raise ManifestError("combined shard headers do not match checkpoint index")

    metadata_files = []
    for name in _METADATA_FILES:
        path = model_dir / name
        if path.is_file():
            metadata_files.append(
                {
                    "file": name,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )

    text_config = config.get("text_config", config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "architectures": config.get("architectures", []),
            "model_type": config.get("model_type"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "engram_layer_ids": text_config.get("engram_layer_ids", []),
        },
        "quantization": {
            "method": quant_config.get("quant_method"),
            "version": quant_config.get("version"),
            "codebook": quant_config.get("codebook"),
            "bits": quant_config.get("bits"),
            "head_bits": quant_config.get("head_bits"),
            "mtp_bits": quant_config.get("mtp_bits"),
            "packed_suffixes": list(_PACKED_SUFFIXES),
        },
        "index": {
            "declared_total_size": index.get("metadata", {}).get("total_size"),
            "tensor_count": len(weight_map),
            "shard_count": len(shard_names),
        },
        "storage": _storage_summary(storage),
        "index_only": {
            "packed_tensor_count": len(packed_index_only),
            "packed_component_count": len(packed_groups),
            "native_tensor_count": len(native_index_only),
            "role_counts": dict(
                sorted(Counter(_tensor_role(name) for name in index_only_names).items())
            ),
            "names_sha256": _canonical_sha256(sorted(index_only_names)),
        },
        "header_dtype_counts": dict(sorted(dtype_counts.items())),
        "metadata_files": metadata_files,
        "shards": shard_records,
    }
    if include_components:
        manifest["components"] = storage
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-shards", action="store_true")
    parser.add_argument("--include-components", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the manifest command."""
    args = _parse_args()
    try:
        manifest = build_manifest(
            args.model_dir,
            hash_shards=args.hash_shards,
            include_components=args.include_components,
        )
    except ManifestError as exc:
        raise SystemExit(f"EXL3 manifest validation failed: {exc}") from exc
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
