# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned rank-local Engram storage. No CUDA or vLLM initialization required."""

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

MAGIC = b"VENGRAM1"
HEADER_BYTES = 4096
HEADER = struct.Struct("<8s11Q32s")
GEOMETRY = (
    "version",
    "header_bytes",
    "layer",
    "tp_size",
    "tp_rank",
    "row_start",
    "rows",
    "global_rows",
    "dim",
    "scale_dim",
    "stride",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def checkpoint_identity(root):
    root = Path(root)
    files = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "exl3_moe_plan.json",
    )
    hashes = {name: digest(root / name) for name in files if (root / name).is_file()}
    if not all(name in hashes for name in files[:4]):
        raise ValueError(
            "Checkpoint identity requires config, index and tokenizer files"
        )
    return hashlib.sha256(canonical(hashes)).hexdigest()


def tensor_source(root, name):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    relative = index["weight_map"][name]
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint tensor path escapes model directory")
    with path.open("rb") as stream:
        raw = stream.read(8)
        size = int.from_bytes(raw, "little")
        if len(raw) != 8 or not 0 < size <= 128 << 20:
            raise ValueError("Invalid safetensors header")
        header = stream.read(size)
        info = json.loads(header)[name]
    start, end = info["data_offsets"]
    if not 0 <= start <= end <= path.stat().st_size - 8 - size:
        raise ValueError("Invalid tensor data range")
    return dict(
        file=relative,
        offset=8 + size + start,
        bytes=end - start,
        shape=info["shape"],
        dtype=info["dtype"],
        file_bytes=path.stat().st_size,
        header_sha256=hashlib.sha256(raw + header).hexdigest(),
    )


def pack_header(record, identity):
    return HEADER.pack(
        MAGIC, *(record[k] for k in GEOMETRY), bytes.fromhex(identity)
    ).ljust(HEADER_BYTES, b"\0")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def validate_record(root, manifest, record, *, verification="sample"):
    """Validate geometry, source descriptors and pack bytes before opening CUDA."""
    root = Path(root)
    if manifest.get("schema") != 1 or verification not in ("full", "sample"):
        raise ValueError("Unsupported Engram manifest/verification version")
    if record["version"] != 1 or record["header_bytes"] != HEADER_BYTES:
        raise ValueError("Unsupported Engram pack header")
    if not (
        0 <= record["tp_rank"] < record["tp_size"]
        and record["rows"] > 0
        and 0 <= record["row_start"] < record["global_rows"]
        and record["row_start"] + record["rows"] <= record["global_rows"]
        and record["stride"] >= record["dim"] + record["scale_dim"]
        and record["stride"] % 512 == 0
    ):
        raise ValueError("Invalid Engram pack geometry")
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Engram pack escapes manifest directory")
    expected_size = HEADER_BYTES + record["rows"] * record["stride"]
    if path.stat().st_size != expected_size:
        raise ValueError("Engram pack truncated or wrong geometry")
    with path.open("rb") as stream:
        if stream.read(HEADER_BYTES) != pack_header(record, manifest["identity"]):
            raise ValueError("Engram pack header/identity mismatch")
        for sample in record["samples"]:
            stream.seek(HEADER_BYTES + sample["row"] * record["stride"])
            if (
                hashlib.sha256(stream.read(record["stride"])).hexdigest()
                != sample["sha256"]
            ):
                raise ValueError("Engram pack sample mismatch")
    if verification == "full" and digest(path) != record["sha256"]:
        raise ValueError("Engram pack full SHA256 mismatch")
    return path


def select_pack(
    manifest_path,
    model_root,
    layer,
    tp_size,
    tp_rank,
    row_start,
    rows,
    dim,
    scale_dim,
    verification,
):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if checkpoint_identity(model_root) != manifest["identity"]:
        raise ValueError("Engram manifest belongs to a different checkpoint/tokenizer")
    matches = [
        r
        for r in manifest["packs"]
        if (r["layer"], r["tp_size"], r["tp_rank"]) == (layer, tp_size, tp_rank)
    ]
    if len(matches) != 1:
        raise ValueError("Engram manifest missing or duplicate rank/layer")
    record = matches[0]
    if tuple(record[k] for k in ("row_start", "rows", "dim", "scale_dim")) != (
        row_start,
        rows,
        dim,
        scale_dim,
    ):
        raise ValueError("Engram manifest does not match runtime TP layout")
    for suffix, source in record["sources"].items():
        actual = tensor_source(model_root, f"layers.{layer}.engram.embed.{suffix}")
        if actual != source:
            raise ValueError("Engram checkpoint source descriptor changed")
        width = dim if suffix == "weight" else scale_dim
        if source["shape"] != [record["global_rows"], width]:
            raise ValueError("Engram manifest global geometry differs from source")
    # Bind sampled pack rows to the original checkpoint, without claiming a
    # full source-payload comparison. Full verification hashes the PACK file.
    pack_path = validate_record(
        path.parent, manifest, record, verification=verification
    )
    with pack_path.open("rb") as packed:
        for sample in record["samples"]:
            packed.seek(HEADER_BYTES + sample["row"] * record["stride"])
            raw = packed.read(record["stride"])
            cursor = 0
            for suffix, width in (("weight", dim), ("scale", scale_dim)):
                source = record["sources"][suffix]
                with (Path(model_root) / source["file"]).open("rb") as original:
                    original.seek(
                        source["offset"] + (row_start + sample["row"]) * width
                    )
                    if original.read(width) != raw[cursor : cursor + width]:
                        raise ValueError("Engram pack does not match checkpoint sample")
                cursor += width
    return pack_path, record


def write_pack(model_root, output, record, identity, *, chunk_rows=32768):
    """Stream raw weight/scale bytes into interleaved rows without dequantizing."""
    import numpy as np

    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".part")
    required = HEADER_BYTES + record["rows"] * record["stride"]
    free = shutil.disk_usage(output.parent).free
    if free < required + (8 << 30):
        raise ValueError(
            f"Insufficient storage: need {required} bytes plus 8 GiB reserve"
        )
    sources = record["sources"]
    streams = [
        (Path(model_root) / sources[k]["file"]).open("rb") for k in ("weight", "scale")
    ]
    hasher = hashlib.sha256()
    sample_rows = sorted({0, record["rows"] // 2, record["rows"] - 1})
    samples = []
    try:
        with temporary.open("xb") as stream:
            header = pack_header(record, identity)
            stream.write(header)
            hasher.update(header)
            for local in range(0, record["rows"], chunk_rows):
                count = min(chunk_rows, record["rows"] - local)
                data = np.zeros((count, record["stride"]), dtype=np.uint8)
                offset = 0
                for name, source_stream, width in zip(
                    ("weight", "scale"), streams, (record["dim"], record["scale_dim"])
                ):
                    source_stream.seek(
                        sources[name]["offset"] + (record["row_start"] + local) * width
                    )
                    raw = source_stream.read(count * width)
                    if len(raw) != count * width:
                        raise ValueError("Truncated Engram source while packing")
                    data[:, offset : offset + width] = np.frombuffer(
                        raw, np.uint8
                    ).reshape(count, width)
                    offset += width
                view = memoryview(data)
                stream.write(view)
                hasher.update(view)
                for row in sample_rows:
                    if local <= row < local + count:
                        samples.append(
                            dict(
                                row=row,
                                sha256=hashlib.sha256(
                                    data[row - local].tobytes()
                                ).hexdigest(),
                            )
                        )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        result = dict(record, sha256=hasher.hexdigest(), samples=samples)
        validate_record(
            output.parent,
            dict(schema=1, identity=identity),
            result,
            verification="full",
        )
        return result
    finally:
        for stream in streams:
            stream.close()
