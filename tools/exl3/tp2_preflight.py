#!/usr/bin/env python3
"""Fail-closed preflight for the two-node GB10 EXL3 deployment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from vllm.models.deepseek_v41.common.engram_file import (
    resolve_engram_tensor_names,
)

from manifest import ManifestError, build_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engram-model", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29501)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--min-cuda-free-gib", type=float, default=110.0)
    parser.add_argument(
        "--expected-manifest",
        default="4a450ef6c7c4c96d68daad649188fcf817549c77cbb4fa6c1221a2baea2e5ae8",
    )
    parser.add_argument(
        "--expected-extension",
        default="fa5167ae316e7015b5ffb96177a56fee6d88efacac188611632623c3345b2cdb",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    checks: dict[str, object] = {}
    failures: list[str] = []

    try:
        manifest = build_manifest(args.model)
        actual_manifest = manifest["manifest_sha256"]
        checks["model_manifest"] = actual_manifest
        if actual_manifest != args.expected_manifest:
            failures.append("EXL3 checkpoint manifest mismatch")
    except (ManifestError, OSError) as exc:
        checks["model_manifest_error"] = str(exc)
        failures.append("EXL3 checkpoint validation failed")

    try:
        index = json.loads(
            (args.engram_model / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )["weight_map"]
        required = []
        for layer in (1, 14):
            required.extend(
                resolve_engram_tensor_names(
                    index, f"language_model.model.layers.{layer}.engram.embed"
                )
            )
        checks["engram_required_tensors"] = sorted(required)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        checks["engram_index_error"] = str(exc)
        failures.append("native Engram index validation failed")

    extension_spec = importlib.util.find_spec("exllamav3_ext")
    if extension_spec is None or extension_spec.origin is None:
        failures.append("exllamav3_ext is not installed")
    else:
        extension_hash = _sha256(Path(extension_spec.origin))
        checks["extension_sha256"] = extension_hash
        if extension_hash != args.expected_extension:
            failures.append("ExLlamaV3 extension hash mismatch")

    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        free, total = torch.cuda.mem_get_info(0)
        checks["cuda"] = {
            "name": device_name,
            "capability": list(capability),
            "free_bytes": free,
            "total_bytes": total,
        }
        if "GB10" not in device_name or capability < (12, 1):
            failures.append("GPU is not the validated GB10/SM121 target")
        if free < args.min_cuda_free_gib * 1024**3:
            failures.append(
                f"CUDA free memory is below {args.min_cuda_free_gib:.1f} GiB"
            )

    try:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=str(args.model),
            quantization="exl3",
            tensor_parallel_size=2,
            nnodes=2,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            engram_config={
                "file_backed": True,
                "table_dir": str(args.engram_model),
                "cpu_offload": True,
            },
        )
        config = engine_args.create_engine_config()
        checks["engine_config"] = {
            "architecture": config.model_config.architectures,
            "quantization": config.model_config.quantization,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "nnodes": config.parallel_config.nnodes,
            "node_rank": config.parallel_config.node_rank,
            "engram_file_backed": config.engram_config.file_backed,
        }
    except Exception as exc:
        checks["engine_config_error"] = repr(exc)
        failures.append("vLLM engine configuration did not resolve")

    result = {
        "passed": not failures,
        "node_rank": args.node_rank,
        "checks": checks,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
