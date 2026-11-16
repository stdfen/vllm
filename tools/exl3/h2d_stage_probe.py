# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path

import torch
from safetensors import safe_open


def _bytes_from_proc(path: str, keys: tuple[str, ...]) -> dict[str, int]:
    values: dict[str, int] = {}
    with open(path, encoding="utf-8") as proc_file:
        for line in proc_file:
            fields = line.split()
            name = fields[0].removesuffix(":")
            if name in keys:
                multiplier = 1024 if len(fields) > 2 and fields[2] == "kB" else 1
                values[name] = int(fields[1]) * multiplier
    return values


def _snapshot(label: str) -> dict[str, int | str]:
    memory = _bytes_from_proc(
        "/proc/meminfo", ("MemAvailable", "MemFree", "Cached")
    )
    process = _bytes_from_proc(
        "/proc/self/smaps_rollup", ("Rss", "Pss", "Anonymous")
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    free, total = torch.cuda.mem_get_info()
    return {
        "label": label,
        **{f"system_{key.lower()}": value for key, value in memory.items()},
        **{f"process_{key.lower()}": value for key, value in process.items()},
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "cuda_free": free,
        "cuda_total": total,
        "cuda_allocated": torch.cuda.memory_allocated(),
        "cuda_reserved": torch.cuda.memory_reserved(),
    }


def _locate_tensor(model: Path, tensor_name: str) -> Path:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    try:
        return model / index[tensor_name]
    except KeyError as exc:
        raise ValueError(f"tensor not found in checkpoint: {tensor_name}") from exc


def _staged_copy(
    destination: torch.Tensor, source: torch.Tensor, stage_bytes: int
) -> None:
    elements = max(1, stage_bytes // source.element_size())
    stage = torch.empty(elements, dtype=source.dtype, pin_memory=True)
    source_flat = source.view(-1)
    destination_flat = destination.view(-1)
    for offset in range(0, source_flat.numel(), elements):
        count = min(elements, source_flat.numel() - offset)
        stage_slice = stage[:count]
        stage_slice.copy_(source_flat[offset : offset + count])
        destination_flat[offset : offset + count].copy_(stage_slice)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument(
        "--mode", choices=("direct", "staged", "helper"), required=True
    )
    parser.add_argument("--stage-mib", type=int, default=64)
    args = parser.parse_args()

    if args.stage_mib <= 0:
        raise ValueError("--stage-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    shard = _locate_tensor(args.model, args.tensor)
    snapshots = [_snapshot("before_open")]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        source = handle.get_tensor(args.tensor)
        snapshots.append(_snapshot("after_map"))
        destination = torch.empty_like(source, device="cuda")
        snapshots.append(_snapshot("after_alloc"))
        if args.mode == "direct":
            destination.copy_(source)
        elif args.mode == "staged":
            _staged_copy(destination, source, args.stage_mib * 1024 * 1024)
        else:
            from vllm.model_executor.weight_staging import copy_loaded_weight

            copy_loaded_weight(destination, source)
        torch.cuda.synchronize()
        snapshots.append(_snapshot("after_copy"))

    print(
        json.dumps(
            {
                "mode": args.mode,
                "tensor": args.tensor,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "tensor_bytes": source.numel() * source.element_size(),
                "stage_bytes": args.stage_mib * 1024 * 1024,
                "pid": os.getpid(),
                "snapshots": snapshots,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
