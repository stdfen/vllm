# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--elements", type=int, default=262_144)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    value = torch.full((args.elements,), float(rank + 1), device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if not torch.all(value == expected).item():
        raise RuntimeError(f"NCCL warmup mismatch on rank {rank}")

    start = time.perf_counter()
    for _ in range(args.iterations):
        value.fill_(rank + 1)
        dist.all_reduce(value)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    passed = bool(torch.all(value == expected).item())
    print(
        json.dumps(
            {
                "backend": dist.get_backend(),
                "bytes": value.numel() * value.element_size(),
                "elapsed_s": elapsed,
                "iterations": args.iterations,
                "passed": passed,
                "rank": rank,
                "world_size": dist.get_world_size(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
