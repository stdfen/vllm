#!/usr/bin/env python3
"""Validate bounded-RSS row reads from the native DeepSeek Engram table."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.models.deepseek_v41.common.engram_file import SafetensorsEngramStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--rows", type=int, default=384006168)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> None:
    args = parse_args()
    before = rss_bytes()
    prefix = f"layers.{args.layer}.engram.embed"
    store = SafetensorsEngramStore(args.model, prefix, 0, args.rows, 256, 32)
    mapped = rss_bytes()
    device = torch.device(args.device)
    row_ids = torch.tensor([[0], [1], [100], [args.rows - 1]], device=device)
    output = torch.empty((4, 1, 256), dtype=torch.bfloat16, device=device)
    store.lookup(row_ids, 0, 1, output)
    if device.type == "cuda":
        torch.cuda.synchronize()
    after = rss_bytes()

    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weight_name, scale_name = f"{prefix}.weight", f"{prefix}.scale"
    with safe_open(
        args.model / index[weight_name], framework="pt", device="cpu"
    ) as handle:
        values = handle.get_tensor(weight_name)[row_ids.cpu().flatten()]
    with safe_open(
        args.model / index[scale_name], framework="pt", device="cpu"
    ) as handle:
        scales = handle.get_tensor(scale_name)[row_ids.cpu().flatten()]
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    decoded = values.to(device).float() * (
        (scales.to(device).to(torch.int32) << 23)
        .view(torch.float32)
        .repeat_interleave(32, dim=-1)
    )
    expected = decoded.to(torch.bfloat16).unsqueeze(1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    rss_growth = after - before
    passed = rss_growth < 512 * 1024**2
    result = {
        "passed": passed,
        "layer": args.layer,
        "rows": args.rows,
        "mapped_bytes": store.mapped_bytes,
        "rss_before": before,
        "rss_after_mapping": mapped,
        "rss_after_lookup": after,
        "rss_growth": rss_growth,
        "output_shape": list(output.shape),
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
