#!/usr/bin/env python3
"""Replay MiaAI's published cooperative-MoE workloads against an OpenAI API."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request(base: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    finished = None
    chunks: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                finished = time.perf_counter()
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning") or ""
                if text:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(text)
    finished = finished or time.perf_counter()
    if first is None:
        first = finished
    output_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(finished - first, 1e-9)
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": output_tokens,
        "ttft_s": first - started,
        "e2e_s": finished - started,
        "decode_s": decode_seconds,
        "decode_tps_client": max(output_tokens - 1, 0) / decode_seconds,
        "response_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/model")
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.workloads.read_text(encoding="utf-8"))

    warmup = spec["warmup"]
    request(args.base, {
        "model": args.model,
        "messages": [{"role": "user", "content": warmup["prompt"]}],
        "max_tokens": warmup["max_tokens"],
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "seed": warmup["seed"],
        "stream": True,
        "stream_options": spec["stream_options"],
        "chat_template_kwargs": {"enable_thinking": warmup["thinking"]},
    })

    result = {"protocol": spec, "cases": {}}
    for name, case in spec["cases"].items():
        runs = []
        for seed in spec["seeds"]:
            measured = request(args.base, {
                "model": args.model,
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": spec["max_tokens"],
                "temperature": spec["temperature"],
                "top_p": spec["top_p"],
                "seed": seed,
                "stream": spec["stream"],
                "stream_options": spec["stream_options"],
                "chat_template_kwargs": {"enable_thinking": case["thinking"]},
            })
            measured["seed"] = seed
            runs.append(measured)
            print(name, seed, json.dumps(measured), flush=True)
        result["cases"][name] = {
            "runs": runs,
            "median_decode_tps_client": statistics.median(
                item["decode_tps_client"] for item in runs
            ),
            "median_ttft_s": statistics.median(item["ttft_s"] for item in runs),
        }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
