#!/usr/bin/env python3
"""Issue exact-token-count OpenAI completion probes against a vLLM server."""

import argparse
import json
import time
import urllib.request


def post(base: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        base + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/model")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed = "The quick brown fox jumps over the lazy dog. "
    token_ids = post(
        args.base,
        "/tokenize",
        {"model": args.model, "prompt": seed},
        args.timeout,
    )["tokens"]
    prompt = (token_ids * ((args.prompt_tokens + len(token_ids) - 1) // len(token_ids)))[
        : args.prompt_tokens
    ]
    started = time.monotonic()
    result = post(
        args.base,
        "/v1/completions",
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
        args.timeout,
    )
    record = {
        "target_prompt_tokens": args.prompt_tokens,
        "usage": result.get("usage"),
        "e2e_s": time.monotonic() - started,
        "text": result["choices"][0]["text"],
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
