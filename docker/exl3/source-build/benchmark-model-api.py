"""Measure a bounded warm text workload; keep credentials out of the receipt."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import runpy
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    helper = runpy.run_path(
        str(pathlib.Path(__file__).with_name("verify-model-api.py"))
    )
    state = helper["container_state"](args.container)
    if not state["State"]["Running"] or args.repeats < 3:
        raise RuntimeError("Running candidate and at least three repeats required")
    environment = dict(item.split("=", 1) for item in state["Config"].get("Env", []))
    headers = {"Content-Type": "application/json"}
    if environment.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + environment["VLLM_API_KEY"]

    def request(payload):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        query = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with opener.open(query, timeout=300) as response:
            result = json.load(response)
        choice = result["choices"][0]["message"]
        answer = (choice.get("reasoning") or "") + (choice.get("content") or "")
        tokens = result["usage"]["completion_tokens"]
        if not answer or tokens <= 0:
            raise RuntimeError("Generation returned no output")
        return (
            tokens,
            hashlib.sha256(answer.encode()).hexdigest(),
            result["usage"]["prompt_tokens"],
        )

    rows = []
    for name, count, concurrency in (
        ("short-c1", 1, 1),
        ("medium-c1", 128, 1),
        ("medium-c4", 128, 4),
        ("long-c1", 512, 1),
    ):
        text = (
            "Matrix multiplication combines rows and columns into dot products. "
            * count
            + "Explain matrix multiplication in detail with an example."
        )
        payload = {
            "model": "dsv41",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 128,
            "temperature": 0,
            "seed": 7,
            "stream": False,
            "ignore_eos": True,
        }
        samples = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for repeat in range(args.repeats + 1):
                started = time.monotonic()
                outputs = list(pool.map(request, [payload] * concurrency))
                seconds = time.monotonic() - started
                if repeat:  # The first batch primes shapes and prefix caches.
                    samples.append(
                        {
                            "seconds": seconds,
                            "completionTokens": sum(item[0] for item in outputs),
                            "tokensPerSecond": sum(item[0] for item in outputs)
                            / seconds,
                            "outputSha256": [item[1] for item in outputs],
                            "promptTokens": [item[2] for item in outputs],
                        }
                    )
        rows.append(
            {
                "name": name,
                "concurrency": concurrency,
                "payloadSha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
                "medianTokensPerSecond": statistics.median(
                    row["tokensPerSecond"] for row in samples
                ),
                "samples": samples,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "imageId": state["Image"],
        "status": "measured",
        "scope": "Warm text API throughput; excludes vision and maximum context",
        "repeats": args.repeats,
        "cases": rows,
    }
    if args.reference:
        reference = json.loads(args.reference.read_text())
        comparisons = []
        for row, original in zip(rows, reference["cases"], strict=True):
            if row["payloadSha256"] != original["payloadSha256"]:
                raise RuntimeError("Benchmark payload differs from reference")
            ratio = row["medianTokensPerSecond"] / original["medianTokensPerSecond"]
            comparisons.append({"name": row["name"], "throughputRatio": ratio})
        result["referenceImageId"] = reference["imageId"]
        result["comparisons"] = comparisons
        result["threePercentGatePassed"] = all(
            row["throughputRatio"] >= 0.97 for row in comparisons
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
