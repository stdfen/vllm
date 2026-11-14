# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA admission diagnostics, without changing KV budgets or scheduling."""

import argparse
import hashlib
import json
from pathlib import Path

from vllm.models.deepseek_v41.common.engram_pack import (
    atomic_json,
    canonical,
    checkpoint_identity,
    select_pack,
)

GIB = 1 << 30


def common_recipe(recipe):
    argv = list(recipe["engine_argv"])
    position = argv.index("--engram-config") + 1
    config = json.loads(argv[position])
    for key in (
        "disk_pack_manifest",
        "disk_cache_bytes",
        "disk_verify",
        "disk_stats_dir",
    ):
        config.pop(key, None)
    argv[position] = canonical(config).decode()
    return hashlib.sha256(canonical(argv)).hexdigest()


def memory_info(path=Path("/proc/meminfo")):
    return {
        parts[0].rstrip(":"): int(parts[1]) * 1024
        for line in path.read_text().splitlines()
        if (parts := line.split()) and len(parts) >= 3 and parts[2] == "kB"
    }


def calibrate(resources, identity, recipe, validated_context):
    samples = [
        json.loads(line)["memory_bytes"]
        for line in Path(resources).read_text().splitlines()
    ]
    samples = [s for s in samples if "MemAvailable" in s and "MemTotal" in s]
    if len(samples) < 2:
        raise ValueError("A profile needs before-start and running samples")
    total = samples[0]["MemTotal"]
    used = [s["MemTotal"] - s["MemAvailable"] for s in samples]
    return dict(
        schema=1,
        identity=identity,
        common_recipe=common_recipe(recipe),
        mem_total=total,
        measured_increment_bytes=max(used) - used[0],
        observed_peak_used_bytes=max(used),
        samples=len(samples),
        validated_context=validated_context,
        limitation="1 Hz empirical envelope; not a proof of all transient peaks",
    )


def estimate(
    profile,
    recipe,
    identity,
    memory,
    *,
    cache_bytes,
    reserve_bytes,
    sampling_margin=4 * GIB,
):
    if (
        profile["schema"] != 1
        or profile["identity"] != identity
        or profile["common_recipe"] != common_recipe(recipe)
        or profile["mem_total"] != memory["MemTotal"]
    ):
        raise ValueError(
            "A calibration profile does not match model, recipe or UMA hardware"
        )
    argv = recipe["engine_argv"]
    options = json.loads(argv[argv.index("--engram-config") + 1])
    tokens = int(argv[argv.index("--max-num-batched-tokens") + 1])
    context = int(argv[argv.index("--max-model-len") + 1])
    # Conservative for this model: 24 heads/layer instead of the actual 12/rank.
    # Includes sort/group metadata, thread scratch, status, and allocator margin.
    reader_extra = 2 * (
        cache_bytes + tokens * 24 * 32 + options["disk_threads"] * 512 + (1 << 20)
    )
    # Separate policy margin covers sampling blind spots, not a measured workspace.
    required = (
        profile["measured_increment_bytes"]
        + reader_extra
        + sampling_margin
        + reserve_bytes
    )
    status = "PASS" if memory["MemAvailable"] >= required else "FAIL_CAPACITY"
    if status == "PASS" and context > profile["validated_context"]:
        status = "WARN_UNVALIDATED_CONTEXT"
    return dict(
        status=status,
        mem_available_bytes=memory["MemAvailable"],
        required_available_bytes=required,
        measured_A_increment_bytes=profile["measured_increment_bytes"],
        bounded_B_extra_bytes=reader_extra,
        sampling_policy_margin_bytes=sampling_margin,
        reserve_policy_bytes=reserve_bytes,
        validated_context=profile["validated_context"],
        configured_context=context,
        accounting="MemTotal-MemAvailable once; CUDA/RSS/KV are not added again",
        scope="diagnostic only; no scheduler, KV or batch changes",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calibrate-resources", type=Path)
    parser.add_argument("--validated-context", type=int, default=64890)
    parser.add_argument("--cache-bytes", type=int, default=0)
    parser.add_argument("--reserve-bytes", type=int, default=8 * GIB)
    parser.add_argument("--sampling-margin-bytes", type=int, default=4 * GIB)
    args = parser.parse_args()
    identity = checkpoint_identity(args.model)
    recipe = json.loads(args.recipe.read_text())
    if args.calibrate_resources:
        atomic_json(
            args.profile,
            calibrate(
                args.calibrate_resources, identity, recipe, args.validated_context
            ),
        )
        print(args.profile.read_text())
        return
    if min(args.cache_bytes, args.reserve_bytes, args.sampling_margin_bytes) < 0:
        parser.error("cache and reserve must be nonnegative")
    report = estimate(
        json.loads(args.profile.read_text()),
        recipe,
        identity,
        memory_info(),
        cache_bytes=args.cache_bytes,
        reserve_bytes=args.reserve_bytes,
        sampling_margin=args.sampling_margin_bytes,
    )
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        argv = recipe["engine_argv"]
        size = int(argv[argv.index("--tensor-parallel-size") + 1])
        rank = int(argv[argv.index("--node-rank") + 1])
        config = json.loads((args.model / "config.json").read_text())
        config = config.get("text_config", config)
        layers = config["engram_layer_ids"]
        if len(manifest["packs"]) != len(layers) or {
            p["layer"] for p in manifest["packs"]
        } != set(layers):
            raise ValueError("Preflight manifest does not cover all Engram layers")
        for record in manifest["packs"]:
            select_pack(
                args.manifest,
                args.model,
                record["layer"],
                size,
                rank,
                record["row_start"],
                record["rows"],
                config["engram_head_dim"],
                config["engram_head_dim"] // 32,
                "sample",
            )
        report["manifest_sample_validation"] = (
            "passed; runtime rechecks native TP layout"
        )
    print(json.dumps(report), flush=True)
    if report["status"].startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
