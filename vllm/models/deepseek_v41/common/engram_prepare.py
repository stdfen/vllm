# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline rank-local packs for the native vLLM Engram layout."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm.models.deepseek_v41.common.engram import (
    EngramLayout,
    build_compressed_token_map,
)
from vllm.models.deepseek_v41.common.engram_pack import (
    HEADER_BYTES,
    atomic_json,
    checkpoint_identity,
    tensor_source,
    validate_record,
    write_pack,
)
from vllm.transformers_utils.config import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.tp_rank < args.tp_size:
        parser.error("rank must belong to TP group")
    config = get_config(str(args.model), trust_remote_code=False).get_text_config()
    layout = EngramLayout(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    token_map, vocab = build_compressed_token_map(tokenizer)
    if vocab != layout.compressed_vocab_size:
        raise ValueError("Tokenizer compressed vocabulary does not match Engram layout")
    identity = checkpoint_identity(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    records = []
    for index, layer in enumerate(layout.layer_ids):
        heads = [size for order in layout.primes[index] for size in order]
        width = (len(heads) + args.tp_size - 1) // args.tp_size
        start = sum(heads[: width * args.tp_rank])
        rows = sum(heads[width * args.tp_rank : width * (args.tp_rank + 1)])
        sources = {
            suffix: tensor_source(args.model, f"layers.{layer}.engram.embed.{suffix}")
            for suffix in ("weight", "scale")
        }
        dim, scale_dim = layout.head_dim, layout.head_dim // 32
        for suffix, cols in (("weight", dim), ("scale", scale_dim)):
            source = sources[suffix]
            if source["shape"] != [layout.num_embeddings[index], cols] or (
                source["bytes"] != layout.num_embeddings[index] * cols
            ):
                raise ValueError(f"Unsupported Engram {suffix} tensor geometry")
        if sources["weight"]["dtype"] != "F8_E4M3" or sources["scale"]["dtype"] not in (
            "U8",
            "F8_E8M0",
        ):
            raise ValueError("Expected raw FP8 weight and UE8M0 scale bytes")
        record = dict(
            version=1,
            header_bytes=HEADER_BYTES,
            layer=layer,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            row_start=start,
            rows=rows,
            global_rows=layout.num_embeddings[index],
            dim=dim,
            scale_dim=scale_dim,
            stride=512,
            file=f"engram-l{layer}-tp{args.tp_size}-r{args.tp_rank}.bin",
            sources=sources,
        )
        receipt = args.output / (record["file"] + ".json")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if any(previous.get(k) != v for k, v in record.items()):
                raise ValueError(
                    "Existing pack receipt differs; use another output directory"
                )
            validate_record(
                args.output,
                dict(schema=1, identity=identity),
                previous,
                verification="full",
            )
            result = previous
        else:
            if (args.output / record["file"]).exists():
                raise ValueError("Pack exists without receipt; inspect before retrying")
            print(json.dumps(dict(event="pack-start", **record)), flush=True)
            result = write_pack(
                args.model, args.output / record["file"], record, identity
            )
            atomic_json(receipt, result)
        records.append(result)
        print(
            json.dumps(
                dict(
                    event="pack-verified", file=result["file"], sha256=result["sha256"]
                )
            ),
            flush=True,
        )
    manifest = dict(
        schema=1,
        identity=identity,
        packs=records,
        compressed_vocab=vocab,
        token_count=len(token_map),
        runtime_verification="sample; full verification performed offline",
    )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing manifest differs")
    else:
        atomic_json(manifest_path, manifest)
    print(
        json.dumps(dict(event="manifest-published", path=str(manifest_path))),
        flush=True,
    )


if __name__ == "__main__":
    main()
