#!/usr/bin/env python3
"""Exercise ParallelEngramEmbedding with zero-sized file-backed parameters."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
from types import SimpleNamespace

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> None:
    args = parse_args()
    from vllm.models.deepseek_v41.common import engram

    engram.get_tensor_model_parallel_world_size = lambda: 2
    engram.get_tensor_model_parallel_rank = lambda: args.tp_rank
    engram.get_engram_dp_size = lambda: 1
    before = rss_bytes()
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    layout = engram.EngramLayout(SimpleNamespace(**config))
    layer_index = layout.layer_ids.index(args.layer)
    rows = layout.num_embeddings[layer_index]
    head_sizes = tuple(
        size for order in layout.primes[layer_index] for size in order
    )
    table = engram.ParallelEngramEmbedding(
        rows,
        layout.head_dim,
        head_sizes,
        cpu_offload=True,
        file_table_dir=str(args.model),
        file_tensor_prefix=f"layers.{args.layer}.engram.embed",
    )
    mapped = rss_bytes()
    offsets = torch.tensor([0, *torch.tensor(head_sizes).cumsum(0)[:-1]])
    indices = torch.zeros((4, len(head_sizes)), dtype=torch.long, device="cuda")
    for head in range(table.head_start, table.head_start + table.part_n_hash_cols):
        size = head_sizes[head]
        indices[:, head] = offsets[head] + torch.tensor(
            [0, 1, min(100, size - 1), size - 1], device="cuda"
        )
    output = torch.empty(
        (4, table.part_n_hash_cols, layout.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    table.lookup(indices, output)
    torch.cuda.synchronize()
    after = rss_bytes()
    direct = torch.empty_like(output)
    table._file_store.lookup(
        indices, table.head_start, table.part_n_hash_cols, direct
    )
    torch.testing.assert_close(output, direct, rtol=0, atol=0)
    passed = (
        table.weight.numel() == 0
        and table.weight_scale_inv.numel() == 0
        and after - before < 512 * 1024**2
    )
    result = {
        "passed": passed,
        "layer": args.layer,
        "tp_rank": args.tp_rank,
        "vocab_start": table.vocab_start_idx,
        "vocab_end": table.vocab_end_idx,
        "local_heads": table.part_n_hash_cols,
        "dummy_weight_numel": table.weight.numel(),
        "dummy_scale_numel": table.weight_scale_inv.numel(),
        "mapped_bytes": table._file_store.mapped_bytes,
        "rss_before": before,
        "rss_after_mapping": mapped,
        "rss_after_lookup": after,
        "rss_growth": after - before,
        "output_shape": list(output.shape),
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
