#!/usr/bin/env python3
"""Load one complete 384-expert EXL3 layer and compare fused routing to oracle."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from gemm_smoke import parity_metrics
from safetensors import safe_open

from vllm.model_executor.layers.quantization.exl3.config import Exl3Config
from vllm.model_executor.layers.quantization.exl3.moe import Exl3MoEMethod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--experts", type=int, default=384)
    return parser.parse_args()


class AllExperts(torch.nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.local_num_experts = experts
        self.expert_map = None

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id if 0 <= expert_id < self.local_num_experts else -1


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    root = f"layers.{args.layer}.ffn.experts"
    component_names = [
        f"{root}.{expert}.{projection}"
        for expert in range(args.experts)
        for projection in ("w1", "w2", "w3")
    ]
    tensor_bits: dict[str, int] = {}
    trellis_by_shard: dict[str, list[str]] = defaultdict(list)
    for component in component_names:
        name = f"{component}.trellis"
        trellis_by_shard[index[name]].append(name)
    for shard_name, names in trellis_by_shard.items():
        with safe_open(args.model / shard_name, framework="pt", device="cpu") as handle:
            for name in names:
                tensor_bits[name.removesuffix(".trellis")] = (
                    handle.get_slice(name).get_shape()[-1] // 16
                )
    config = Exl3Config(tensor_bits=tensor_bits)
    moe_config = SimpleNamespace(
        tp_rank=args.tp_rank,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    method = Exl3MoEMethod(moe_config, config, f"model.layers.{args.layer}.ffn.experts")
    layer = AllExperts(args.experts)
    method.create_weights(
        layer,
        num_experts=args.experts,
        hidden_size=5120,
        intermediate_size_per_partition=1152,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *unused: None,
    )

    loads_by_shard: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    for expert in range(args.experts):
        for projection, shard_id, stem in (
            ("w1", "w1", "w13"),
            ("w3", "w3", "w13"),
            ("w2", "w2", "w2"),
        ):
            for role in ("trellis", "suh", "svh", config.codebook):
                checkpoint_name = f"{root}.{expert}.{projection}.{role}"
                loads_by_shard[index[checkpoint_name]].append(
                    (checkpoint_name, f"{stem}_{role}", shard_id, expert)
                )
    loaded_count = 0
    for shard_name, loads in sorted(loads_by_shard.items()):
        with safe_open(args.model / shard_name, framework="pt", device="cpu") as handle:
            for checkpoint_name, parameter_name, shard_id, expert in loads:
                parameter = getattr(layer, parameter_name)
                success = parameter.weight_loader(
                    parameter,
                    handle.get_tensor(checkpoint_name),
                    f"experts.{parameter_name}",
                    shard_id,
                    expert,
                    True,
                )
                if not success:
                    raise RuntimeError(f"loader rejected {checkpoint_name}")
                loaded_count += 1
    expected_loads = args.experts * 3 * 4
    if loaded_count != expected_loads:
        raise RuntimeError(f"loaded {loaded_count}, expected {expected_loads}")

    load_seconds = time.perf_counter() - started
    layer.cuda()
    method.process_weights_after_loading(layer)
    ready_seconds = time.perf_counter() - started
    generator = torch.Generator(device="cuda").manual_seed(300 + args.tp_rank)
    x = torch.randn(
        (args.rows, 5120),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    topk_ids = torch.randint(
        0,
        args.experts,
        (args.rows, 8),
        device="cuda",
        generator=generator,
    )
    topk_weights = torch.rand(
        (args.rows, 8),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    fused_state = layer._exl3_fused_decode
    fused = method.apply(layer, x, topk_weights, topk_ids, None, None).float()
    fused_execution = layer._exl3_last_apply
    layer._exl3_fused_decode = None
    oracle = method.apply(layer, x, topk_weights, topk_ids, None, None).float()
    oracle_execution = layer._exl3_last_apply
    layer._exl3_fused_decode = fused_state
    metrics = parity_metrics(fused, oracle)

    static_x = x.clone()
    static_weights = topk_weights.clone()
    static_ids = topk_ids.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = method.apply(
            layer, static_x, static_weights, static_ids, None, None
        )
    graph.replay()
    torch.cuda.synchronize()
    graph_replay_max_abs = (graph_output.float() - fused).abs().max().item()
    markers_ok = bool(
        torch.all(layer.w13_mul1 == -2082680531)
        and torch.all(layer.w2_mul1 == -2082680531)
    )
    expected_execution = "fused_decode" if args.rows <= 128 else "fused_prefill_tiles"
    passed = (
        metrics["passed"]
        and markers_ok
        and fused_execution == expected_execution
        and oracle_execution == "reference_loop"
        # Multi-expert fused routing accumulates into FP32 with atomics. CUDA
        # Graph replay may change accumulation order, so accept only a tightly
        # bounded FP16-scale delta rather than requiring bitwise identity.
        and graph_replay_max_abs <= 5e-4
    )
    result = {
        "passed": passed,
        "layer": args.layer,
        "experts": args.experts,
        "tp_rank": args.tp_rank,
        "loaded_tensors": loaded_count,
        "expected_tensors": expected_loads,
        "source_shards": len(loads_by_shard),
        "bits": sorted(set(tensor_bits.values())),
        "codebook": config.codebook,
        "markers_ok": markers_ok,
        "fused_execution": fused_execution,
        "oracle_execution": oracle_execution,
        "graph_replay_max_abs": graph_replay_max_abs,
        "metrics": metrics,
        "load_seconds": load_seconds,
        "ready_seconds": ready_seconds,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
