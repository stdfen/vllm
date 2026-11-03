#!/usr/bin/env python3
"""Validate one real routed expert through the v0.32 EXL3 MoE ABI."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--expert", type=int, default=10)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--graph", action="store_true")
    return parser.parse_args()


def read_tensor(model: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model / index[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


class OneExpert(torch.nn.Module):
    def __init__(self, global_expert: int) -> None:
        super().__init__()
        self.global_expert = global_expert
        self.local_num_experts = 1
        self.expert_map = None

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return 0 if expert_id == self.global_expert else -1


def main() -> None:
    args = parse_args()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    root = f"layers.{args.layer}.ffn.experts.{args.expert}"
    tensor_bits = {
        f"{root}.{projection}": read_tensor(
            args.model, index, f"{root}.{projection}.trellis"
        ).shape[-1]
        // 16
        for projection in ("w1", "w2", "w3")
    }
    config = Exl3Config(tensor_bits=tensor_bits)
    moe_config = SimpleNamespace(
        tp_rank=args.tp_rank,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    method = Exl3MoEMethod(moe_config, config, f"model.layers.{args.layer}.ffn.experts")
    layer = OneExpert(args.expert)
    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=5120,
        intermediate_size_per_partition=1152,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *unused: None,
    )
    for projection, shard_id, stem in (
        ("w1", "w1", "w13"),
        ("w3", "w3", "w13"),
        ("w2", "w2", "w2"),
    ):
        for role in ("trellis", "suh", "svh", config.codebook):
            parameter = getattr(layer, f"{stem}_{role}")
            value = read_tensor(args.model, index, f"{root}.{projection}.{role}")
            assert parameter.weight_loader(
                parameter,
                value,
                f"experts.{stem}_{role}",
                shard_id,
                args.expert,
                True,
            )
    layer.cuda()
    method.process_weights_after_loading(layer)

    generator = torch.Generator(device="cuda").manual_seed(200 + args.tp_rank)
    x = torch.randn(
        (args.rows, 5120),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    topk_ids = torch.zeros((args.rows, 1), dtype=torch.long, device="cuda")
    topk_weights = torch.ones((args.rows, 1), dtype=torch.float16, device="cuda")
    actual = method.apply(layer, x, topk_weights, topk_ids, None, None).float()
    execution = layer._exl3_last_apply
    graph_replay_max_abs = None
    if args.graph:
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
        graph_replay_max_abs = (graph_output.float() - actual).abs().max().item()

    from exllamav3.modules.quant.exl3 import LinearEXL3

    marker13 = getattr(layer, f"w13_{config.codebook}")
    marker2 = getattr(layer, f"w2_{config.codebook}")

    def reference(stem: str, slot: int | None, in_size: int, out_size: int):
        indexer = (0, slot) if slot is not None else (0,)
        marker = marker13[indexer] if stem == "w13" else marker2[indexer]
        kwargs = {"mcg": None, "mul1": None}
        kwargs[config.codebook] = marker
        return LinearEXL3(
            None,
            in_size,
            out_size,
            suh=getattr(layer, f"{stem}_suh")[indexer],
            svh=getattr(layer, f"{stem}_svh")[indexer],
            trellis=getattr(layer, f"{stem}_trellis")[indexer],
            key=f"{root}.{stem}",
            **kwargs,
        )

    gate = reference("w13", 0, 5120, 1152)
    up = reference("w13", 1, 5120, 1152)
    down = reference("w2", None, 1152, 5120)
    intermediate = torch.nn.functional.silu(
        x.float() @ gate.get_weight_tensor().float()
    ) * (x.float() @ up.get_weight_tensor().float())
    expected = intermediate @ down.get_weight_tensor().float()
    metrics = parity_metrics(actual, expected)
    result = {
        "passed": metrics["passed"]
        and execution == "fused_decode"
        and (graph_replay_max_abs is None or graph_replay_max_abs == 0.0),
        "layer": args.layer,
        "expert": args.expert,
        "tp_rank": args.tp_rank,
        "bits": sorted(set(tensor_bits.values())),
        "codebook": config.codebook,
        "shape": list(actual.shape),
        "execution": execution,
        "graph_replay_max_abs": graph_replay_max_abs,
        "metrics": metrics,
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
