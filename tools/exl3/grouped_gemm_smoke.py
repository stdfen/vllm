#!/usr/bin/env python3
"""Exercise vLLM's grouped EXL3 loader and runtime with real wo_a slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gemm_smoke import parity_metrics
from safetensors import safe_open

from vllm.model_executor.layers.quantization.exl3.config import Exl3Config
from vllm.model_executor.layers.quantization.exl3.linear import Exl3LinearMethod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", default="layers.0.attn.wo_a")
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rows", type=int, default=7)
    return parser.parse_args()


def read_tensor(model: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model / index[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the grouped EXL3 smoke test")

    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    slice_prefixes = tuple(
        f"{args.component}.slice.{slice_id}" for slice_id in range(8)
    )
    bits = {
        prefix: read_tensor(args.model, index, f"{prefix}.trellis").shape[-1] // 16
        for prefix in slice_prefixes
    }
    config = Exl3Config(tensor_bits=bits)
    method = Exl3LinearMethod(config, f"model.{args.component}")
    layer = torch.nn.Module()
    layer.tp_rank = args.tp_rank
    method.create_weights(
        layer,
        4096,
        [4096],
        4096,
        8192,
        torch.bfloat16,
        weight_loader=lambda *unused: None,
    )

    owned = range(layer._exl3_group_start, layer._exl3_group_start + 4)
    for slice_id in owned:
        prefix = slice_prefixes[slice_id]
        for suffix in ("trellis", "suh", "svh", config.codebook):
            parameter = getattr(layer, suffix)
            loaded = read_tensor(args.model, index, f"{prefix}.{suffix}")
            assert parameter.weight_loader(parameter, loaded, slice_id)

    layer.cuda()
    method.process_weights_after_loading(layer)
    generator = torch.Generator(device="cuda").manual_seed(100 + args.tp_rank)
    x = torch.randn(
        (args.rows, 4, 4096),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    actual = method.apply(layer, x).float()

    from exllamav3.modules.quant.exl3 import LinearEXL3

    expected_groups = []
    for local_id, slice_id in enumerate(owned):
        prefix = slice_prefixes[slice_id]
        reference = LinearEXL3(
            None,
            4096,
            1024,
            suh=layer.suh[local_id],
            svh=layer.svh[local_id],
            trellis=layer.trellis[local_id],
            mcg=(layer.mcg[local_id] if config.codebook == "mcg" else None),
            mul1=(layer.mul1[local_id] if config.codebook == "mul1" else None),
            key=prefix,
        )
        expected_groups.append(
            x[:, local_id].float() @ reference.get_weight_tensor().float()
        )
    expected = torch.stack(expected_groups, dim=1)
    metrics = parity_metrics(actual, expected)
    result = {
        "passed": metrics["passed"],
        "component": args.component,
        "tp_rank": args.tp_rank,
        "global_slices": list(owned),
        "shape": list(actual.shape),
        "bits": sorted(set(bits.values())),
        "codebook": config.codebook,
        "metrics": metrics,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
