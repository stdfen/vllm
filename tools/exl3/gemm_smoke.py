#!/usr/bin/env python3
"""Validate an EXL3 CUDA linear kernel against reconstructed real weights."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", default="layers.0.ffn.experts.10.w1")
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--vllm-runtime",
        type=Path,
        default=Path("vllm/model_executor/layers/quantization/exl3/runtime.py"),
    )
    return parser.parse_args()


def parity_metrics(output: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (output - expected).abs()
    expected_rms = expected.square().mean().sqrt().item()
    expected_peak = expected.abs().max().item()
    max_abs = difference.max().item()
    normalized_rmse = difference.square().mean().sqrt().item() / max(
        expected_rms, 1e-12
    )
    relative_peak_error = max_abs / max(expected_peak, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        output.flatten(), expected.flatten(), dim=0
    ).item()
    elementwise = torch.allclose(output, expected, rtol=2e-2, atol=5e-2)
    # Decode uses the row-1 GEMV kernel. Its near-zero outputs can fail a
    # pointwise relative check even when aggregate FP16 error remains bounded,
    # so require all three scale-aware limits as the alternate acceptance path.
    scaled = (
        cosine >= 0.9999 and normalized_rmse <= 1e-2 and relative_peak_error <= 1e-2
    )
    return {
        "passed": elementwise or scaled,
        "elementwise_allclose": elementwise,
        "max_abs": max_abs,
        "max_rel": (difference / expected.abs().clamp_min(1e-4)).max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "normalized_rmse": normalized_rmse,
        "relative_peak_error": relative_peak_error,
        "error_p99": torch.quantile(difference, 0.99).item(),
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EXL3 GEMM smoke test")

    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensor_names = {
        suffix: f"{args.component}.{suffix}"
        for suffix in ("trellis", "suh", "svh", "mcg", "mul1")
    }
    shard_names = {index[name] for name in tensor_names.values() if name in index}
    if len(shard_names) != 1:
        raise RuntimeError(
            f"component tensors must share one shard, got {sorted(shard_names)}"
        )

    shard = args.model / shard_names.pop()
    tensors: dict[str, torch.Tensor | None] = {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for suffix, name in tensor_names.items():
            tensors[suffix] = handle.get_tensor(name).cuda() if name in index else None

    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    assert trellis is not None and suh is not None and svh is not None
    in_features = suh.numel()
    out_features = svh.numel()

    # Import after the extension is on PYTHONPATH. This is the upstream reference
    # wrapper around the same packed tensors used by the vLLM integration.
    from exllamav3.modules.quant.exl3 import LinearEXL3

    linear = LinearEXL3(
        None,
        in_features,
        out_features,
        suh=suh,
        svh=svh,
        trellis=trellis,
        mcg=tensors["mcg"],
        mul1=tensors["mul1"],
        key=args.component,
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        (args.rows, in_features),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    actual = linear.forward(x, {"reconstruct": False}, torch.float32)
    weight = linear.get_weight_tensor()
    expected = x.float() @ weight.float()
    spec = importlib.util.spec_from_file_location(
        "vllm_exl3_runtime_smoke", args.vllm_runtime
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vLLM EXL3 runtime: {args.vllm_runtime}")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    marker = tensors["mcg"] if tensors["mcg"] is not None else tensors["mul1"]
    assert marker is not None
    backend = runtime.create_exl3_linear_backend(
        trellis=trellis,
        suh=suh,
        svh=svh,
        marker=marker,
        bits=int(trellis.shape[-1] // 16),
        codebook="mcg" if tensors["mcg"] is not None else "mul1",
    )
    vllm_actual = backend(x).float()
    graph_replay_max_abs = None
    if args.graph:
        static_x = x.clone()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = backend(static_x)
        graph.replay()
        torch.cuda.synchronize()
        graph_replay_max_abs = (graph_output.float() - vllm_actual).abs().max().item()
    torch.cuda.synchronize()

    reference_metrics = parity_metrics(actual, expected)
    vllm_metrics = parity_metrics(vllm_actual, expected)
    expected_peak = expected.abs().max().item()
    graph_passed = graph_replay_max_abs is None or graph_replay_max_abs == 0.0
    result = {
        "passed": reference_metrics["passed"]
        and vllm_metrics["passed"]
        and graph_passed,
        "component": args.component,
        "shape": [args.rows, in_features, out_features],
        "bits": int(trellis.shape[-1] // 16),
        "codebook": "mcg" if tensors["mcg"] is not None else "mul1",
        "expected_peak": expected_peak,
        "reference": reference_metrics,
        "vllm": vllm_metrics,
        "graph_replay_max_abs": graph_replay_max_abs,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
