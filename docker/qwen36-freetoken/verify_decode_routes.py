# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interpret and compile the route sanitizer without allocating a GPU."""

import argparse
import ast
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def load_sanitizer(source, directory):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "_prepare_decode_routes"
    )
    module_path = directory / "route_sanitizer.py"
    module_path.write_text(
        "import triton\nfrom triton import language as tl\n"
        + ast.unparse(kernel)
        + "\n"
    )
    spec = importlib.util.spec_from_file_location("route_sanitizer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._prepare_decode_routes


def interpret(kernel):
    inputs = (
        ([0, 3, 17, 40, 60, 70, 111, 255], [0, 3, 17, 40, 60, 70, 111, 255]),
        ([-1, 17, 255, -1, 128, 256, 9, -1], [9, 17, 255, 9, 128, 9, 9, 9]),
        ([-1] * 8, [0] * 8),
    )
    for input_ids, expected_ids in inputs:
        ids = torch.tensor([input_ids], dtype=torch.int32)
        valid = (ids >= 0) & (ids < 256)
        weights = torch.arange(1, 9, dtype=torch.float32).view(1, 8) / 16
        weights[~valid] = torch.nan
        safe_ids = torch.empty_like(ids)
        safe_weights = torch.empty_like(weights)
        no_valid_route = torch.empty(1, dtype=torch.bool)
        kernel[(1,)](ids, weights, safe_ids, safe_weights, no_valid_route, 8, 256)
        torch.testing.assert_close(
            safe_ids, torch.tensor([expected_ids], dtype=torch.int32)
        )
        torch.testing.assert_close(safe_weights, weights.masked_fill(~valid, 0))
        assert no_valid_route.item() == (not valid.any().item())
        assert ids.tolist() == [input_ids]
        print(f"Route sanitizer interpreter passed {input_ids}")


def compile_sm89(kernel):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    signature = {
        "ids_ptr": "*i32",
        "weights_ptr": "*fp32",
        "safe_ids_ptr": "*i32",
        "safe_weights_ptr": "*fp32",
        "no_valid_route_ptr": "*i1",
        "top_k": "constexpr",
        "num_experts": "constexpr",
    }
    source = ASTSource(kernel, signature, constexprs={"top_k": 8, "num_experts": 256})
    result = compile(source, target=GPUTarget("cuda", 89, 32))
    assert result.asm["cubin"]
    print("Route sanitizer compiled for SM89")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("interpret", "compile"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        kernel = load_sanitizer(args.source, Path(directory))
        if args.mode == "interpret":
            interpret(kernel)
        else:
            compile_sm89(kernel)


if __name__ == "__main__":
    main()
