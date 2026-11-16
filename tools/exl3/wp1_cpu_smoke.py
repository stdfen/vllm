# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline CPU smoke checks for the v0.33 EXL3 migration.

This intentionally avoids pytest so it can run in the historical GB10 runtime
image without downloading test-only dependencies.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
)
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.quantization.exl3.checkpoint import Exl3Checkpoint
from vllm.model_executor.layers.quantization.exl3.linear import Exl3LinearMethod
from vllm.model_executor.layers.quantization.exl3.moe import Exl3MoEMethod


class _FakeExperts(torch.nn.Module):
    def __init__(self, tp_rank: int) -> None:
        super().__init__()
        self.local_num_experts = 2
        self.expert_map = None
        self.tp_rank = tp_rank

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id if expert_id < self.local_num_experts else -1


class _FakeLinear(torch.nn.Module):
    tp_rank = 0


def _write_synthetic_checkpoint(root: Path) -> None:
    tensors = {
        "layers.0.mlp.down_proj.trellis": [4, 4, 48],
        "layers.0.mlp.down_proj.suh": [64],
        "layers.0.mlp.down_proj.svh": [64],
        "layers.0.mlp.down_proj.mul1": [],
        "embed.weight": [128, 64],
    }
    header = {
        name: {"dtype": "I16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    encoded = json.dumps(header).encode()
    shard = "model-00001-of-00001.safetensors"
    (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )
    (root / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "codebook": "mul1",
                "tensor_storage": {
                    "layers.0.mlp.down_proj": {"bits_per_weight": 3}
                },
            }
        ),
        encoding="utf-8",
    )


def _make_moe_method(tp_rank: int) -> Exl3MoEMethod:
    moe = SimpleNamespace(
        tp_rank=tp_rank,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    tensor_bits = {
        f"layers.3.ffn.experts.{expert}.{projection}": 3
        for expert in range(2)
        for projection in ("w1", "w2", "w3")
    }
    return Exl3MoEMethod(
        moe,
        Exl3Config(tensor_bits=tensor_bits),
        "model.layers.3.ffn.experts",
    )


def run_smoke(real_model: Path | None) -> None:
    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is Exl3Config
    assert Exl3Config.override_quantization_method(
        {"quant_method": "exl3"}, None
    ) == "exl3"
    assert Exl3Config.override_quantization_method({"quant_method": "fp8"}, None) is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_synthetic_checkpoint(root)
        checkpoint = Exl3Checkpoint.parse(root)
        packed = checkpoint.specs["layers.0.mlp.down_proj"]
        assert (packed.role, packed.bits, packed.codebook) == ("packed", 3, "mul1")

    linear = Exl3LinearMethod(
        Exl3Config(tensor_bits={"layers.0.mlp.down_proj": 3}),
        "model.layers.0.mlp.down_proj",
    )
    layer = _FakeLinear()
    linear.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *unused: None,
    )
    assert layer.trellis.shape == (4, 4, 48)
    assert layer.suh.shape == (64,)
    assert layer.svh.shape == (64,)

    for tp_rank in (0, 1):
        method = _make_moe_method(tp_rank)
        experts = _FakeExperts(tp_rank)
        method.create_weights(
            experts,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *unused: None,
        )
        full_w1 = torch.arange(4 * 8 * 48, dtype=torch.int16).reshape(4, 8, 48)
        assert experts.w13_trellis.weight_loader(
            experts.w13_trellis,
            full_w1,
            "experts.w13_trellis",
            "w1",
            1,
            True,
        )
        torch.testing.assert_close(
            experts.w13_trellis[1, 0], full_w1.chunk(2, dim=1)[tp_rank]
        )

    if real_model is not None:
        checkpoint = Exl3Checkpoint.parse(real_model)
        packed = [item for item in checkpoint.specs.values() if item.role == "packed"]
        assert len(packed) == 47_900
        assert {item.codebook for item in packed} == {"mul1"}
        print(f"REAL_MODEL_PACKED={len(packed)}")
    print("WP1_CPU_SMOKE=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    run_smoke(args.model)


if __name__ == "__main__":
    main()
