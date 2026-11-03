# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3.config import Exl3Config
from vllm.model_executor.layers.quantization.exl3.moe import Exl3MoEMethod


class FakeExperts(torch.nn.Module):
    def __init__(self, tp_rank: int) -> None:
        super().__init__()
        self.local_num_experts = 2
        self.expert_map = None
        self.tp_rank = tp_rank

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id if expert_id < self.local_num_experts else -1


def make_method(tp_rank: int) -> Exl3MoEMethod:
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


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_exl3_moe_creates_and_loads_tp_packed_parameters(tp_rank: int) -> None:
    method = make_method(tp_rank)
    layer = FakeExperts(tp_rank)
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *unused: None,
    )

    assert layer.w13_trellis.shape == (2, 2, 4, 4, 48)
    assert layer.w2_trellis.shape == (2, 4, 4, 48)
    full_w1 = torch.arange(4 * 8 * 48, dtype=torch.int16).reshape(4, 8, 48)
    loaded = layer.w13_trellis.weight_loader(
        layer.w13_trellis,
        full_w1,
        "experts.w13_trellis",
        "w1",
        1,
        True,
    )
    assert loaded
    torch.testing.assert_close(
        layer.w13_trellis[1, 0], full_w1.chunk(2, dim=1)[tp_rank]
    )

    full_w2 = torch.arange(8 * 4 * 48, dtype=torch.int16).reshape(8, 4, 48)
    assert layer.w2_trellis.weight_loader(
        layer.w2_trellis,
        full_w2,
        "experts.w2_trellis",
        "w2",
        0,
        True,
    )
    torch.testing.assert_close(layer.w2_trellis[0], full_w2.chunk(2, dim=0)[tp_rank])


def test_exl3_moe_reference_routing_accumulates_topk_weights() -> None:
    method = make_method(0)
    layer = FakeExperts(0)
    layer._exl3_expert_backends = (
        (lambda x: x + 1, lambda x: x + 2, lambda x: x[:, :4]),
        (lambda x: x + 3, lambda x: x + 4, lambda x: x[:, :4] * 2),
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    topk_weights = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])

    result = method.apply(layer, x, topk_weights, topk_ids, None, None)

    expert0 = torch.nn.functional.silu((x + 1).clamp(max=10)) * (x + 2).clamp(
        min=-10, max=10
    )
    expert1 = (
        torch.nn.functional.silu((x + 3).clamp(max=10))
        * (x + 4).clamp(min=-10, max=10)
        * 2
    )
    expert_outputs = torch.stack((expert0, expert1), dim=1)
    expected = (
        expert_outputs.gather(
            1, topk_ids[..., None].expand(-1, -1, expert_outputs.shape[-1])
        )
        * topk_weights[..., None]
    ).sum(dim=1)
    torch.testing.assert_close(result, expected)
