# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import torch


def test_text_mapper_maps_exl3_lm_head_components() -> None:
    from vllm.models.deepseek_v41.nvidia.model import (
        _make_deepseek_v4_weights_mapper,
    )

    mapper = _make_deepseek_v4_weights_mapper("fp4", "weight_scale")
    for component in ("trellis", "suh", "svh", "mul1", "mcg"):
        assert mapper._map_name(f"head.{component}") == f"lm_head.{component}"


def test_vl_mapper_maps_exl3_lm_head_components() -> None:
    from vllm.models.deepseek_v41.nvidia.vl_model import (
        _make_deepseek_v4_vl_weights_mapper,
    )

    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale")
    for component in ("trellis", "suh", "svh", "mul1", "mcg"):
        assert mapper._map_name(f"head.{component}") == (
            f"language_model.lm_head.{component}"
        )


def test_o_proj_dispatches_grouped_exl3_without_dense_weight(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia.ops import o_proj

    batch, groups, input_size, output_size = 3, 2, 4, 2
    grouped_input = torch.arange(
        batch * groups * input_size, dtype=torch.float32
    ).reshape(batch, groups, input_size)

    monkeypatch.setattr(
        o_proj,
        "fused_inv_rope_fp8_quant",
        lambda *args, **kwargs: (grouped_input, None),
    )

    class FakeQuantMethod:
        def __init__(self) -> None:
            self.calls = 0

        def apply(self, layer, value):
            assert layer is wo_a
            self.calls += 1
            return value[..., :output_size] + 7

    wo_a = torch.nn.Module()
    wo_a.register_parameter(
        "trellis", torch.nn.Parameter(torch.empty(0), requires_grad=False)
    )
    wo_a.quant_method = FakeQuantMethod()

    observed: dict[str, torch.Tensor] = {}

    def wo_b(value: torch.Tensor) -> torch.Tensor:
        observed["value"] = value
        return value

    result = o_proj.deep_gemm_fp8_o_proj(
        torch.empty(batch, groups, input_size),
        torch.arange(batch),
        torch.empty(1),
        wo_a,
        wo_b,
        n_groups=groups,
        heads_per_group=1,
        nope_dim=input_size,
        rope_dim=0,
        o_lora_rank=output_size,
        einsum_recipe=(1, 1, 128),
        tma_aligned_scales=False,
    )

    assert wo_a.quant_method.calls == 1
    expected = (grouped_input[..., :output_size] + 7).to(torch.bfloat16)
    torch.testing.assert_close(observed["value"], expected.flatten(1))
    torch.testing.assert_close(result, expected.flatten(1))


def test_process_weights_registers_exl3_l2_prefetch(monkeypatch) -> None:
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV41LLMForCausalLM

    calls = []
    plugin = ModuleType("vllm_exl3_moe")
    l2pf = ModuleType("vllm_exl3_moe.l2pf")
    l2pf.register_decoder_layers = lambda model: calls.append(("l2pf", model))
    monkeypatch.setitem(sys.modules, "vllm_exl3_moe", plugin)
    monkeypatch.setitem(sys.modules, "vllm_exl3_moe.l2pf", l2pf)

    model = SimpleNamespace(
        finalize_mega_moe_weights=lambda: calls.append(("mega", None)),
        finalize_mhc_broadcast_weights=lambda: calls.append(("mhc", None)),
    )
    instance = SimpleNamespace(model=model)

    DeepseekV41LLMForCausalLM.process_weights_after_loading(instance)

    assert calls == [("mega", None), ("mhc", None), ("l2pf", model)]
