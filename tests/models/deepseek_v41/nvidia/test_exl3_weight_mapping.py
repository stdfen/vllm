import pytest

from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.models.deepseek_v41.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper,
)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("piece", ["weight", "trellis", "suh", "svh", "mul1", "mcg"])
def test_exl3_head_pieces_follow_output_head(expert_dtype, vision, piece):
    factory = (
        _make_deepseek_v4_vl_weights_mapper
        if vision
        else _make_deepseek_v4_weights_mapper
    )
    mapper = factory(expert_dtype, "weight_scale")
    prefix = "language_model." if vision else ""
    assert mapper.map_name(f"head.{piece}") == f"{prefix}lm_head.{piece}"


@pytest.mark.parametrize("vision", [False, True])
def test_exl3_shared_expert_pieces_preserve_projection_mapping(vision):
    factory = (
        _make_deepseek_v4_vl_weights_mapper
        if vision
        else _make_deepseek_v4_weights_mapper
    )
    mapper = factory("fp4", "weight_scale")
    prefix = "language_model.model." if vision else "model."
    assert mapper.map_name("layers.3.ffn.shared_experts.w2.trellis") == (
        f"{prefix}layers.3.ffn.shared_experts.down_proj.trellis"
    )
    assert mapper.map_name("layers.3.attn.wq_b.trellis") == (
        f"{prefix}layers.3.attn.wq_b.trellis"
    )
