# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
    _dsv41_cache_block_size,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_prefill_config
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend


def test_sm120_flashinfer_uses_supported_sparse_decode_page_size():
    assert DeepseekV4Attention.swa_cache_block_size == 32
    assert DeepseekV4FlashInferSM120Attention.swa_cache_block_size == 64


@pytest.mark.parametrize(
    "compress_ratio,expected_block_size", [(1, 64), (2, 128)]
)
def test_attention_spec_preserves_64_states_per_page(
    compress_ratio, expected_block_size
):
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    layer = SimpleNamespace(
        is_kv_source=True,
        compress_ratio=compress_ratio,
        head_dim=512,
        kv_cache_dtype="fp8_ds_mla",
        kv_cache_torch_dtype=torch.uint8,
        kv_page_alignment=512,
        compressed_bytes_per_token=528,
    )

    with mock.patch(
        "vllm.models.deepseek_v41.attention.current_platform"
    ) as platform:
        platform.is_device_capability_family.return_value = False
        spec = DeepseekV4Attention.get_kv_cache_spec(layer, config)

    assert spec is not None
    assert spec.block_size == expected_block_size
    assert spec.tokens_per_state == compress_ratio
    assert spec.num_states == 64


@pytest.mark.parametrize(
    "compress_ratio,expected_block_size", [(1, 64), (2, 128)]
)
def test_indexer_spec_preserves_64_states_per_page(
    compress_ratio, expected_block_size
):
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8"))
    cache = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64),
        compress_ratio=compress_ratio,
        head_dim=132,
        dtype=torch.uint8,
        sparse_logits=False,
    )

    with mock.patch(
        "vllm.models.deepseek_v41.attention.current_platform"
    ) as platform:
        platform.is_device_capability_family.return_value = False
        spec = DeepseekV4IndexerCache.get_kv_cache_spec(cache, config)

    assert spec.block_size == expected_block_size
    assert spec.tokens_per_state == compress_ratio
    assert spec.num_states == 64


def test_sm12x_rejects_non_64_state_cache_geometry():
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32))
    layer = SimpleNamespace(
        is_kv_source=True,
        compress_ratio=2,
        head_dim=512,
        kv_cache_dtype="fp8_ds_mla",
        kv_cache_torch_dtype=torch.uint8,
        kv_page_alignment=512,
        compressed_bytes_per_token=528,
    )

    with (
        mock.patch(
            "vllm.models.deepseek_v41.attention.current_platform"
        ) as platform,
        pytest.raises(ValueError, match="requires 64 states per page"),
    ):
        platform.is_device_capability_family.side_effect = lambda family: (
            family == 120
        )
        DeepseekV4Attention.get_kv_cache_spec(layer, config)


@pytest.mark.parametrize(
    "extra_page_block_size,expected", [(64, True), (32, False)]
)
def test_sm120_prefill_capability_rejects_wrong_extra_page_geometry(
    extra_page_block_size, expected
):
    with mock.patch(
        "vllm.utils.flashinfer.has_flashinfer_sparse_mla_sm120_config",
        return_value=True,
    ):
        has_flashinfer_sparse_mla_sm120_prefill_config.cache_clear()
        assert (
            has_flashinfer_sparse_mla_sm120_prefill_config(
                32, 128, 64, 512, extra_page_block_size
            )
            is expected
        )
        has_flashinfer_sparse_mla_sm120_prefill_config.cache_clear()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 contract"
)
def test_dsv41_indexer_uses_deep_gemm_compatible_kernel_page_size():
    # These assertions run on the ARM64 SM120 validation host.
    assert _dsv41_cache_block_size(64, 1) == 64
    assert _dsv41_cache_block_size(64, 2) == 128
    assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [64, 128]
    assert DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes() == [
        64,
        128,
    ]
