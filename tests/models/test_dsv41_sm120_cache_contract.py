# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    _dsv41_cache_block_size,
)
from vllm.platforms import current_platform
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend


def test_sm120_flashinfer_uses_supported_sparse_decode_page_size():
    assert DeepseekV4Attention.swa_cache_block_size == 32
    assert DeepseekV4FlashInferSM120Attention.swa_cache_block_size == 64


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
