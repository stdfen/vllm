# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.weight_staging import copy_loaded_weight


def test_copy_loaded_weight_preserves_cpu_copy_behavior() -> None:
    source = torch.arange(64, dtype=torch.float32)
    destination = torch.empty_like(source)

    copy_loaded_weight(destination, source)

    torch.testing.assert_close(destination, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int16])
def test_copy_loaded_weight_stages_large_cuda_copy(monkeypatch, dtype) -> None:
    free, total = torch.cuda.mem_get_info()
    if free < total * 0.05:
        pytest.skip("CUDA memory is occupied by another workload")
    monkeypatch.setenv("VLLM_PINNED_WEIGHT_STAGING", "on")
    monkeypatch.setenv("VLLM_PINNED_WEIGHT_STAGING_MIN_MIB", "1")
    monkeypatch.setenv("VLLM_PINNED_WEIGHT_STAGING_MIB", "1")
    source = torch.arange(1024 * 1024, dtype=dtype)
    destination = torch.empty_like(source, device="cuda")

    copy_loaded_weight(destination, source)

    torch.testing.assert_close(destination.cpu(), source)
