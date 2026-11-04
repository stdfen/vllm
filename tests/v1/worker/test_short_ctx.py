# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu import short_ctx


def test_short_context_bounds_are_sorted_and_deduplicated():
    assert short_ctx._parse_bounds("1025, 512,512") == [512, 1025]


@pytest.mark.parametrize("value", ["0", "-1", "512,0"])
def test_short_context_bounds_must_be_positive(value):
    with pytest.raises(ValueError, match="must be positive"):
        short_ctx._parse_bounds(value)


def test_short_context_capture_and_dispatch_state_can_be_cleared():
    short_ctx.set_capture_bound(512)
    short_ctx.set_dispatch_max_seq_len(500)
    assert short_ctx.capture_bound() == 512
    assert short_ctx.dispatch_max_seq_len() == 500

    short_ctx.set_capture_bound(None)
    short_ctx.set_dispatch_max_seq_len(None)
    assert short_ctx.capture_bound() is None
    assert short_ctx.dispatch_max_seq_len() is None
