# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-policy contract: contiguous bounded drafts priced at actual total rows.

The CPU selector owns no model state; array tests reproduce cost/length mistakes
without constructing a GPU runner. Runner integration needs separate real weights.
"""

import itertools
import runpy
from pathlib import Path

import numpy as np
import pytest

# Load the dependency-free leaf without initializing vLLM/PyTorch on CPU hosts.
select_prefixes = runpy.run_path(
    str(
        Path(__file__).parents[3]
        / "vllm/v1/worker/gpu/spec_decode/dspark/prefix_policy.py"
    )
)["select_prefixes"]


def test_single_request_maximizes_expected_tokens_per_round():
    """Select by prefix survival, not by independent confidence or deepest draft."""
    q = np.array([[0.9, 0.6, 0.2]])
    costs = np.array([0, 10, 12, 14, 20])
    result = select_prefixes(
        q, np.array([3]), np.array([1]), np.array([True]), costs, 2, 3
    )
    assert result.depths.tolist() == [2]
    assert result.expected_tokens[0] == pytest.approx(2.44)


def test_equal_rates_keep_shallowest_and_zero_confidence_skips_drafts():
    """Zero-depth verification still executes the target's one mandatory row."""
    for q, costs in [([1.0, 1.0], [0, 1, 2, 3]), ([0.0, 0.9], [0, 1, 1, 1])]:
        result = select_prefixes(
            np.array([q]), np.array([2]), np.array([1]), np.array([True]),
            np.array(costs), 0, 2,
        )
        assert result.depths.tolist() == [0]


def test_single_request_matches_exhaustive_reference():
    """Every reachable confidence/depth combination matches direct enumeration."""
    for q in itertools.product([0.0, 0.3, 0.9, 1.0], repeat=3):
        costs = np.array([0, 10, 11, 14, 22])
        expected = [1 + sum(np.prod(q[:j]) for j in range(1, d + 1)) for d in range(4)]
        want = max(range(4), key=lambda d: expected[d] / (costs[d + 1] + 2))
        result = select_prefixes(
            np.array([q]), np.array([3]), np.array([1]), np.array([True]), costs, 2, 3
        )
        assert result.depths.tolist() == [want]


def test_global_limit_preserves_high_value_prefix_and_prefill():
    """Low-confidence tails lose capacity without admitting holes or prefill drafts."""
    result = select_prefixes(
        np.array([[0.9, 0.8], [0.3, 0.2], [1.0, 1.0]]),
        np.array([2, 2, 0]), np.array([1, 1, 8]), np.array([True, True, False]),
        np.ones(15), 1, 2,
    )
    assert result.depths.tolist() == [2, 0, 0]
    assert result.expected_tokens[2] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_confidence_fails_closed(bad):
    with pytest.raises(ValueError, match="confidence probabilities"):
        select_prefixes(
            np.array([[bad]]), np.array([1]), np.array([1]), np.array([True]),
            np.array([0, 1, 2]), 0, 1,
        )


@pytest.mark.parametrize("costs", [[0, 1], [0, 2, 1], [0, 1, np.nan]])
def test_missing_or_invalid_costs_are_never_extrapolated_silently(costs):
    with pytest.raises(ValueError, match="cost"):
        select_prefixes(
            np.array([[0.9]]), np.array([1]), np.array([1]), np.array([True]),
            np.array(costs), 0, 1,
        )
