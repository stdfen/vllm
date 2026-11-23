# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference policy for DSpark compact prefix verification.

Uses the expected-prefix-tokens / round-time objective documented by the
Bertholomus TensorFold dsv41 implementation. It does not import its engine or
reuse its device layouts. Costs describe this vLLM deployment, not TensorFold.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrefixChoice:
    depths: np.ndarray
    expected_tokens: np.ndarray


def select_prefixes(
    confidence: np.ndarray,
    capacities: np.ndarray,
    non_draft_rows: np.ndarray,
    sampling_requests: np.ndarray,
    verify_ms: np.ndarray,
    draft_ms: float,
    max_drafts: int,
) -> PrefixChoice:
    """Choose each prefix against the other requests' scheduled windows.

    Confidence contains conditional acceptance probabilities, not logits or
    marginal acceptance rates. verify_ms is indexed by TOTAL live rows (zero
    is unused). Ties select the shallower prefix. A binding global limit removes
    the smallest marginal survival first, preserving contiguous prefixes.
    """
    q = np.asarray(confidence, dtype=np.float64)
    caps = np.asarray(capacities)
    base = np.asarray(non_draft_rows)
    sampling = np.asarray(sampling_requests)
    costs = np.asarray(verify_ms, dtype=np.float64)
    n = len(caps)
    if q.ndim != 2 or q.shape[0] != n:
        raise ValueError("confidence must have one row per request")
    if base.shape != (n,) or sampling.shape != (n,):
        raise ValueError("request metadata must match capacities")
    if not np.issubdtype(caps.dtype, np.integer) or not np.issubdtype(
        base.dtype, np.integer
    ):
        raise ValueError("row counts must be integers")
    if np.any(caps < 0) or np.any(caps > q.shape[1]) or np.any(base < 1):
        raise ValueError("invalid scheduled prefix capacities")
    if max_drafts < 0 or not np.isfinite(draft_ms) or draft_ms < 0:
        raise ValueError("invalid draft budget or pass cost")
    if not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("confidence probabilities must be finite in [0, 1]")
    caps = np.where(sampling, caps, 0)
    total = int(np.sum(base) + np.sum(caps))
    if costs.ndim != 1 or len(costs) <= total:
        raise ValueError("verify cost table does not cover scheduled rows")
    if not np.all(np.isfinite(costs[1:])) or np.any(costs[1:] <= 0):
        raise ValueError("verify costs must be finite and positive")
    if np.any(np.diff(costs[1:]) < 0):
        raise ValueError("verify costs must be non-decreasing")
    survival = np.cumprod(q, axis=1)
    expected = np.concatenate(
        (np.ones((n, 1)), 1 + np.cumsum(survival, axis=1)), axis=1
    )
    depths = np.zeros(n, dtype=np.int32)
    for i, cap in enumerate(caps):
        if not sampling[i] or cap == 0:
            continue
        other_rows = total - int(caps[i])
        denominator = costs[other_rows : other_rows + int(cap) + 1] + draft_ms
        depths[i] = np.argmax(expected[i, : int(cap) + 1] / denominator)
    # Unlike independent requests, a mixed batch also has a hard logit budget.
    while int(depths.sum()) > max_drafts:
        active = np.flatnonzero(depths)
        tail = survival[active, depths[active] - 1]
        i = active[np.argmin(tail)]
        depths[i] -= 1
    return PrefixChoice(depths, np.where(sampling, expected[np.arange(n), depths], 0))
