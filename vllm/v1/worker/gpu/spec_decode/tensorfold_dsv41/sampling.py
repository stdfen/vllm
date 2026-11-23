# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 TensorFold contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned DSV4.1 sampling.zig keyed exact sampling, independent of vLLM RNG."""

import math

MASK = (1 << 64) - 1


def mix(x: int) -> int:
    x &= MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def uniform(seed: int, position: int, token: int) -> float:
    x = mix(seed + 0x9E3779B97F4A7C15)
    x = mix(x ^ ((position * 0xD1B54A32D192ED03) & MASK))
    x = mix(x ^ token)
    return float(x >> 11) * 2**-53 + 2**-54


def pairwise_sum(values: list[float]) -> float:
    if len(values) < 8:
        total = 0.0
        for value in values:
            total += value
        return total
    r = values[:8].copy()
    whole = len(values) - len(values) % 8
    for i in range(8, whole, 8):
        for j in range(8):
            r[j] += values[i + j]
    result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    for v in values[whole:]:
        result += v
    return result


def choose(
    values: list[float],
    ids: list[int],
    *,
    seed: int,
    position: int,
    temperature: float,
    top_k: int = 20,
    top_p: float = 0.95,
    min_p: float = 0,
) -> int:
    """Values/ids must be ordered by descending logit, then ascending id."""
    if temperature <= 0:
        return ids[0]
    top_k = 20 if top_k <= 0 else top_k
    if top_k + 8 > 64:
        raise ValueError("TensorFold sampling supports at most 56 top-k candidates")
    k = max(1, min(top_k, len(values)))
    scaled = [float(v) / max(temperature, 1e-6) for v in values[:k]]
    scores = [
        v - math.log(-math.log(uniform(seed, position, token)))
        for v, token in zip(scaled, ids[:k], strict=True)
    ]
    if 0 < top_p < 1:
        mx = max(scaled)
        probs = [math.exp(v - mx) for v in scaled]
        total = pairwise_sum(probs)
        acc, below = 0.0, 0
        for p in probs:
            acc += p / total
            below += int(acc < top_p)
        scores[below + 1 :] = [-math.inf] * max(0, k - below - 1)
    if min_p > 0:
        floor = scaled[0] + math.log(min_p)
        scores = [v if scaled[i] >= floor else -math.inf for i, v in enumerate(scores)]
    return ids[max(range(k), key=scores.__getitem__)]
