# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 TensorFold contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Port of pinned TensorFold core/lanes/{allocate,depth,proposer,accept}.zig.

Source artifact and ordered patches: benchmarks/dspark-tensorfold-full-path/
source-lock/manifest.json. No dependency on vLLM DSpark/adaptive/rejection code.
Keep strict comparisons, float64 arithmetic and insertion-order tie breaking.
"""

import math
from dataclasses import dataclass, field

ROUND_MS = (
    30.0,
    35.2,
    39.9,
    44.9,
    48.6,
    52.1,
    55.6,
    59.1,
    62.7,
    66.3,
    69.6,
    72.9,
    75.5,
    78.2,
    81.3,
    84.4,
    87.5,
    90.6,
    93.7,
    96.8,
    99.9,
    103.0,
    106.1,
    109.2,
)
PRIOR = (0.85, 0.75, 0.7, 0.65, 0.6)


def survivals(logits: list[float]) -> list[float]:
    result, reach = [], 1.0
    for x in logits:
        if not math.isfinite(x):
            raise ValueError("nonfinite TensorFold confidence")
        try:
            p = 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            # The original libm exp returns +inf for finite, very negative x.
            p = 0.0
        reach *= p
        result.append(reach)
    return result


def allocate(
    fixed: list[int],
    probabilities: list[list[float]],
    costs: dict[int, float],
    overhead_ms: float,
    max_rows: int,
) -> list[int]:
    """Exact greedy prefix frontier of allocate.zig, including zero depth."""
    if len(fixed) != len(probabilities) or any(x < 1 for x in fixed):
        raise ValueError("invalid fixed rows")
    if not math.isfinite(overhead_ms) or overhead_ms < 0:
        raise ValueError("invalid round overhead")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for q in probabilities for p in q):
        raise ValueError("invalid prefix probability")
    if any(not math.isfinite(x) or x <= 0 for x in costs.values()):
        raise ValueError("invalid TensorFold forward cost")
    counts = [0] * len(fixed)
    best_counts = counts.copy()
    rows = sum(fixed)
    expected = float(rows)

    def rate() -> float:
        cost = costs.get(rows) if costs else 1.0
        return -1.0 if cost is None else expected / (cost + overhead_ms)

    best = rate()
    while rows < max_rows:
        pick = None
        pick_p = 0.0
        for s, p in enumerate(probabilities):
            if counts[s] >= len(p):
                continue
            value = p[counts[s]]
            if pick is None or value > pick_p:
                pick, pick_p = s, value
        if pick is None:
            break
        counts[pick] += 1
        rows += 1
        expected += pick_p
        now = rate()
        if now > best:
            best, best_counts = now, counts.copy()
    return best_counts


def accept_chain(drafts: list[int], target_picks: list[int]) -> tuple[list[int], int]:
    """accept.zig + engine.conclude: matched prefix and one target correction."""
    if len(target_picks) != len(drafts) + 1:
        raise ValueError("one target pick is required per verification row")
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == target_picks[accepted]:
        accepted += 1
    return drafts[:accepted] + [target_picks[accepted]], accepted


@dataclass
class SuffixLookup:
    """proposer.zig defaults; index persists only for the owning request."""

    ngram: int = 3
    min_match: int = 6
    max_extension: int = 64
    silence_rounds: int = 16
    window: int = 4
    confident_match: int = 24
    index: dict[tuple[int, ...], list[int]] = field(default_factory=dict)
    indexed: int = 0
    last_key: tuple[int, ...] = ()
    recent: list[int] = field(default_factory=list)
    silent_for: int = 0
    last_match: int = 0
    last_confident: bool = False

    def propose(self, context: list[int], max_draft: int) -> list[int]:
        n = self.ngram
        if max_draft <= 0 or len(context) < n + 1:
            return []
        if self.silent_for > 0:
            self.silent_for -= 1
            return []
        changed = self.indexed > len(context) or (
            self.indexed != 0
            and tuple(context[self.indexed - n : self.indexed]) != self.last_key
        )
        if changed:
            self.index.clear()
            self.indexed = 0
        for position in range(max(self.indexed, n - 1), len(context)):
            key = tuple(context[position + 1 - n : position + 1])
            self.index.setdefault(key, []).append(position)
        self.indexed = len(context)
        self.last_key = tuple(context[-n:])
        positions = self.index.get(self.last_key, [])
        if not positions:
            return []
        best_end, best_len = -1, 0
        for position in reversed(positions):
            end = position + 1
            if end >= len(context):
                continue
            length = 0
            limit = min(self.max_extension, end)
            while length < limit and context[end - 1 - length] == context[-1 - length]:
                length += 1
            if length > best_len:
                best_end, best_len = end, length
                if length >= self.max_extension:
                    break
        self.last_match = best_len
        self.last_confident = False
        if best_end < 0 or best_len < self.min_match:
            return []
        self.last_confident = best_len >= self.confident_match
        return context[best_end : min(len(context), best_end + max_draft)]

    def observe(self, proposed: int, accepted: int) -> None:
        if proposed <= 0:
            return
        self.recent.append(accepted)
        self.recent[:] = self.recent[-self.window :]
        if len(self.recent) >= self.window and max(self.recent) == 0:
            self.silent_for = self.silence_rounds
            self.recent.clear()


@dataclass
class LaneState:
    """The C1 chain lane's history, held candidates, copy and exact counters."""

    context: list[int] = field(default_factory=list)
    lookup: SuffixLookup = field(default_factory=SuffixLookup)
    copy_width: int = 5
    granted: int | None = None
    kind: str = "none"
    rounds: int = 0
    conditional_acceptance: list[float] = field(default_factory=lambda: list(PRIOR))
    proposed: int = 0
    verified: int = 0
    accepted: int = 0
    committed: int = 0

    def copy_proposal(self, room: int = 6) -> list[int]:
        result = self.lookup.propose(self.context, min(self.copy_width, room - 1))
        return result if len(result) >= 2 and self.lookup.last_match >= 8 else []

    def head_depth(self, copy_ahead: bool, room: int = 16, shared: bool = False) -> int:
        # C1 familyRound calls draftLate -> headDepth, not budgets. Only shared
        # roundStreams applies the granted + draft_slack offer limit.
        want = min(5, max(1, room - 1))
        if shared and self.granted is not None:
            want = min(want, self.granted + 2)
        return 1 if copy_ahead and want else want

    def observe(self, drafts: int, accepted: int, committed: list[int]) -> None:
        if not 0 <= accepted <= drafts:
            raise ValueError("invalid accepted prefix")
        self.rounds += 1
        self.verified += drafts
        self.accepted += accepted
        self.committed += len(committed)
        if self.kind == "copy":
            self.lookup.observe(drafts, accepted)
            self.copy_width = (
                min(5, 2 * self.copy_width + 1) if accepted == drafts else 5
            )
        elif self.kind == "head":
            for j in range(min(drafts, len(self.conditional_acceptance))):
                if accepted < j:
                    break
                p = self.conditional_acceptance[j]
                self.conditional_acceptance[j] = p + 0.15 * (int(accepted > j) - p)
        self.context.extend(committed)


@dataclass
class LaneRule:
    """Active native C1 allocator; source cost table, live overhead EWMA."""

    costs: dict[int, float] = field(
        default_factory=lambda: {i + 1: v for i, v in enumerate(ROUND_MS)}
    )
    overhead: dict[int, float] = field(default_factory=dict)
    measured_cost: dict[int, float] = field(default_factory=dict)

    def overhead_ms(self, streams: int) -> float:
        if not self.overhead:
            return 8.0
        closest = min(self.overhead, key=lambda n: abs(n - streams))
        return self.overhead[closest]

    def observe_overhead(self, streams: int, rows: int, elapsed_ms: float) -> None:
        forward = self.costs.get(rows)
        if forward is None:
            return
        extra = max(0.0, elapsed_ms - forward)
        before = self.overhead.get(streams)
        self.overhead[streams] = (
            extra if before is None else before + 0.2 * (extra - before)
        )

    def observe_cost(self, drafts: int, ms: float, initializing: bool) -> None:
        if drafts <= 0 or initializing:
            return
        before = self.measured_cost.get(drafts)
        self.measured_cost[drafts] = (
            ms if before is None else before + 0.2 * (ms - before)
        )

    def observe_round(
        self, kind: str, drafts: int, elapsed_ms: float, initializing: bool
    ) -> None:
        # familyRound changes a zero-allocation head to kind=none. Neither
        # cost nor overhead is observed for that plain window or for copies.
        if kind == "head" and drafts > 0:
            self.observe_cost(drafts, elapsed_ms, initializing)
            self.observe_overhead(1, drafts + 1, elapsed_ms)

    def admit(
        self, lane: LaneState, confidence_logits: list[float], room: int = 16
    ) -> int:
        probabilities = survivals(confidence_logits)
        depth = allocate([1], [probabilities], self.costs, self.overhead_ms(1), room)[0]
        lane.granted = depth
        return depth


# Runtime optimization; original LaneRule above remains the source oracle.
@dataclass
class CalibratedLaneState(LaneState):
    # Thirty-two neutral observations shrink sparse/tail-position estimates.
    residual: list[float] = field(default_factory=lambda: [0.0] * 5)
    trials: list[int] = field(default_factory=lambda: [0] * 5)

    def observe_confidence(self, logits: list[float], drafts: int, accepted: int):
        """Only positions reached before the first rejection are observations."""
        for j in range(min(drafts, accepted + 1, len(logits))):
            probability = survivals([logits[j]])[0]
            self.residual[j] += int(accepted > j) - probability
            self.trials[j] += 1

    def calibrated_survivals(self, logits: list[float]) -> list[float]:
        if not any(self.trials):
            return survivals(logits)
        result, reach = [], 1.0
        for j, logit in enumerate(logits):
            raw = survivals([logit])[0]
            correction = self.residual[j] / (32 + self.trials[j])
            correction = max(-0.2, min(0.2, correction))
            reach *= max(0.0, min(1.0, raw + correction))
            result.append(reach)
        return result


class CalibratedLaneRule(LaneRule):
    def admit(
        self, lane: CalibratedLaneState, confidence_logits: list[float], room: int = 16
    ) -> int:
        probabilities = lane.calibrated_survivals(confidence_logits)
        depth = allocate([1], [probabilities], self.costs, self.overhead_ms(1), room)[0]
        lane.granted = depth
        return depth


class MultiLaneRule(CalibratedLaneRule):
    """Joint prefix allocation; C1 retains the existing allocator exactly."""

    def admit_uniform(self, entries, max_rows):
        """Maximize shared-window yield under uniform CUDA graph geometry.

        Every active request gets the same valid prefix length. Copies supply
        unit survival priors; head prefixes retain request-local calibration.
        Ragged prompt batches continue through admit_batch and eager execution.
        """
        if any(base != 1 for base, _, _, _ in entries):
            raise ValueError("uniform windows require one anchor per request")
        streams = len(entries)
        if not streams:
            raise ValueError("empty uniform batch")
        probabilities = [
            [1.0] * copy if copy else lane.calibrated_survivals(logits)
            for _, lane, logits, copy in entries
        ]
        limit = min(min(map(len, probabilities)), max_rows // streams - 1)
        if limit < 0:
            raise ValueError("anchor budget exceeds available rows")
        end = max(self.costs)
        marginal = self.costs[end] - self.costs[end - 1]
        overhead = self.overhead_ms(streams)
        best, depth = -1.0, 0
        for k in range(limit + 1):
            rows = streams * (k + 1)
            cost = self.costs.get(rows, self.costs[end] + marginal * (rows - end))
            expected = streams + sum(sum(p[:k]) for p in probabilities)
            rate = expected / (cost + overhead)
            if rate > best:
                best, depth = rate, k
        return [depth] * streams

    def admit_batch(self, entries, max_rows):
        """Allocate head prefixes after reserving prompt and copy rows.

        Entries are (base rows, lane, held confidence, copy depth). The pinned
        cost prior ends at 24 rows: larger windows use its last marginal cost,
        explicitly as an extrapolated prior, rather than a measured cost.
        """
        if len(entries) == 1:
            base, lane, logits, copy = entries[0]
            return [copy if copy else self.admit(lane, logits, max_rows)]
        fixed = [base + copy for base, _, _, copy in entries]
        probabilities = [
            [] if copy else lane.calibrated_survivals(logits)
            for _, lane, logits, copy in entries
        ]
        costs = dict(self.costs)
        end = max(costs)
        marginal = costs[end] - costs[end - 1]
        for rows in range(end + 1, max_rows + 1):
            costs[rows] = costs[end] + marginal * (rows - end)
        allocated = allocate(
            fixed, probabilities, costs, self.overhead_ms(len(entries)), max_rows
        )
        return [n + entry[3] for n, entry in zip(allocated, entries)]
