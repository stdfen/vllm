# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side indexing for TensorFold trees in a vLLM decode batch."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from itertools import accumulate

C1_GRAPH_ROWS = 16


@dataclass(frozen=True)
class TreeWindow:
    """A pending token followed by candidates in parent-before-child order."""

    tokens: tuple[int, ...]
    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.tokens) != len(self.parents)
            or not 1 <= len(self.tokens) <= 32
            or self.parents[0] != -1
            or any(not 0 <= p < i for i, p in enumerate(self.parents[1:], 1))
        ):
            raise ValueError("A tree window needs one root and up to 31 draft nodes.")

    def prefix(self, rows: int) -> "TreeWindow":
        if not 1 <= rows <= len(self.tokens):
            raise ValueError("Scheduled rows must be a prefix of the proposed tree.")
        return TreeWindow(self.tokens[:rows], self.parents[:rows])

    def pad(self, rows: int) -> "TreeWindow":
        """Append provisional leaves. Acceptance still uses the unpadded tree."""
        extra = rows - len(self.tokens)
        if extra < 0:
            raise ValueError("Cannot pad a tree to fewer rows.")
        return TreeWindow(self.tokens + (0,) * extra, self.parents + (0,) * extra)

    def select(self, predictions: list[int]) -> tuple[list[int], list[int]]:
        """Return accepted row indices and emitted ids, including the bonus id."""
        if len(predictions) != len(self.tokens):
            raise ValueError("Each tree node needs one target prediction.")
        children: dict[tuple[int, int], int] = {}
        for row, (parent, token) in enumerate(zip(self.parents, self.tokens)):
            children.setdefault((parent, token), row)

        path, emitted = [], []
        row = 0
        while row >= 0:
            path.append(row)
            token = predictions[row]
            emitted.append(token)
            row = children.get((row, token), -1)
        return path, emitted

    def chose_sibling(self, path: list[int]) -> bool:
        # Best-first puts the preferred child first for each parent even
        # when unrelated branches are interleaved in the packed rows.
        return any(
            self.parents.index(parent) != child for parent, child in zip(path, path[1:])
        )


@dataclass
class TreeBatchPlan:
    starts: list[int]
    roots: list[int]
    depths: list[int]
    ancestor_bits: list[int]
    streams: list[int]

    @classmethod
    def build(cls, trees: list[TreeWindow]) -> "TreeBatchPlan":
        starts = list(accumulate((len(t.tokens) for t in trees), initial=0))
        roots, depths, bits, streams = [], [], [], []
        for request, tree in enumerate(trees):
            start = starts[request]
            for row, parent in enumerate(tree.parents):
                roots.append(start)
                depths.append(0 if parent < 0 else depths[start + parent] + 1)
                parent_bits = 0 if parent < 0 else bits[start + parent]
                bits.append(parent_bits | (1 << row))
                streams.append(request)
        return cls(starts, roots, depths, bits, streams)

    def convolution_windows(
        self, trees: list[TreeWindow], kept: int
    ) -> list[list[int]]:
        # TensorFold identifies old history by indices smaller than `kept`,
        # and new work by global packed row plus `kept`.
        windows: list[list[int]] = []
        for request, tree in enumerate(trees):
            start = self.starts[request]
            for row, parent in enumerate(tree.parents):
                history = (
                    list(range(kept)) if parent < 0 else windows[start + parent][1:]
                )
                windows.append([*history, kept + start + row])
        return windows

    def commit_rows(self, paths: list[list[int]]) -> tuple[list[int], list[int]]:
        sources, targets = [], []
        for start, path in zip(self.starts[:-1], paths, strict=True):
            sources.extend(start + row for row in path)
            targets.extend(range(start, start + len(path)))
        return sources, targets

    def convolution_commit_indices(
        self, paths: list[list[int]], kept: int
    ) -> list[int]:
        """Select each stream's tail from [all histories | packed model rows]."""
        prefix_size = len(paths) * kept
        indices = []
        for stream, (start, path) in enumerate(
            zip(self.starts[:-1], paths, strict=True)
        ):
            accepted = len(path)
            indices.extend(stream * kept + i for i in range(accepted, kept))
            indices.extend(prefix_size + start + row for row in path[-kept:])
        return indices


def c8_attention_chunk_bucket(
    lengths: Sequence[int], widths: Sequence[int], chunk_size: int
) -> int | None:
    if len(lengths) != 8 or tuple(widths) != (C1_GRAPH_ROWS,) * 8:
        return None
    return batch_attention_chunk_bucket(lengths, widths, chunk_size)


def batch_attention_chunk_bucket(
    lengths: Sequence[int], widths: Sequence[int], chunk_size: int
) -> int | None:
    if (
        len(lengths) != len(widths)
        or not 2 <= len(lengths) <= 8
        or min(lengths) < 2048
        or any(not 1 <= width <= C1_GRAPH_ROWS for width in widths)
    ):
        return None
    extent = max(length + width for length, width in zip(lengths, widths, strict=True))
    chunks = (extent + chunk_size - 1) // chunk_size
    if 5 <= chunks <= 8:
        return (chunks + 1) // 2 * 2
    return None


def can_capture_batch_verification_graph(
    shapes: Collection[tuple], requests: int
) -> bool:
    return 2 <= requests <= 8 and all(shape[0] != requests for shape in shapes)


def is_long_verification_graph(requests: int, rows: int, chunks: int) -> bool:
    return requests == 8 and rows == 128 and 17 <= chunks <= 19


def can_capture_verification_graph(
    shapes: Collection[tuple], requests: int, expanded_c1: bool
) -> bool:
    if expanded_c1:
        return len(shapes) < 6
    return len(shapes) < 8 and sum(shape[0] == requests for shape in shapes) < 2


class VerifyCosts:
    """Cost curve used by TensorFold's expected-tokens-per-ms allocator."""

    # GB10 calibration from the pinned TensorFold reference. Real model
    # forwards replace these estimates, independently for each observed width.
    _initial = (
        (1, 82.8),
        (2, 83.1),
        (4, 83.6),
        (8, 83.6),
        (12, 85.1),
        (16, 87.1),
        (24, 89.2),
        (32, 90.9),
        (48, 100.8),
        (64, 104.5),
        (96, 154.0),
        (128, 160.6),
    )

    def __init__(self) -> None:
        self.observed: dict[int, float] = {}

    def observe(self, rows: int, ms: float) -> None:
        previous = self.observed.get(rows, ms)
        self.observed[rows] = 0.875 * previous + 0.125 * ms

    def curve(self, *, use_reference: bool = False) -> Callable[[int], float]:
        if use_reference:
            return lambda rows: self._interpolate(self._initial, rows)
        ratios = sorted(
            (rows, ms / self._interpolate(self._initial, rows))
            for rows, ms in self.observed.items()
        )
        estimates = {
            rows: ms * (self._interpolate(ratios, rows) if ratios else 1.0)
            for rows, ms in self._initial
        }
        estimates.update(self.observed)
        points = sorted(estimates.items())
        for i in range(1, len(points)):
            rows, ms = points[i]
            points[i] = rows, max(ms, points[i - 1][1])
        return lambda rows: self._interpolate(points, rows)

    @staticmethod
    def _interpolate(points: Sequence[tuple[int, float]], rows: int) -> float:
        if rows <= points[0][0]:
            return points[0][1]
        for (r0, t0), (r1, t1) in zip(points, points[1:]):
            if rows <= r1:
                return t0 + (t1 - t0) * (rows - r0) / (r1 - r0)
        return points[-1][1]
