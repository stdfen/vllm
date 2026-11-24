# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2's layered candidate tree and target-guided branch selection."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

_Result = TypeVar("_Result")


@dataclass
class DraftTree:
    # Each depth has a vocabulary shortlist. Its incoming edges are indexed
    # by the previous depth's candidate slot (slot zero for the root).
    token_ids: list[list[int]]
    scores: list[list[list[float]]]

    def follow(self, path: list[int], accepted: int, target_token: int) -> list[int]:
        """Replace a rejected suffix if the target chose a valid sibling."""
        if not 0 <= accepted < len(path) <= len(self.token_ids):
            return path
        if path[accepted] == target_token:
            return path
        try:
            parent = (
                self.token_ids[accepted - 1].index(path[accepted - 1])
                if accepted
                else 0
            )
            child = self.token_ids[accepted].index(target_token)
        except ValueError:
            return path
        if not math.isfinite(self.scores[accepted][parent][child]):
            return path

        branch = path[:accepted] + [target_token]
        for depth in range(accepted + 1, len(path)):
            edges = self.scores[depth][child]
            ranked = (i for i, score in enumerate(edges) if math.isfinite(score))
            child = max(ranked, key=edges.__getitem__, default=-1)
            if child < 0:
                return path
            branch.append(self.token_ids[depth][child])
        return branch


def _advances(
    before: list[list[int]], after: list[list[int]], changed: list[bool]
) -> bool:
    """A replay may extend changed requests and must preserve every old token."""
    return all(
        new[: len(old)] == old and (len(new) > len(old) if edited else new == old)
        for old, new, edited in zip(before, after, changed, strict=True)
    )


def verify_paths(
    paths: list[list[int]],
    trees: list[DraftTree | None],
    run: Callable[[list[list[int]], bool], tuple[_Result, list[list[int]]]],
    max_replays: int,
) -> tuple[_Result, bool]:
    """Run the initial draft, then follow target-selected siblings if present.

    ``run`` verifies one path per request. Its replay flag requires resetting
    the target to the original committed prefix before the forward.
    """
    result, emitted = run(paths, False)
    committed_branch = False
    for _ in range(max_replays):
        alternatives = [
            tree.follow(path, len(row) - 1, row[-1])
            if tree is not None and row
            else path
            for tree, path, row in zip(trees, paths, emitted, strict=True)
        ]
        changed = [alt != path for alt, path in zip(alternatives, paths, strict=True)]
        if not any(changed):
            break

        probe, probe_emitted = run(alternatives, True)
        if not _advances(emitted, probe_emitted, changed):
            result, recovered = run(paths, True)
            if recovered != emitted:
                raise RuntimeError("Tree-DFlash2 could not reproduce its saved path")
            break

        paths, emitted, result = alternatives, probe_emitted, probe
        committed_branch = True
    return result, committed_branch
