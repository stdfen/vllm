# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-size branch layout for K=7 packed verification."""

from vllm.v1.worker.gpu.spec_decode.dflash2.tree import DraftTree

PARENTS = (-1, 0, 0, 1, 2, 3, 4, 5)
DEPTHS = (0, 1, 1, 2, 2, 3, 3, 4)
LEVELS = ((0,), (1, 2), (3, 4), (5, 6), (7,))
WIDTH = len(PARENTS)


def ancestor(node: int, distance: int) -> int:
    for _ in range(distance):
        if node < 0:
            break
        node = PARENTS[node]
    return node


def ancestor_bits(node: int) -> int:
    return sum(1 << ancestor(node, i) for i in range(DEPTHS[node] + 1))


def conv_history_rows(history: int) -> list[list[int]]:
    """Gather conv inputs from [committed history | packed node inputs]."""
    rows = []
    for node, depth in enumerate(DEPTHS):
        row = []
        for distance in range(history, 0, -1):
            parent = ancestor(node, distance)
            row.append(history + parent if parent >= 0 else history + depth - distance)
        rows.append(row)
    return rows


def commit_rows(paths: list[list[int]]) -> tuple[list[int], list[int]]:
    """Map accepted packed rows back to their ordinary scheduler slot order."""
    sources = [req * WIDTH + node for req, path in enumerate(paths) for node in path]
    targets = [
        req * WIDTH + i for req, path in enumerate(paths) for i in range(len(path))
    ]
    return sources, targets


def pack_drafts(path: list[int], tree: DraftTree) -> list[int] | None:
    """Pack the primary chain and the highest ranked complete root sibling."""
    if len(path) != WIDTH - 1 or min(path) < 0 or len(tree.token_ids) < WIDTH - 1:
        return None
    root_edges = tree.scores[0][0]
    for slot in sorted(
        range(len(root_edges)), key=root_edges.__getitem__, reverse=True
    ):
        token = tree.token_ids[0][slot]
        if token == path[0]:
            continue
        sibling = tree.follow(path[:3], 0, token)
        if sibling[0] != path[0]:
            return [
                path[0],
                sibling[0],
                path[1],
                sibling[1],
                path[2],
                sibling[2],
                path[3],
            ]
    return None


def select_path(tokens: list[int], target: list[int]) -> tuple[list[int], list[int]]:
    """Return verified nodes and emitted tokens including the target bonus."""
    path = [0]
    emitted = []
    while True:
        parent = path[-1]
        token = target[parent]
        emitted.append(token)
        child = next(
            (
                i
                for i in range(parent + 1, WIDTH)
                if PARENTS[i] == parent and tokens[i] == token
            ),
            None,
        )
        if child is None:
            return path, emitted
        path.append(child)
