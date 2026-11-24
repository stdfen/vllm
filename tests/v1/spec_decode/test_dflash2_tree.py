# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.spec_decode.dflash2.tree import DraftTree, verify_paths
from vllm.v1.worker.gpu.spec_decode.dflash2.tree_layout import (
    ancestor_bits,
    commit_rows,
    conv_history_rows,
    pack_drafts,
    select_path,
)


def candidate_tree() -> DraftTree:
    return DraftTree(
        token_ids=[[10, 11], [20, 21], [30, 31]],
        scores=[
            [[4, 3], [4, 3]],
            [[4, 3], [3, 4]],
            [[4, 3], [3, 4]],
        ],
    )


def test_packed_layout_preserves_branches():
    tree = DraftTree(
        token_ids=[[i, i + 1] for i in range(10, 80, 10)],
        scores=[[[4, 3], [3, 4]] for _ in range(7)],
    )
    row = pack_drafts([10, 20, 30, 40, 50, 60, 70], tree)
    assert row == [10, 11, 20, 21, 30, 31, 40]
    # Node 6 can see root and its two ancestors, plus itself.
    assert ancestor_bits(6) == 0b01010101
    assert ancestor_bits(7) == 0b10101011


def test_packed_conv_history_is_branch_local():
    history = conv_history_rows(3)
    assert history == [
        [0, 1, 2],
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 5],
        [3, 4, 6],
        [3, 5, 7],
        [4, 6, 8],
    ]


@pytest.mark.parametrize(
    ("predictions", "nodes", "emitted"),
    [
        ([11, 20, 21, 30, 31, 40, 99, 99], [0, 2, 4, 6], [11, 21, 31, 99]),
        ([10, 20, 21, 31, 30, 40, 99, 99], [0, 1, 3], [10, 20, 31]),
        ([99, 20, 21, 30, 31, 40, 99, 99], [0], [99]),
        ([10, 20, 21, 30, 31, 40, 99, 99], [0, 1, 3, 5, 7], [10, 20, 30, 40, 99]),
    ],
)
def test_packed_verify_selects_only_direct_children(predictions, nodes, emitted):
    tokens = [0, 10, 11, 20, 21, 30, 31, 40]
    assert select_path(tokens, predictions) == (nodes, emitted)


def test_packed_commit_orders_chosen_rows_for_the_scheduler():
    src, dst = commit_rows([[0, 2, 4, 6], [0, 1, 3], [0]])
    assert src == [0, 2, 4, 6, 8, 9, 11, 16]
    assert dst == [0, 1, 2, 3, 8, 9, 10, 16]


@pytest.mark.parametrize(
    ("prefix", "target", "expected"),
    [
        (0, 11, [11, 21, 31]),
        (1, 21, [10, 21, 31]),
        (2, 31, [10, 20, 31]),
        (0, 99, [10, 20, 30]),
        (3, 99, [10, 20, 30]),
    ],
)
def test_tree_child_preserves_accepted_prefix(prefix, target, expected):
    assert candidate_tree().follow([10, 20, 30], prefix, target) == expected


def test_tree_does_not_follow_a_masked_edge():
    tree = candidate_tree()
    tree.scores[1][0][1] = float("-inf")
    assert tree.follow([10, 20, 30], 1, 21) == [10, 20, 30]
    assert tree.follow([11, 20, 30], 1, 21) == [11, 21, 31]


def test_tree_does_not_pad_invalid_target_inputs():
    tree = candidate_tree()
    tree.scores[2][1] = [float("-inf"), float("nan")]
    assert tree.follow([10, 20, 30], 1, 21) == [10, 20, 30]


def test_replay_bypasses_unrelated_and_empty_paths():
    paths = [[10, 20, 30], []]

    def run(selected, replay):
        assert selected == paths and not replay
        return selected, [[99], [11]]

    assert verify_paths(paths, [None, candidate_tree()], run, 2) == (paths, False)


def test_target_guided_replay_with_short_batch_member():
    calls = []
    targets = [[11, 21, 30, 99], [10, 90]]

    def run(paths, replay):
        calls.append((paths, replay))
        emitted = []
        for path, target in zip(paths, targets, strict=True):
            mismatch = next(
                (i for i, token in enumerate(path) if token != target[i]),
                len(path),
            )
            emitted.append(target[: mismatch + 1])
        return emitted, emitted

    output, used_branch = verify_paths(
        [[10, 20, 30], [10]], [candidate_tree(), candidate_tree()], run, max_replays=2
    )
    assert used_branch
    assert output == targets
    assert calls == [
        ([[10, 20, 30], [10]], False),
        ([[11, 21, 31], [10]], True),
        ([[11, 21, 30], [10]], True),
    ]


def test_replay_budget_is_bounded():
    observations = iter([[[11]], [[11, 21, 30]]])

    def run(paths, replay):
        return paths, next(observations)

    output, used_branch = verify_paths(
        [[10, 20, 30]], [candidate_tree()], run, max_replays=1
    )
    assert used_branch
    assert output == [[11, 21, 31]]


@pytest.mark.parametrize("recoverable", [True, False])
def test_divergent_replay_restores_previous_path(recoverable):
    recorded = []
    observations = iter([[[11]], [[12]], [[11 if recoverable else 12]]])

    def run(paths, replay):
        recorded.append((paths, replay))
        return paths, next(observations)

    if recoverable:
        assert verify_paths([[10, 20, 30]], [candidate_tree()], run, 2) == (
            [[10, 20, 30]],
            False,
        )
    else:
        with pytest.raises(RuntimeError, match="could not reproduce"):
            verify_paths([[10, 20, 30]], [candidate_tree()], run, 2)
    assert recorded == [
        ([[10, 20, 30]], False),
        ([[11, 21, 31]], True),
        ([[10, 20, 30]], True),
    ]
