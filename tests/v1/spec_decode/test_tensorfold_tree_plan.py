# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
    TreeBatchPlan,
    TreeWindow,
    VerifyCosts,
    batch_attention_chunk_bucket,
    c8_attention_chunk_bucket,
    can_capture_batch_verification_graph,
    can_capture_verification_graph,
    is_long_verification_graph,
)


@pytest.mark.parametrize(
    ("lengths", "widths", "expected"),
    [
        ([2048] * 8, [16] * 8, 6),
        ([2048] * 7 + [3056], [16] * 8, 6),
        ([2048] * 7 + [3057], [16] * 8, 8),
        ([4080] * 8, [16] * 8, 8),
        ([4081] * 8, [16] * 8, None),
        ([2047] + [3000] * 7, [16] * 8, None),
        ([8192] * 8, [16] * 8, None),
        ([2048] * 7, [16] * 7, None),
        ([2048] * 8, [15, 17] + [16] * 6, None),
    ],
)
def test_short_c8_attention_bucket_boundaries(lengths, widths, expected):
    assert c8_attention_chunk_bucket(lengths, widths, 512) == expected


@pytest.mark.parametrize("requests", range(2, 9))
def test_short_multi_request_bucket_keeps_variable_widths(requests):
    lengths = [3057, 3040] + [2048] * (requests - 2)
    widths = [1, 16] + [7] * (requests - 2)
    assert batch_attention_chunk_bucket(lengths, widths, 512) == 6
    widths[0], widths[1] = widths[1], widths[0]
    assert batch_attention_chunk_bucket(lengths, widths, 512) == 8


@pytest.mark.parametrize(
    ("lengths", "widths", "expected"),
    [
        ([3056, 3057], [16, 15], 6),
        ([3056, 3057], [15, 16], 8),
        ([4080, 2048], [16, 1], 8),
        ([4081, 2048], [16, 1], None),
        ([8192, 8192], [8, 16], None),
        ([2047, 2048], [8, 16], None),
        ([2048], [16], None),
        ([2048, 2048], [16], None),
        ([2048, 2048], [16, 0], None),
        ([2048, 2048], [16, 17], None),
        ([2048] * 9, [16] * 9, None),
    ],
)
def test_multi_request_bucket_uses_per_request_extent(lengths, widths, expected):
    assert batch_attention_chunk_bucket(lengths, widths, 512) == expected


def test_multi_request_graph_bank_limits_captures_per_real_batch_size():
    shapes = []
    for requests in (8, 3, 7, 4, 2, 6, 5):
        assert can_capture_batch_verification_graph(shapes, requests)
        shapes.append((requests, 6, 4))
        assert not can_capture_batch_verification_graph(shapes, requests)
    assert not can_capture_batch_verification_graph(shapes, 1)
    assert not can_capture_batch_verification_graph(shapes, 9)


@pytest.mark.parametrize(
    ("requests", "rows", "chunks", "eligible"),
    [
        (8, 128, 16, False),
        (8, 128, 17, True),
        (8, 128, 18, True),
        (8, 128, 19, True),
        (8, 128, 20, False),
        (8, 127, 18, False),
        (1, 16, 18, False),
        (7, 112, 18, False),
    ],
)
def test_long_context_graph_eligibility(requests, rows, chunks, eligible):
    assert is_long_verification_graph(requests, rows, chunks) is eligible


def test_graph_bank_reserves_additional_recurring_c1_shapes():
    c1_keys = [(1, rows, bucket) for rows, bucket in [(16, 2), (16, 4), (13, 4)]]
    assert not can_capture_verification_graph(c1_keys[:2], 1, expanded_c1=False)
    assert can_capture_verification_graph(c1_keys[:2], 1, expanded_c1=True)
    c1_keys.extend((1, rows, 4) for rows in (12, 14, 15))
    assert can_capture_verification_graph(c1_keys[:-1], 1, expanded_c1=True)
    assert not can_capture_verification_graph(c1_keys, 1, expanded_c1=True)
    other_shapes = [(requests, 16, 4) for requests in range(2, 10)]
    assert can_capture_verification_graph(other_shapes[:-1], 1, expanded_c1=False)
    assert not can_capture_verification_graph(other_shapes, 1, expanded_c1=False)


def test_tensorfold_padding_does_not_change_logical_paths():
    logical = TreeWindow((11, 12, 13, 14), (-1, 0, 0, 2))
    padded = logical.pad(16)
    logical_plan = TreeBatchPlan.build([logical])
    padded_plan = TreeBatchPlan.build([padded])
    assert padded.tokens[:4] == logical.tokens
    assert padded_plan.ancestor_bits[:4] == logical_plan.ancestor_bits
    assert padded_plan.depths[:4] == logical_plan.depths

    # Zero is a valid model token and matches a provisional padding leaf.
    assert logical.select([0, 6, 5, 7]) == ([0], [0])
    assert logical.select([13, 6, 14, 0]) == ([0, 2, 3], [13, 14, 0])
    paths = [[0, 2, 3]]
    assert padded_plan.commit_rows(paths) == logical_plan.commit_rows(paths)
    assert padded_plan.convolution_commit_indices(
        paths, 4
    ) == logical_plan.convolution_commit_indices(paths, 4)


def test_tensorfold_variable_widths_preserve_sibling_paths_and_scatter_order():
    first = TreeWindow((1, 4, 5, 6, 7), (-1, 0, 0, 1, 2))
    second = TreeWindow((9, 10), (-1, 0))
    plan = TreeBatchPlan.build([first, second])
    assert plan.starts == [0, 5, 7]
    assert plan.roots == [0, 0, 0, 0, 0, 5, 5]
    assert plan.depths == [0, 1, 1, 2, 2, 0, 1]
    assert plan.ancestor_bits == [1, 3, 5, 11, 21, 1, 3]
    assert plan.convolution_windows([first, second], 2) == [
        [0, 1, 2],
        [1, 2, 3],
        [1, 2, 4],
        [2, 3, 5],
        [2, 4, 6],
        [0, 1, 7],
        [1, 7, 8],
    ]
    path, outputs = first.select([5, 6, 7, 99, 42])
    assert path == [0, 2, 4]
    assert outputs == [5, 7, 42]
    assert first.chose_sibling(path)
    assert not first.chose_sibling([0, 1, 3])
    assert second.select([99, 42]) == ([0], [99])
    assert plan.commit_rows([path, [0]]) == ([0, 2, 4, 5], [0, 1, 2, 5])
    # Three accepted rows replace all old history on the first stream.
    # The second keeps one historical row followed by the accepted root.
    assert plan.convolution_commit_indices([path, [0]], 2) == [6, 8, 3, 9]


def test_tensorfold_scheduler_truncation_keeps_ancestors_and_bonus():
    requested = TreeWindow((10, 20, 21, 30), (-1, 0, 0, 2))
    truncated = requested.prefix(3)
    # A target bonus may match an unscheduled child. The token is emitted,
    # but the child's state must wait until it is actually computed.
    assert truncated.select([21, 70, 30]) == ([0, 2], [21, 30])
    assert requested.select([21, 70, 30, 40]) == ([0, 2, 3], [21, 30, 40])


def test_tensorfold_wide_tree_commits_the_accepted_sibling_beyond_row_sixteen():
    parents = (-1, *range(15), 0, *range(16, 30), 1)
    tree = TreeWindow(tuple(range(32)), parents)
    plan = TreeBatchPlan.build([tree])
    predictions = [100] * 32
    predictions[0] = 16
    predictions[16:30] = list(range(17, 31))
    path, emitted = tree.select(predictions)
    assert path == [0, *range(16, 31)]
    assert emitted == [*range(16, 31), 100]
    assert tree.chose_sibling(path)
    assert plan.commit_rows([path]) == (path, list(range(16)))
    assert plan.ancestor_bits[30] == 1 | sum(1 << i for i in range(16, 31))
    assert tree.prefix(18).select(predictions[:18]) == ([0, 16, 17], [16, 17, 18])
    with pytest.raises(ValueError, match="up to 31"):
        TreeWindow((*tree.tokens, 32), (*tree.parents, 31))


def test_tensorfold_tree_rejects_out_of_order_parents():
    with pytest.raises(ValueError, match="one root"):
        TreeWindow((1, 2, 3), (-1, 2, 0))


def test_tensorfold_verify_cost_estimates_follow_observed_rows():
    costs = VerifyCosts()
    initial = costs.curve()
    assert initial(8) < initial(64)
    costs.observe(8, 94)
    costs.observe(64, 200)
    estimate = costs.curve()
    assert estimate(8) == 94
    assert estimate(64) == 200
    assert 94 < estimate(16) <= 200


def test_tensorfold_reference_costs_remain_fixed_during_online_sampling():
    costs = VerifyCosts()
    widths = [1, 2, 4, 8, 12, 13, 15, 16, 96, 128]
    reference = costs.curve(use_reference=True)
    expected = [82.8, 83.1, 83.6, 83.6, 85.1, 85.6, 86.6, 87.1, 154.0, 160.6]
    assert [reference(rows) for rows in widths] == pytest.approx(expected)
    costs.observe(13, 108.0)
    costs.observe(15, 115.0)
    assert costs.curve()(13) == 108.0
    assert costs.curve()(15) == 115.0
    assert [costs.curve(use_reference=True)(rows) for rows in widths] == (
        pytest.approx(expected)
    )
