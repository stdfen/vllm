# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(
    overlap: bool = False,
    device_accept: bool = False,
    device_metadata: bool = False,
    direct_recurrent_commit: bool = False,
):
    from tensorfold.cuda.kernels import gdn

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import (
        commit_paged_kv,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
        C1CommitGraph,
        KVCommitLayer,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
        TreeBatchPlan,
        TreeWindow,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        GDNRecord,
        StateSlots,
        prepare_resident_recurrent,
    )

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(192)
    width, page = 7, 16
    tokens = (0, 1, 2, 3, 4, 4, 6) if device_accept else tuple(range(width))
    tree = TreeWindow(tokens, (-1, 0, 0, 1, 2, 2, 3))
    plan = TreeBatchPlan.build([tree])

    def rand(shape):
        return torch.randn(shape, generator=rng, device=device).mul_(0.1)

    def slots(values):
        return torch.tensor(values, device=device, dtype=torch.int64)

    records = []
    for layer, kept in enumerate((3, 2)):
        records.append(
            GDNRecord(
                rand((width, 2, 128)).bfloat16(),
                rand((width, 2, 128)).bfloat16(),
                torch.full((width, 2), 0.9, device=device),
                torch.full((width, 2), 0.2, device=device),
                rand((width, 40)).bfloat16(),
                rand((12, 40, width + kept)).bfloat16(),
                rand((1, kept, 40)).bfloat16(),
                rand((12, 2, 128, 128)),
                StateSlots(slots([1]), slots([0]), [1], [[1] * width]),
                f"gdn-{layer}",
            )
        )
    mapping = slots(list(range(page - 1, page + width - 1)))
    scale_k, scale_v = (torch.tensor(x, device=device) for x in (0.5, 2.0))
    kv_layers = {
        name: KVCommitLayer(
            rand((width, 2, 128)).bfloat16(),
            rand((width, 2, 128)).bfloat16(),
            rand((4, 2, page, 256)).to(torch.float8_e4m3fn),
            scale_k,
            scale_v,
            mapping,
        )
        for name in ("kv-a", "kv-b")
    }
    graph = C1CommitGraph(
        records,
        kv_layers,
        width,
        device,
        overlap=overlap,
        device_accept=device_accept,
        direct_recurrent_commit=direct_recurrent_commit,
    )
    ref_recurrent = [record.recurrent.clone() for record in records]
    ref_conv = [record.conv.clone() for record in records]
    ref_kv = {name: layer.cache.clone() for name, layer in kv_layers.items()}
    sibling = 4 if device_accept else 5
    for i, path in enumerate(([0, 2, sibling], [0], [0, 1, 3, 6])):
        n = len(path)
        logical_width = 6 if i == 0 else max(path) + 1
        current = mapping[:logical_width] + page * (i % 2)
        accepted = slots(path)
        state_rows = []
        for j, (record, recurrent, conv) in enumerate(
            zip(records, ref_recurrent, ref_conv, strict=True)
        ):
            slot = i + j + 1
            state_slots = list(range(slot, slot + logical_width))
            record.slots.conv.fill_(slot)
            record.slots.all_slots = [state_slots]
            source = state_slots[n - 1] if i == 0 else state_slots[-1]
            if device_metadata:
                # Model the previous result having accepted more rows than
                # the current scheduler proposed for this verification window.
                reserved = list(range(slot, slot + width))
                if i == 1:
                    source = reserved[-1]
                record.slots.device_ids = slots(reserved).int()[None]
                record.slots.device_counts = slots([reserved.index(source) + 1]).int()
            record.slots.source = [source]
            target = recurrent[state_slots[n - 1]]
            if source != state_slots[n - 1]:
                target.copy_(recurrent[source])
            state_rows.append(target)
            kept = record.prefix.shape[1]
            indices = slots(plan.convolution_commit_indices([path], kept))
            history = torch.cat((record.prefix[0], record.qkv)).index_select(0, indices)
            conv[slot, :, n - 1 : n - 1 + kept] = history.T
        reference_pointers = gdn.replay_table(
            [r.k for r in records],
            [r.v for r in records],
            [r.g for r in records],
            [r.beta for r in records],
            [state_rows],
        )
        gdn.replay(
            slots(reference_pointers),
            len(records),
            1,
            accepted.int()[None],
            slots([n]).int(),
            records[0].k,
            records[0].v,
            in_place=True,
        )
        for name, layer in kv_layers.items():
            commit_paged_kv(
                layer.key,
                layer.value,
                accepted,
                current[:n],
                ref_kv[name],
                layer.key_scale,
                layer.value_scale,
            )
        mappings = dict.fromkeys(kv_layers, current)
        if device_accept:
            predictions = [0] * width
            for parent, child in zip(path, path[1:]):
                predictions[parent] = tree.tokens[child]
            expected_path, expected_ids = tree.prefix(logical_width).select(
                predictions[:logical_width]
            )
            assert expected_path == path
            resident = None
            if device_metadata:
                layout, resident = prepare_resident_recurrent(
                    [(record.recurrent, record.slots) for record in records],
                    width,
                    device,
                )
                assert layout.flatten().tolist() == [
                    record.recurrent[record.slots.source[0]].data_ptr()
                    for record in records
                ]
            receipt = graph.accept_and_commit(
                tree.prefix(logical_width), slots(predictions), mappings, resident
            )
            assert receipt is not None
            actual_path, actual_ids, src_rows, dst_rows, replay = receipt
            assert actual_path == path and actual_ids == expected_ids
            assert replay == (i > 0)
            result = src_rows, dst_rows
        else:
            result = graph.commit(plan, path, mappings)
        assert result is not None
        assert graph.pending == (overlap and i > 0)
        draft_values = records[0].qkv.index_select(0, result[0]) + 1
        graph.wait()
        assert not graph.pending
        assert result[0].tolist() == path and result[1].tolist() == list(range(n))
        torch.testing.assert_close(
            draft_values, records[0].qkv[accepted] + 1, rtol=0, atol=0
        )
        for record, recurrent, conv in zip(
            records, ref_recurrent, ref_conv, strict=True
        ):
            torch.testing.assert_close(record.recurrent, recurrent, rtol=0, atol=0)
            torch.testing.assert_close(record.conv, conv, rtol=0, atol=0)
        for name, layer in kv_layers.items():
            assert torch.equal(
                layer.cache.view(torch.uint8), ref_kv[name].view(torch.uint8)
            )

    assert graph.commit(plan, [0], {"kv-a": mapping, "kv-b": mapping.clone()}) is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_commit_graph_joins_before_cache_reuse():
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(overlap=True)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_device_accepts_siblings_and_commits_pages():
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(device_accept=True)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_recurrent_replay_writes_selected_slots_directly():
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(
        device_accept=True, direct_recurrent_commit=True
    )
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(
        direct_recurrent_commit=True
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_ordinary_commit_replays_directly_to_accepted_slot():
    from tensorfold.cuda.kernels import gdn

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        GDNRecord,
        StateSlots,
        eager_recurrent_table,
    )

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(215)
    indices = torch.arange(1, device=device)
    k = torch.randn((6, 2, 128), device=device, generator=rng).mul_(0.1).bfloat16()
    v = torch.randn((6, 2, 128), device=device, generator=rng).mul_(0.1).bfloat16()
    g = torch.full((6, 2), 0.9, device=device)
    beta = torch.full((6, 2), 0.2, device=device)
    state = torch.randn((6, 2, 128, 128), device=device, generator=rng).mul_(0.1)
    expected = state.clone()
    unused = torch.empty(0, device=device)
    slots = StateSlots(indices, indices, [0], [[1, 2, 4, 5, 0, 3]])
    record = GDNRecord(k, v, g, beta, unused, unused, unused, state, slots, "gdn")

    for source, path in ((0, [0, 2, 4]), (1, [0]), (4, [0, 1])):
        slots.source = [source]
        accepted = len(path)
        destination = slots.all_slots[0][accepted - 1]
        if destination != source:
            expected[destination].copy_(expected[source])
        old_table = gdn.replay_table([k], [v], [g], [beta], [[expected[destination]]])
        new_table = eager_recurrent_table([record], [accepted], direct=True)
        assert new_table[-2:] == gdn.pointers([state[source], state[destination]])
        rows = gdn.to_device(path, torch.int32, device)[None]
        counts = gdn.to_device([accepted], torch.int32, device)
        for pointers in (old_table, new_table):
            table = gdn.to_device(pointers, torch.int64, device)
            gdn.replay(table, 1, 1, rows, counts, k, v, in_place=True)
        torch.testing.assert_close(state, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_device_accept_joins_side_stream():
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(
        overlap=True, device_accept=True
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_device_cache_metadata_drives_replay_and_commit():
    test_tensorfold_c1_commit_graph_changes_path_slots_and_pages(
        device_accept=True, device_metadata=True
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_reads_shared_gdn_slot_groups_and_exact_sequence_lengths():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify"
    )

    def metadata(rows, accepted):
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=2,
            num_spec_decode_tokens=6,
            num_actual_tokens=6,
            spec_state_indices_tensor=torch.tensor(
                rows, dtype=torch.int32, device="cuda"
            ),
            num_accepted_tokens=torch.tensor(
                accepted, dtype=torch.int32, device="cuda"
            ),
        )

    first = metadata([[2, 3, 4], [5, 6, 7]], [2, 1])
    second = metadata([[11, 13, 12], [21, 23, 22]], [3, 2])
    lengths, groups = ops.read_verify_metadata(
        torch.tensor([511, 529], dtype=torch.int32, device="cuda"),
        {"one": first, "shared": first, "other": second},
    )
    assert lengths == [511, 529]
    assert list(groups) == [id(first), id(second)]
    assert groups[id(first)].source == [3, 5]
    assert groups[id(second)].source == [12, 23]
    assert groups[id(second)].all_slots == [[11, 13, 12], [21, 23, 22]]
    assert groups[id(second)].conv.tolist() == [11, 21]
    assert groups[id(second)].offset.tolist() == [2, 1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_fp8_commit_writes_only_accepted_rows_across_pages():
    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention"
    )
    device = "cuda"
    page, heads, d = 16, 2, 128
    slots = torch.tensor([15, 16, 32], device=device)
    accepted = torch.tensor([0, 2, 4], device=device)
    key = torch.arange(1, 6, device=device, dtype=torch.bfloat16)[:, None, None]
    key = key.expand(-1, heads, d).contiguous()
    value = key * 2
    scales = torch.tensor([0.5, 2.0], device=device)
    for slot_major in (False, True):
        shape = (3, page, heads, 2 * d) if slot_major else (3, heads, page, 2 * d)
        cache = torch.zeros(shape, dtype=torch.uint8, device=device)
        if slot_major:
            cache = cache.transpose(1, 2)
        ops.commit_paged_kv(
            key, value, accepted, slots, cache.view(torch.float8_e4m3fn), *scales
        )
        expected = torch.zeros((3, page, heads, 2 * d), device=device)
        expected[slots // page, slots % page] = torch.cat(
            (key[accepted] * 2, value[accepted] * 0.5), dim=-1
        ).float()
        actual = cache.transpose(1, 2).view(torch.float8_e4m3fn).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_paged_attention_graph_updates_pages_and_sibling_paths(
    cache_dtype: torch.dtype = torch.float8_e4m3fn,
    round_fp8_tail: bool = False,
    prefix_shift: int = 0,
):
    attention = pytest.importorskip("tensorfold.cuda.kernels.attention")
    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention"
    )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(78)
    page, h, hk, d = 16, 4, 2, 128
    q = torch.randn(7, h, d, device=device, generator=generator).bfloat16()
    k, v = [
        torch.randn(7, hk, d, device=device, generator=generator).bfloat16()
        for _ in range(2)
    ]
    blocks_per_request = 36 + prefix_shift // page
    pages = 2 * blocks_per_request
    cache = torch.randn(pages, hk, page, 2 * d, device=device, generator=generator)
    cache = cache.to(cache_dtype)
    blocks = (
        torch.randperm(pages, device=device, generator=generator)
        .view(2, blocks_per_request)
        .int()
    )
    scales = torch.tensor([0.5, 2.0], device=device)
    lens = [prefix_shift + 511, prefix_shift + 521]
    parents = [(-1, 0, 0, 1, 2), (-1, 0)]
    plan = attention.plan(parents, lens, h // hk, device)
    output = torch.empty_like(q)
    partial = ops.partials(q, plan)

    def launched():
        ops.paged_attention(
            q, k, v, cache, blocks, plan, *scales, d**-0.5, output, partial,
            round_fp8_tail=round_fp8_tail,
        )

    def native():
        keys, values = [], []
        for row, length in enumerate(lens):
            seq = torch.arange(length, device=device)
            page_ids = blocks[row, seq // page].long()
            if cache_dtype == torch.float8_e4m3fn:
                packed = cache.view(torch.uint8).transpose(1, 2)[page_ids, seq % page]
                x, y = packed.view(cache_dtype).float().chunk(2, dim=-1)
                x = (x * scales[0]).bfloat16()
                y = (y * scales[1]).bfloat16()
            else:
                x, y = cache.transpose(1, 2)[page_ids, seq % page].chunk(2, dim=-1)
            keys.append(x.contiguous())
            values.append(y.contiguous())
        offs = torch.tensor(
            attention.offsets(list(zip(keys, values, strict=True)), device),
            device=device,
            dtype=torch.int64,
        ).view(2, 2)
        native_k, native_v = k, v
        if round_fp8_tail:
            # The accepted rows can straddle a page, with independent K/V scales.
            scratch = torch.zeros(
                (2, hk, page, 2 * d), device=device, dtype=torch.uint8
            )
            rows = torch.arange(k.shape[0], device=device)
            slots = rows + page - 2
            ops.commit_paged_kv(
                k, v, rows, slots, scratch.view(cache_dtype), *scales
            )
            packed = scratch.transpose(1, 2)[slots // page, slots % page]
            keys_fp8, values_fp8 = packed.view(cache_dtype).float().chunk(2, dim=-1)
            native_k = (keys_fp8 * scales[0]).bfloat16().contiguous()
            native_v = (values_fp8 * scales[1]).bfloat16().contiguous()
        return attention.attention(q, native_k, native_v, offs, plan, scale=d**-0.5)

    launched()
    torch.testing.assert_close(output, native(), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launched()
    blocks.copy_(blocks.flip(0))
    lens[:] = [prefix_shift + 510, prefix_shift + 522]
    updated = attention.plan([(-1, 0, 1, 0, 3), (-1, 0)], lens, h // hk, device)
    for name in ("streams", "items", "paths", "depths"):
        getattr(plan, name).copy_(getattr(updated, name))
    q.mul_(0.75)
    if round_fp8_tail:
        k.mul_(-0.625)
        v.mul_(1.125)
        scales.copy_(scales.flip(0) * 1.25)
    graph.replay()
    torch.testing.assert_close(output, native(), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_paged_attention_graph_updates_long_prefixes():
    attention = pytest.importorskip("tensorfold.cuda.kernels.attention")
    for chunks in (17, 18, 19):
        test_tensorfold_paged_attention_graph_updates_pages_and_sibling_paths(
            prefix_shift=(chunks - 2) * attention.CHUNK
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_batch_bucketed_attention_refreshes_full_chunks():
    attention = pytest.importorskip("tensorfold.cuda.kernels.attention")
    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention"
    )
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(11230)
    page, h, hk, d = 16, 4, 2, 128
    scales = torch.tensor([0.5, 2.0], device=device)

    def check_replay(widths, chunks):
        requests = len(widths)
        q, k, v = [
            torch.randn(sum(widths), heads, d, device=device, generator=rng).bfloat16()
            for heads in (h, hk, hk)
        ]

        def launch(cache, blocks, plan, out, scratch):
            ops.paged_attention(
                q, k, v, cache, blocks, plan, *scales, d**-0.5, out, scratch
            )

        columns = chunks * attention.CHUNK // page
        pages = requests * columns
        cache = torch.randn(pages, hk, page, 2 * d, device=device, generator=rng).to(
            torch.float8_e4m3fn
        )
        blocks = (
            torch.randperm(pages, device=device, generator=rng)
            .view(requests, columns)
            .int()
        )
        base = (chunks - 2) * attention.CHUNK
        lengths = [base + i for i in range(requests)]
        parents = [tuple(range(-1, width - 1)) for width in widths]
        bucket = ops.batch_graph_attention_plan(
            parents, lengths, h // hk, chunks, device
        )
        output = torch.empty_like(q)
        workspace = ops.partials(q, bucket)

        launch(cache, blocks, bucket, output, workspace)
        exact = attention.plan(parents, lengths, h // hk, device)
        expected = torch.empty_like(q)
        launch(cache, blocks, exact, expected, ops.partials(q, exact))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(cache, blocks, bucket, output, workspace)
        lengths = [base + attention.CHUNK + 11 * i for i in range(requests)]
        parents = [
            (-1, 0, 0, *range(2, width - 1))
            if width >= 3
            else tuple(range(-1, width - 1))
            for width in reversed(widths)
        ]
        refreshed = ops.batch_graph_attention_plan(
            parents, lengths, h // hk, chunks, device
        )
        assert bucket.items.shape == refreshed.items.shape
        for name in ("rows", "streams", "items", "paths", "depths"):
            getattr(bucket, name).copy_(getattr(refreshed, name))
        blocks.copy_(blocks.roll(1, dims=0))
        q.mul_(0.75)
        graph.replay()
        exact = attention.plan(parents, lengths, h // hk, device)
        launch(cache, blocks, exact, expected, ops.partials(q, exact))
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    for chunks in (6, 8):
        check_replay([16] * 8, chunks)
        for requests in range(2, 9):
            check_replay([16] + [1 + i for i in range(requests - 1)], chunks)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_batch_gdn_slots_share_one_graph():
    from dataclasses import replace

    gdn = pytest.importorskip("tensorfold.cuda.kernels.gdn")
    verify = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify"
    )
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(75)
    first = (-1, 0, 1, 1, 1, 4, 0, 4, 4, 8, 2, 1, 9, 9, 10, 3)
    second = (-1, 0, 0, 0, 3, 1, 1, 4, 7, 5, 0, 0, 4, 7, 4, 3)
    assert verify.gdn_slot_bucket(0) == 0
    assert verify.gdn_slot_bucket(3) == verify.gdn_slot_bucket(4) == 4

    def check_replay(widths):
        q, k, v = [
            torch.randn(sum(widths), 2, 128, device=device, generator=rng)
            .mul_(0.1)
            .bfloat16()
            for _ in range(3)
        ]
        g = torch.full((sum(widths), 2), 0.9, device=device)
        beta = torch.full_like(g, 0.2)
        before = gdn.plan([first[:width] for width in widths], device)
        updated = gdn.plan([second[:width] for width in reversed(widths)], device)
        assert [before.slots, updated.slots] == [3, 4]
        states = [
            torch.randn(2, 128, 128, device=device, generator=rng).mul_(0.1)
            for _ in widths
        ]
        pointers = gdn.to_device(gdn.pointers(states), torch.int64, device)
        captured_plan = replace(
            before,
            entries=before.entries.clone(),
            starts=before.starts.clone(),
            slots=4,
        )

        def run(plan):
            return gdn.tree(q, k, v, g, beta, plan, table=pointers)

        torch.testing.assert_close(run(captured_plan), run(before), atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(captured_plan)
        captured_plan.entries.copy_(updated.entries)
        captured_plan.starts.copy_(updated.starts)
        pointers.copy_(gdn.to_device(gdn.pointers(states[::-1]), torch.int64, device))
        graph.replay()
        torch.testing.assert_close(out, run(updated), atol=0, rtol=0)

    check_replay([16] * 8)
    for requests in range(2, 9):
        check_replay([16] + [3 + i for i in range(requests - 1)])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_fp8_tree_tail_matches_paged_cache_rounding():
    """Tree candidates use the cache writer's precision before and after replay."""
    test_tensorfold_paged_attention_graph_updates_pages_and_sibling_paths(
        round_fp8_tail=True
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_c1_graph_replays_across_prefix_chunk_boundaries(
    width: int = 7, pad_to: int | None = None, direct_stage: bool = False
):
    attention = pytest.importorskip("tensorfold.cuda.kernels.attention")
    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention"
    )
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(103)
    page, h, hk, d = 16, 4, 2, 128
    capacity = ops.C1_GRAPH_PREFIX_CAPACITY

    def rand(shape):
        return torch.randn(shape, device=device, generator=gen)

    rows = width if pad_to is None else pad_to
    q, k, v = [rand((rows, heads, d)).bfloat16() for heads in (h, hk, hk)]
    cache = rand((capacity // page, hk, page, 2 * d)).to(torch.float8_e4m3fn)
    blocks = torch.randperm(capacity // page, device=device)[None].int()
    scales = torch.tensor([0.5, 2.0], device=device)
    trees = ((-1, 0, 0, 1, 1, 2, 3), (-1, 0, 0, 1, 2, 2, 4))
    if width == 32:
        common = (-1, *range(15), 0, *range(16, 30))
        trees = ((*common, 1), (*common, 16))
    padding = (0,) * (rows - width)
    plan = ops.c1_graph_attention_plan(trees[0] + padding, 2047, h // hk, device)
    if direct_stage:
        from tensorfold.cuda.kernels import gdn

        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
            TreeBatchPlan,
            TreeWindow,
        )
        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
            C1GraphPlanBuffers,
        )

        buffers = C1GraphPlanBuffers(
            gdn.plan([trees[0]], device),
            plan,
            {3: torch.empty((width, 4), dtype=torch.int32, device=device)},
        )
        plan = buffers.attn_plan
    output = torch.empty_like(q)
    workspace = ops.partials(q, plan)

    def launch(active, destination, partial, q_in=q, k_in=k, v_in=v):
        ops.paged_attention(
            q_in,
            k_in,
            v_in,
            cache,
            blocks,
            active,
            *scales,
            d**-0.5,
            destination,
            partial,
        )

    launch(plan, output, workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(plan, output, workspace)
    eager = torch.empty_like(q[:width])
    for i, length in enumerate((2047, 2048, 2571, 3073, capacity, 511)):
        parents = trees[i % len(trees)]
        if direct_stage:
            window = TreeWindow(tuple(range(width)), parents)
            host = TreeBatchPlan.build([window])
            entries, _, _, _ = gdn.plan_host([parents])
            buffers.update(window, host, length, entries)
            assert buffers.gdn_plan.entries.flatten().tolist() == entries
            assert buffers.position_depths.tolist() == host.depths
            assert buffers.conv_windows[3].tolist() == host.convolution_windows(
                [window], 3
            )
        else:
            current = ops.c1_graph_attention_plan(
                parents + padding, length, h // hk, device
            )
            for name in ("streams", "paths", "depths"):
                getattr(plan, name).copy_(getattr(current, name))
        blocks.copy_(blocks.roll(1, dims=1))
        graph.replay()
        exact = attention.plan([parents], [length], h // hk, device)
        launch(
            exact,
            eager,
            ops.partials(q[:width], exact),
            q[:width],
            k[:width],
            v[:width],
        )
        torch.testing.assert_close(output[:width], eager, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_c1_direct_stage_refreshes_captured_tree():
    from types import SimpleNamespace

    from tensorfold.cuda.kernels import gdn

    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import (
        c1_graph_attention_plan,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
        TreeBatchPlan,
        TreeWindow,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        StateSlots,
        TensorFoldVerify,
    )

    test_tensorfold_c1_graph_replays_across_prefix_chunk_boundaries(direct_stage=True)
    device = torch.device("cuda")
    tree = TreeWindow((2, 3, 4), (-1, 0, 0))
    plan = TreeBatchPlan.build([tree])
    entries, _, _, _ = gdn.plan_host([tree.parents])
    zeros = torch.zeros(1, device=device, dtype=torch.int64)
    old = StateSlots(zeros.clone(), zeros.clone(), [0], [[0, 1, 2]])
    new = StateSlots(zeros + 5, zeros + 1, [6], [[5, 6, 7]])
    recurrent = torch.empty((8, 16), device=device)

    meta = object.__new__(TritonAttentionMetadata)
    meta.block_table = torch.arange(4, dtype=torch.int32, device=device)[None]
    gdn_meta = object()
    metadata = {"attention0": meta, "attention1": meta, "gdn": gdn_meta}
    dest = torch.zeros_like(meta.block_table)
    state = TensorFoldVerify.__new__(TensorFoldVerify)
    state.gdn_plan = gdn.plan([tree.parents], device)
    state.attention_plan = c1_graph_attention_plan(tree.parents, 511, 1, device)
    state.conv_windows = {3: torch.empty((3, 4), dtype=torch.int32, device=device)}
    state.gdn_table = torch.empty((1, 1), dtype=torch.int64, device=device)
    state.resident_recurrent = None
    state.block_tables = {"attention0": dest, "attention1": dest}
    state.state_slots = {"gdn": old}
    state.layers = {"gdn": SimpleNamespace(kv_cache=(None, recurrent))}
    state.metadata = metadata
    state.bind_c1_graph_inputs()

    depths = state.stage_c1(tree, plan, 512, entries, metadata, {id(gdn_meta): new})
    assert depths is not None and depths.tolist() == plan.depths
    assert state.gdn_table.item() == recurrent[new.source[0]].data_ptr()
    assert torch.equal(dest, meta.block_table)
    assert old.source == new.source and old.all_slots == new.all_slots
    assert torch.equal(old.conv, new.conv) and torch.equal(old.offset, new.offset)

    separate = object.__new__(TritonAttentionMetadata)
    separate.block_table = meta.block_table.clone()
    conflicting = dict(metadata, attention1=separate)
    assert (
        state.stage_c1(tree, plan, 513, entries, conflicting, {id(gdn_meta): new})
        is None
    )
    assert state.metadata is metadata


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_async_draft_upload_preserves_runner_rows():
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_round import (
        store_tensorfold_draft_tokens,
    )

    device = torch.device("cuda")
    buffer = torch.full((7, 15), -2, dtype=torch.int64, device=device)
    reference = buffer.clone()
    first = torch.tensor([5], dtype=torch.int64, device=device)
    second = torch.tensor([2], dtype=torch.int64, device=device)
    tree_one, tree_two = [[11, 20, 34]], [[62]]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        store_tensorfold_draft_tokens(tree_one, first, buffer, async_upload=True)
        store_tensorfold_draft_tokens(tree_two, second, buffer, async_upload=True)
        upload_done = stream.record_event()

    store_tensorfold_draft_tokens(tree_one, first, reference)
    store_tensorfold_draft_tokens(tree_two, second, reference)
    torch.cuda.current_stream().wait_event(upload_done)
    assert torch.equal(buffer, reference)
    assert buffer[5].tolist() == [11, 20, 34] + [-1] * 12
    assert buffer[2].tolist() == [62] + [-1] * 14
    assert tree_one == [[11, 20, 34]] and tree_two == [[62]]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_qmm_split_replays_selected_fp4_and_fp8_kernels():
    from tensorfold.cuda.nvfp4 import linear

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_target import (
        C1_QMM_ROWS,
        _C1SplitProjection,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1149)
    n, k = 256, 4096
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device=device, generator=generator
    )
    scales = torch.full((n, k // 16), 0.25, device=device).to(torch.float8_e4m3fn)
    fp8 = torch.randn((n, k), device=device, generator=generator)
    fp8 = fp8.to(torch.float8_e4m3fn)
    projections = (
        linear.Fp4Linear.from_checkpoint(packed, scales, 0.125),
        linear.Fp8Linear.from_checkpoint(fp8, 0.125),
    )
    inputs = torch.zeros((C1_QMM_ROWS, k), dtype=torch.bfloat16, device=device)
    selected = [_C1SplitProjection(p, splits=2) for p in projections]
    for tuned in selected:
        tuned(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [tuned(inputs) for tuned in selected]

    for scale in (0.5, 1.25):
        inputs.normal_(generator=generator).mul_(scale)
        graph.replay()
        for output, stock, tuned in zip(outputs, projections, selected, strict=True):
            expected = torch.empty_like(output)
            linear._ext().qmmf(
                inputs,
                tuned.packed,
                tuned.scales,
                tuned.scale,
                expected,
                None,
                tuned.mode,
                n,
                2,
                tuned.npad,
                C1_QMM_ROWS,
                False,
            )
            assert torch.equal(output, expected)
            assert torch.isfinite(output).all()
            reference = stock(inputs).float()
            relative_error = (
                torch.linalg.vector_norm(output.float() - reference)
                / torch.linalg.vector_norm(reference)
            )
            assert relative_error < 0.01


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_grouped_resident_layout_refreshes_distinct_slots():
    from tensorfold.cuda.kernels import gdn

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        C1ResidentRecurrentLayout,
        StateSlots,
        prepare_resident_recurrent,
    )

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(1248)
    parents = (-1, 0, 0, 2)
    width = len(parents)
    plan = gdn.plan([parents], device)
    q, k, v = [
        torch.randn((width, 2, 128), generator=rng, device=device)
        .mul_(0.1)
        .bfloat16()
        for _ in range(3)
    ]
    g = torch.full((width, 2), 0.9, device=device)
    beta = torch.full_like(g, 0.2)
    caches = [
        torch.randn((9, 2, 128, 128), generator=rng, device=device).mul_(0.1)
        for _ in range(3)
    ]
    group_indices = (1, 0, 1)
    layout = C1ResidentRecurrentLayout(list(zip(caches, group_indices, strict=True)))

    def group_slots(ids, count):
        device_ids = gdn.to_device(ids, torch.int32, device).view(1, -1)
        counts = gdn.to_device([count], torch.int32, device)
        return StateSlots(
            device_ids[:, 0].long(), counts.long() - 1, [], [], device_ids, counts
        )

    def reference(groups):
        layers = [
            (cache, groups[i]) for cache, i in zip(caches, group_indices, strict=True)
        ]
        return prepare_resident_recurrent(layers, width, device)

    initial = [group_slots([1, 3, 4, 6, 8], 2), group_slots([2, 4, 5, 6, 7], 1)]
    expected_table, expected_commit = reference(initial)
    table, replay_rows = (
        torch.empty_like(expected_table),
        torch.empty_like(expected_commit),
    )
    layer_tables = table.unbind(0)

    def verify():
        return [gdn.tree(q, k, v, g, beta, plan, table=t) for t in layer_tables]

    layout.refresh(initial, width, table, replay_rows)
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = verify()

    for groups in (
        [group_slots([1, 2, 3, 6, 8], 5), group_slots([2, 4, 6, 7, 8], 2)],
        [group_slots([2, 3, 4, 6, 7], 1), group_slots([1, 2, 3, 5, 8], 4)],
    ):
        layout.refresh(groups, width, table, replay_rows)
        graph.replay()
        expected_table, expected_commit = reference(groups)
        assert torch.equal(table, expected_table)
        assert torch.equal(replay_rows, expected_commit)
        for output, expected in zip(actual, expected_table.unbind(0), strict=True):
            expected_output = gdn.tree(q, k, v, g, beta, plan, table=expected)
            torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_resident_refresh_reuses_captured_gdn_pointers():
    from tensorfold.cuda.kernels import gdn

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        StateSlots,
        prepare_resident_recurrent,
    )

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(2076)
    parents = (-1, 0, 0, 2)
    width = len(parents)
    plan = gdn.plan([parents], device)
    q, k, v = [
        torch.randn((width, 2, 128), generator=rng, device=device).mul_(0.1).bfloat16()
        for _ in range(3)
    ]
    g = torch.full((width, 2), 0.9, device=device)
    beta = torch.full_like(g, 0.2)
    cache = torch.randn((9, 2, 128, 128), generator=rng, device=device).mul_(0.1)
    before = cache.clone()

    def slots(ids, count):
        ids = gdn.to_device(ids, torch.int32, device)[None]
        counts = gdn.to_device([count], torch.int32, device)
        return StateSlots(ids[:, 0].long(), counts.long() - 1, [], [], ids, counts)

    table, recurrent = prepare_resident_recurrent(
        [(cache, slots([1, 2, 3, 4, 5], 1))], width, device
    )
    # The runner passes a one-dimensional pointer table for each layer.
    (layer_table,) = table.unbind(0)
    gdn.tree(q, k, v, g, beta, plan, table=layer_table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gdn.tree(q, k, v, g, beta, plan, table=layer_table)

    for ids, count in (([2, 3, 4, 5, 7], 5), ([6, 5, 3, 2, 1], 2)):
        prepare_resident_recurrent(
            [(cache, slots(ids, count))], width, device, out=(table, recurrent)
        )
        graph.replay()
        source = cache[ids[count - 1]]
        reference = gdn.tree(q, k, v, g, beta, plan, state=source)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert table.item() == source.data_ptr()
        assert recurrent[2].item() == source.data_ptr()
        assert recurrent[3:].tolist() == ids[:width]
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires TensorFold CUDA kernels."
)
def test_tensorfold_c1_padding_preserves_fp8_attention_and_gdn_rows():
    from dataclasses import replace

    test_tensorfold_c1_graph_replays_across_prefix_chunk_boundaries(pad_to=16)
    gdn = pytest.importorskip("tensorfold.cuda.kernels.gdn")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(44)
    q, k, v = [
        torch.randn(16, 2, 128, generator=generator, device=device)
        .mul_(0.05)
        .bfloat16()
        for _ in range(3)
    ]
    g = torch.full((16, 2), 0.9, device=device)
    beta = torch.full_like(g, 0.2)
    source = torch.randn(2, 128, 128, device=device, generator=generator).mul_(0.1)
    table = gdn.to_device(gdn.pointers([source]), torch.int64, device)
    initial = (-1, 0, 0, 1, 2, 4, 3)
    padded = replace(gdn.plan([initial + (0,) * 9], device), slots=4)
    gdn.tree(q, k, v, g, beta, padded, table=table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gdn.tree(q, k, v, g, beta, padded, table=table)
    for parents in (initial, (-1, 0, 1, 1, 3), (-1, 0, 0, 2, 3, 2, 5, 6)):
        n = len(parents)
        current = gdn.plan([parents + (0,) * (16 - n)], device)
        assert current.slots <= padded.slots
        padded.entries.copy_(current.entries)
        graph.replay()
        reference = gdn.tree(
            q[:n],
            k[:n],
            v[:n],
            g[:n],
            beta[:n],
            gdn.plan([parents], device),
            table=table,
        )
        torch.testing.assert_close(out[:n], reference, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires TensorFold CUDA kernels."
)
def test_tensorfold_wide_c1_tree_graph_and_gdn_commit():
    test_tensorfold_c1_graph_replays_across_prefix_chunk_boundaries(width=32)
    gdn = pytest.importorskip("tensorfold.cuda.kernels.gdn")
    device = torch.device("cuda")
    random = torch.Generator(device=device).manual_seed(41)
    q, k, v = [
        torch.randn(32, 2, 128, device=device, generator=random).mul_(0.05).bfloat16()
        for _ in range(3)
    ]
    g = torch.full((32, 2), 0.9, device=device)
    beta = torch.full_like(g, 0.2)
    old = torch.randn(2, 128, 128, device=device, generator=random).mul_(0.1)
    state = old.clone()
    parents = (-1, *range(15), 0, *range(16, 30), 1)
    tree = gdn.plan([parents], device)
    source = gdn.to_device(gdn.pointers([state]), torch.int64, device)
    gdn.tree(q, k, v, g, beta, tree, table=source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = gdn.tree(q, k, v, g, beta, tree, table=source)
    graph.replay()
    accepted = torch.tensor([0, *range(16, 31)], device=device, dtype=torch.int32)
    path = accepted.long()
    chain = gdn.plan([tuple(range(-1, 15))], device)
    final = torch.empty_like(old)
    expected = gdn.tree(
        q[path], k[path], v[path], g[path], beta[path], chain, state=old, final=final
    )
    torch.testing.assert_close(result[path], expected, rtol=0, atol=0)
    torch.testing.assert_close(state, old, rtol=0, atol=0)
    pointers = gdn.to_device(
        gdn.replay_table([k], [v], [g], [beta], [[state]]), torch.int64, device
    )
    counts = torch.tensor([path.numel()], device=device, dtype=torch.int32)
    gdn.replay(pointers, 1, 1, accepted[None], counts, k, v, in_place=True)
    torch.testing.assert_close(state, final, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_fused_conv_history_tracks_graph_slots_and_accepted_paths():
    conv = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_conv")
    tree = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan")
    device = torch.device("cuda")
    random = torch.Generator(device=device).manual_seed(81)
    layers, slots, channels, width, keep = 3, 5, 259, 10, 3
    parents = (-1, 0, 0, 1, 2, 3, 4)
    plan = tree.TreeBatchPlan.build([tree.TreeWindow(tuple(range(7)), parents)])

    def rand(shape):
        return torch.randn(shape, device=device, generator=random).bfloat16()

    for dim_first in (False, True):
        shape = (slots, channels, width) if dim_first else (slots, width, channels)
        caches = [rand(shape) for _ in range(layers)]
        if not dim_first:
            caches = [cache.transpose(-1, -2) for cache in caches]
        updates = [rand((7, channels)) for _ in caches]
        batch = conv.ConvHistoryBatch(caches, keep)
        slot_in = torch.tensor([1], dtype=torch.int64, device=device)
        offset_in = torch.tensor([1], dtype=torch.int64, device=device)
        batch.gather(slot_in, offset_in)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prefix = batch.gather(slot_in, offset_in)
        for slot, offset, path in ((1, 1, [0]), (4, 2, [0, 2, 4, 6]), (1, 0, [0, 1])):
            slot_in.fill_(slot)
            offset_in.fill_(offset)
            graph.replay()
            reference = torch.stack(
                [cache[slot, :, offset : offset + keep].T for cache in caches]
            )
            torch.testing.assert_close(prefix, reference, rtol=0, atol=0)
            picks = plan.convolution_commit_indices([path], keep)
            expected = [cache.clone() for cache in caches]
            write_offset = len(path) - 1
            for i, (cache, update) in enumerate(zip(expected, updates, strict=True)):
                tail = torch.cat((reference[i], update))[picks]
                cache[slot, :, write_offset : write_offset + keep] = tail.T
            batch.commit(
                list(zip(caches, prefix.unbind(0), updates, strict=True)),
                picks,
                slot_in,
                write_offset,
            )
            for cache, ref in zip(caches, expected, strict=True):
                torch.testing.assert_close(cache, ref, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_fused_conv_preserves_distinct_gdn_slot_groups():
    verify = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify"
    )
    device = torch.device("cuda")
    shape, keep = (4, 65, 6), 3
    caches = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)]
    slot_groups = [
        verify.StateSlots(
            torch.tensor([slot], dtype=torch.int64, device=device),
            torch.tensor([offset], dtype=torch.int64, device=device),
            [slot],
            [[slot]],
        )
        for slot, offset in ((1, 0), (2, 1))
    ]
    groups = verify.group_conv_histories(
        {
            "first": (caches[0], slot_groups[0], keep),
            "second": (caches[1], slot_groups[1], keep),
            "third": (caches[2], slot_groups[0], keep),
        }
    )
    assert [group.names for group in groups] == [("first", "third"), ("second",)]
    for group in groups:
        group.batch.gather(group.slots.conv, group.slots.offset)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefixes = [
            group.batch.gather(group.slots.conv, group.slots.offset) for group in groups
        ]
    name_to_cache = dict(zip(("first", "second", "third"), caches, strict=True))
    slots = (3, 1)
    offsets = (2, 0)
    for state, slot, offset in zip(slot_groups, slots, offsets, strict=True):
        state.conv.fill_(slot)
        state.offset.fill_(offset)
    graph.replay()
    for group, prefix, slot, offset in zip(
        groups, prefixes, slots, offsets, strict=True
    ):
        group_caches = [name_to_cache[name] for name in group.names]
        reference = torch.stack(
            [cache[slot, :, offset : offset + keep].T for cache in group_caches]
        )
        torch.testing.assert_close(prefix, reference, rtol=0, atol=0)
        updates = [
            cache.new_full((1, shape[1]), i + 1) for i, cache in enumerate(group_caches)
        ]
        expected = [cache.clone() for cache in group_caches]
        for dst, hist, update in zip(expected, reference, updates, strict=True):
            dst[slot, :, :keep] = torch.cat((hist[1:], update)).T
        group.batch.commit(
            list(zip(group_caches, prefix.unbind(0), updates, strict=True)),
            [1, 2, 3],
            group.slots.conv,
            0,
        )
        for actual, wanted in zip(group_caches, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA Triton.")
def test_tensorfold_bf16_pages_commit_and_replay_without_quantization():
    ops = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention"
    )
    page, heads, d = 16, 2, 128
    slots = torch.tensor([15, 16, 32], device="cuda")
    accepted = torch.tensor([0, 2, 4], device="cuda")
    key = torch.arange(1, 6, device="cuda", dtype=torch.bfloat16)[:, None, None]
    key = key.expand(-1, heads, d).contiguous()
    value = key * 2
    scales = torch.tensor([0.5, 2.0], device="cuda")
    for slot_major in (False, True):
        shape = (3, page, heads, 2 * d) if slot_major else (3, heads, page, 2 * d)
        cache = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        if slot_major:
            cache = cache.transpose(1, 2)
        ops.commit_paged_kv(key, value, accepted, slots, cache, *scales)
        expected = torch.zeros(
            (3, page, heads, 2 * d), device="cuda", dtype=cache.dtype
        )
        expected[slots // page, slots % page] = torch.cat(
            (key[accepted], value[accepted]), dim=-1
        )
        torch.testing.assert_close(cache.transpose(1, 2), expected, rtol=0, atol=0)
    test_tensorfold_paged_attention_graph_updates_pages_and_sibling_paths(
        torch.bfloat16
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(121),
    reason="The TensorFold round targets GB10.",
)
def test_tensorfold_tree_and_batch_replay_match_serial_accepted_paths():
    from dataclasses import replace

    gdn = pytest.importorskip("tensorfold.cuda.kernels.gdn")
    verify = pytest.importorskip(
        "vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify"
    )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    q, k, v = [
        torch.randn(7, 2, 128, device=device, generator=generator).mul_(0.05).bfloat16()
        for _ in range(3)
    ]
    g = torch.full((7, 2), 0.9, device=device)
    beta = torch.full_like(g, 0.2)
    prefix = [
        torch.randn(2, 128, 128, device=device, generator=generator).mul_(0.1)
        for _ in range(2)
    ]
    states = [s.clone() for s in prefix]
    table = gdn.to_device(gdn.pointers(states), torch.int64, device)
    plan = gdn.plan([(-1, 0, 0, 1, 2), (-1, 0)], device)
    actual = gdn.tree(q, k, v, g, beta, plan, table=table)
    for a, b in zip(states, prefix, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    paths = [[0, 2, 4], [5, 6]]
    expected = [torch.empty_like(state) for state in prefix]
    for path, initial, final in zip(paths, prefix, expected, strict=True):
        idx = torch.tensor(path, device=device)
        chain = gdn.plan([list(range(-1, len(path) - 1))], device)
        outputs = gdn.tree(
            q[idx],
            k[idx],
            v[idx],
            g[idx],
            beta[idx],
            chain,
            state=initial,
            final=final,
        )
        torch.testing.assert_close(actual[idx], outputs, rtol=0, atol=0)

    table = gdn.to_device(
        gdn.replay_table([k], [v], [g], [beta], [[s] for s in states]),
        torch.int64,
        device,
    )
    rows = torch.tensor([[0, 2, 4], [5, 6, 0]], device=device, dtype=torch.int32)
    counts = torch.tensor([3, 2], device=device, dtype=torch.int32)
    gdn.replay(table, 1, 2, rows, counts, k, v, in_place=True)
    for a, b in zip(states, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    pointers = gdn.to_device(gdn.pointers(prefix), torch.int64, device)
    gdn.tree(q, k, v, g, beta, plan, table=pointers)
    assert plan.slots == 1
    bucket = replace(plan, slots=verify.gdn_slot_bucket(plan.slots))
    assert bucket.slots == 2
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gdn.tree(q, k, v, g, beta, bucket, table=pointers)
    next_prefix = [s * 0.75 for s in reversed(prefix)]
    pointers.copy_(gdn.to_device(gdn.pointers(next_prefix), torch.int64, device))
    graph.replay()
    torch.testing.assert_close(
        captured, gdn.tree(q, k, v, g, beta, plan, table=pointers), rtol=0, atol=0
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_c1_draft_graph_pinned_readback_changes_each_replay():
    import numpy as np

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_draft_graph import (
        C1DraftGraph,
    )

    class Draft:
        block = 16
        device = torch.device("cuda")
        layers = 1
        kv_local = 1
        head_dim = 16
        mask_id = 5

        def launch_c1_device(self, tokens, positions, tables, lengths):
            ids = (tokens[0].long() + 3 * positions[1:].long()).view(-1, 1)
            floats = torch.stack((positions[1:], lengths[0].expand(15)), dim=1)
            return ids, floats

    draft = Draft()
    ordinary = C1DraftGraph(draft, 15)
    pinned = C1DraftGraph(draft, 15, pinned_readback=True)
    graph = None
    first = None
    first_ids = None
    for length, pos, pending in ((5, 10, 17), (7, 13, 29), (6, 19, 45)):
        kv = torch.zeros(1, length, 16, device=draft.device, dtype=torch.bfloat16)
        snap = ([kv], [kv], length, pos)
        reference = ordinary.launch(snap, pending, 15)
        actual = pinned.launch(snap, pending, 15)
        assert actual is not None and reference is not None
        assert actual[1:] == reference[1:]
        ids, floats, host = actual[0]
        assert all(tensor.is_pinned() for tensor in pinned.host_outputs)
        assert host is not None and reference[0][2] is None
        np.testing.assert_array_equal(host[0], reference[0][0].cpu().numpy())
        np.testing.assert_array_equal(host[1], reference[0][1].cpu().numpy())
        assert host[0].dtype == np.int64 and host[1].dtype == np.float64
        assert ids.is_cuda and floats.is_cuda
        if graph is None:
            graph = pinned.graph
            assert graph is not None
            first, first_ids = host[0], host[0].copy()
        else:
            assert graph is pinned.graph
            np.testing.assert_array_equal(first, first_ids)
            assert not np.array_equal(host[0], first)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
def test_tensorfold_batched_device_commit_matches_reference():
    from tensorfold.cuda.kernels import gdn

    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import (
        commit_paged_kv,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
        BatchCommitGraph,
        KVCommitLayer,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
        TreeBatchPlan,
        TreeWindow,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        GDNRecord,
        StateSlots,
    )

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(211)
    short = TreeWindow((100, 101, 102), (-1, 0, 0))
    long = TreeWindow((100, 101, 102, 103, 104), (-1, 0, 0, 1, 2))

    def rand(*shape):
        return torch.randn(shape, generator=rng, device=device).mul_(0.1)

    def ints(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    for streams, capture in ((2, False), (2, True), (4, True), (6, True), (8, True)):
        rows = 4 * streams
        nslots = 2 + 16 * streams
        records = []
        for i, kept in enumerate((3, 2)):
            conv = rand(nslots, 40, 5 + kept).bfloat16()
            if i:
                conv = rand(nslots, 5 + kept, 40).bfloat16().transpose(1, 2)
            records.append(
                GDNRecord(
                    rand(rows, 2, 128).bfloat16(),
                    rand(rows, 2, 128).bfloat16(),
                    torch.full((rows, 2), 0.9, device=device),
                    torch.full((rows, 2), 0.2, device=device),
                    rand(rows, 40).bfloat16(),
                    conv,
                    rand(streams, kept, 40).bfloat16(),
                    rand(nslots, 2, 128, 128),
                    StateSlots(ints([0] * streams), ints([0] * streams), [], []),
                    f"gdn-{i}",
                )
            )
        mapping = ints(range(15, 15 + rows))
        scale_k, scale_v = (torch.tensor(x, device=device) for x in (0.5, 2.0))
        kv = {
            name: KVCommitLayer(
                rand(rows, 2, 128).bfloat16(),
                rand(rows, 2, 128).bfloat16(),
                rand(12, 2, 16, 256).to(dtype),
                scale_k,
                scale_v,
                mapping,
            )
            for name, dtype in (("fp8", torch.float8_e4m3fn), ("bf16", torch.bfloat16))
        }
        commit = BatchCommitGraph(records, kv, streams, rows, capture=capture)
        reference_gdn = [r.recurrent.clone() for r in records]
        reference_conv = [r.conv.clone() for r in records]
        reference_kv = {name: layer.cache.clone() for name, layer in kv.items()}
        for turn in range(3):
            windows = [short if (i + turn) % 2 == 0 else long for i in range(streams)]
            plan = TreeBatchPlan.build(windows)
            predicted = [
                [999] * len(tree.tokens)
                if (i + turn) % 3 == 0
                else ([102, 987, 988] if tree is short else [102, 900, 104, 901, 902])
                for i, tree in enumerate(windows)
            ]
            selected = [
                tree.select(p) for tree, p in zip(windows, predicted, strict=True)
            ]
            paths, emitted = zip(*selected, strict=True)
            sources, destinations = plan.commit_rows(list(paths))
            src_idx, dst_idx = ints(sources), ints(destinations)
            counts = [len(path) for path in paths]
            pages = mapping.roll(turn) + 2 * turn
            for name, layer in kv.items():
                commit_paged_kv(
                    layer.key,
                    layer.value,
                    src_idx,
                    pages.index_select(0, dst_idx),
                    reference_kv[name],
                    scale_k,
                    scale_v,
                )

            states = [[] for _ in range(streams)]
            for layer, (record, recurrent, conv) in enumerate(
                zip(records, reference_gdn, reference_conv, strict=True)
            ):
                # Change cache slots and convolution inputs under the same graph.
                record.qkv.add_(0.0078125)
                record.prefix.add_(0.015625)
                first = [1 + 16 * i + 6 * (turn % 2) + layer for i in range(streams)]
                slots = [list(range(n, n + 5)) for n in first]
                old = [row[-1] for row in slots]
                record.slots.conv.copy_(ints(first))
                record.slots.source = old
                record.slots.all_slots = slots
                keep = record.prefix.shape[1]
                combined = torch.cat((record.prefix.flatten(0, 1), record.qkv))
                indices = ints(plan.convolution_commit_indices(list(paths), keep))
                tail = combined.index_select(0, indices).view(streams, keep, -1)
                for i, count in enumerate(counts):
                    dest = recurrent[slots[i][count - 1]]
                    dest.copy_(recurrent[old[i]])
                    states[i].append(dest)
                    conv[first[i], :, count - 1 : count - 1 + keep] = tail[i].T
            replay_table = gdn.replay_table(
                *[
                    [getattr(r, part) for r in records]
                    for part in ("k", "v", "g", "beta")
                ],
                states,
            )
            replay_rows = [
                [start + row for row in path] + [start] * (max(counts) - len(path))
                for start, path in zip(plan.starts[:-1], paths, strict=True)
            ]
            gdn.replay(
                ints(replay_table),
                len(records),
                streams,
                ints(replay_rows).int(),
                ints(counts).int(),
                records[0].k,
                records[0].v,
                in_place=True,
            )
            got = commit.accept_and_commit(
                windows,
                plan,
                ints([val for pred in predicted for val in pred]),
                dict.fromkeys(kv, pages),
            )
            assert got is not None
            assert got.replayed == (capture and turn > 0)
            assert got.paths == list(paths) and got.emitted == list(emitted)
            assert (
                got.source.tolist() == sources
                and got.destination.tolist() == destinations
            )
            assert got.counts.tolist() == counts
            assert got.sampled_ids.tolist() == [
                ids + [-1] * (max(counts) - len(ids)) for ids in emitted
            ]
            for r, state, conv in zip(
                records, reference_gdn, reference_conv, strict=True
            ):
                assert torch.equal(r.recurrent, state)
                assert torch.equal(r.conv, conv)
            for name, layer in kv.items():
                dtype = torch.uint8 if name == "fp8" else torch.bfloat16
                assert torch.equal(
                    layer.cache.view(dtype), reference_kv[name].view(dtype)
                )
