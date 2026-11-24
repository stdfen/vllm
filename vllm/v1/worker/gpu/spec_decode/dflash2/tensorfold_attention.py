# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorFold 0.6.0 tree attention with vLLM page addressing."""

import torch
from tensorfold.cuda.kernels.attention import (
    CHUNK,
    MAX_NODES,
    MERGE_COLUMNS,
    QUERY_TILE,
    Plan,
    _merge,
    _tile,
    from_packed,
    padded_host,
    plan_host,
)

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)

C1_GRAPH_PREFIX_CAPACITY = 4096


def c1_graph_attention_plan(
    parents: tuple[int, ...], length: int, group: int, device: torch.device
) -> Plan:
    """Keep launch dimensions fixed and pass the exact prefix on the device."""
    width = len(parents)
    flat, items, chunks = padded_host(parents, C1_GRAPH_PREFIX_CAPACITY, group)
    flat[width + 2 : width + 4] = [length, cdiv(length + width, CHUNK)]
    packed = torch.tensor(flat, dtype=torch.int32).pin_memory()
    return from_packed(packed.to(device, non_blocking=True), 1, width, items, chunks)


def batch_graph_attention_plan(
    parents: list[tuple[int, ...]],
    lengths: list[int],
    group: int,
    chunks: int,
    device: torch.device,
) -> Plan:
    flat, n_items, needed = plan_host(parents, lengths, group)
    assert needed <= chunks
    widths = [len(tree) for tree in parents]
    rows = sum(widths)
    item_capacity = (chunks - 1) * sum(cdiv(w * group, QUERY_TILE) for w in widths)
    parents_offset = rows + 4 * len(parents) + 3 * n_items
    # The last chunk of this bucket is never a full committed chunk.
    flat[parents_offset:parents_offset] = [0, 0, chunks - 1] * (item_capacity - n_items)
    packed = torch.tensor(flat, dtype=torch.int32).pin_memory()
    return from_packed(
        packed.to(device, non_blocking=True), len(parents), rows, item_capacity, chunks
    )


@triton.jit
def _prefix(
    CACHE,
    TABLE,
    KS,
    VS,
    stream,
    head,
    rows,
    dims,
    active,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    CS: tl.constexpr,
    D: tl.constexpr,
):
    block = tl.load(TABLE + stream * TS + rows // PAGE, active, 0).to(tl.int64)
    offset = (block * CS[0] + head * CS[1] + (rows % PAGE) * CS[2])[:, None] + dims[
        None, :
    ] * CS[3]
    if CACHE.dtype.element_ty == tl.bfloat16:
        k = tl.load(CACHE + offset, active[:, None], 0)
        v = tl.load(CACHE + offset + D * CS[3], active[:, None], 0)
    else:
        packed = CACHE.to(tl.pointer_type(tl.uint8))
        kb = tl.load(packed + offset, active[:, None], 0)
        vb = tl.load(packed + offset + D * CS[3], active[:, None], 0)
        k = kb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(KS)
        v = vb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(VS)
    return k.to(tl.bfloat16), v.to(tl.bfloat16)


@triton.jit
def _round_to_fp8_cache(x, scale):
    normalized = tl.clamp(x.to(tl.float32) / scale, -448.0, 448.0)
    packed = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    return (packed.to(tl.float32) * scale).to(tl.bfloat16)


# Adapt TensorFold's _shared and _tail loaders to the vLLM page table. Use
# its original tile update, tree path construction, and ordered chunk merge.
@triton.jit
def _shared_paged(
    Q,
    CACHE,
    TABLE,
    KS,
    VS,
    STREAM,
    ITEMS,
    PO,
    PM,
    PL,
    W,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    CH: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    CS: tl.constexpr,
):
    item = tl.program_id(0) // HK
    hk = tl.program_id(0) % HK
    stream = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    chunk = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + stream * 4)
    rows = tl.load(STREAM + stream * 4 + 1)
    prefix = tl.load(STREAM + stream * 4 + 2)
    if (chunk + 1) * CH <= prefix:
        rr = first + tl.arange(0, 16)
        valid = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        dims = tl.arange(0, D)
        key = chunk * CH + tl.arange(0, 64)
        q = tl.load(
            Q + (node[:, None] * H + head[:, None]) * D + dims[None, :],
            valid[:, None],
            0,
        )
        m = tl.full((16,), float("-inf"), tl.float32)
        denom = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        for t in range(CH // 64):
            ki = key + t * 64
            k, v = _prefix(
                CACHE, TABLE, KS, VS, stream, hk, ki, dims, ki < prefix, PAGE, TS, CS, D
            )
            m, denom, o = _tile(q, k, v, m, denom, o, ki < prefix, SCALE)
        base = (chunk * W + node) * H + head
        tl.store(PO + base[:, None] * D + dims[None, :], o, valid[:, None])
        tl.store(PM + base, m, valid)
        tl.store(PL + base, denom, valid)


@triton.jit
def _tail_paged(
    Q,
    KN,
    VN,
    CACHE,
    TABLE,
    KS,
    VS,
    STREAM,
    ROWS,
    PATHS,
    DEPTHS,
    PO,
    PM,
    PL,
    W,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    CH: tl.constexpr,
    MAXD: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    CS: tl.constexpr,
    ROUND_FP8_TAIL: tl.constexpr,
):
    node = tl.program_id(0)
    hk = tl.program_id(1)
    stream = tl.load(ROWS + node)
    prefix = tl.load(STREAM + stream * 4 + 2)
    chunks = tl.load(STREAM + stream * 4 + 3)
    chunk = prefix // CH + tl.program_id(2)
    if chunk < chunks:
        group = tl.arange(0, 16)
        dims = tl.arange(0, D)
        q = tl.load(
            Q + (node * H + hk * G + group[:, None]) * D + dims[None, :],
            group[:, None] < G,
            0,
        )
        depth = tl.load(DEPTHS + node)
        m = tl.full((16,), float("-inf"), tl.float32)
        denom = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        key = chunk * CH + tl.arange(0, 64)
        for t in range(CH // 64):
            logical = key + t * 64
            committed = logical < prefix
            path_slot = logical - prefix
            on_path = (path_slot >= 0) & (path_slot < depth)
            path_node = tl.load(PATHS + node * MAXD + path_slot, on_path, 0)
            kc, vc = _prefix(
                CACHE,
                TABLE,
                KS,
                VS,
                stream,
                hk,
                logical,
                dims,
                committed,
                PAGE,
                TS,
                CS,
                D,
            )
            offsets = (path_node[:, None] * HK + hk) * D + dims[None, :]
            kn = tl.load(KN + offsets, on_path[:, None], 0)
            vn = tl.load(VN + offsets, on_path[:, None], 0)
            if ROUND_FP8_TAIL:
                kn = _round_to_fp8_cache(kn, tl.load(KS))
                vn = _round_to_fp8_cache(vn, tl.load(VS))
            k = tl.where(committed[:, None], kc, kn)
            v = tl.where(committed[:, None], vc, vn)
            m, denom, o = _tile(q, k, v, m, denom, o, committed | on_path, SCALE)
        base = (chunk * W + node) * H + hk * G + group
        tl.store(PO + base[:, None] * D + dims[None, :], o, group[:, None] < G)
        tl.store(PM + base, m, group < G)
        tl.store(PL + base, denom, group < G)


def partials(query: torch.Tensor, plan: Plan) -> tuple[torch.Tensor, ...]:
    w, h, d = query.shape
    partial_o = torch.empty(
        (plan.chunks, w, h, d), dtype=torch.float32, device=query.device
    )
    partial_m = torch.empty_like(
        partial_o[..., 0], memory_format=torch.contiguous_format
    )
    return partial_o, partial_m, torch.empty_like(partial_m)


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    plan: Plan,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    scale: float,
    output: torch.Tensor,
    workspace: tuple[torch.Tensor, ...],
    *,
    round_fp8_tail: bool = False,
) -> None:
    w, h, d = query.shape
    hk = key.shape[1]
    assert query.dtype == key.dtype == value.dtype == output.dtype == torch.bfloat16
    assert cache.dtype in (torch.float8_e4m3fn, torch.bfloat16)
    assert output.is_contiguous() and block_table.stride(1) == 1
    assert cache.shape[1] == hk and cache.shape[-1] == 2 * d
    assert w == plan.width and d in (128, 256) and h % hk == 0 and h // hk <= 16
    g = h // hk
    args = (h, hk, d, g, CHUNK)
    layout = (cache.shape[2], block_table.stride(0), cache.stride())
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    if plan.items.shape[0]:
        _shared_paged[(plan.items.shape[0] * hk,)](
            q,
            cache,
            block_table,
            key_scale,
            value_scale,
            plan.streams,
            plan.items,
            *workspace,
            w,
            *args,
            scale,
            *layout,
            num_warps=4,
            num_stages=1,
        )
    _tail_paged[(w, hk, 2)](
        q,
        k,
        v,
        cache,
        block_table,
        key_scale,
        value_scale,
        plan.streams,
        plan.rows,
        plan.paths,
        plan.depths,
        *workspace,
        w,
        *args,
        MAX_NODES,
        scale,
        *layout,
        round_fp8_tail and cache.dtype == torch.float8_e4m3fn,
        num_warps=4,
        num_stages=1,
    )
    _merge[(w, hk, d // MERGE_COLUMNS)](
        *workspace,
        output,
        plan.streams,
        plan.rows,
        w,
        h,
        d,
        g,
        MERGE_COLUMNS,
        num_warps=4,
    )


def commit_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    accepted: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> None:
    """Gather accepted rows and store them in the selected cache dtype."""
    k_cache, v_cache = cache.transpose(1, 2).chunk(2, dim=-1)
    triton_reshape_and_cache_flash(
        key.index_select(0, accepted),
        value.index_select(0, accepted),
        k_cache,
        v_cache,
        slots,
        "bfloat16" if cache.dtype == torch.bfloat16 else "fp8",
        key_scale,
        value_scale,
    )
