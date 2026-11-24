# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorFold target tree verification and accepted paths on vLLM pages."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from tensorfold.cuda.kernels import attention, gdn
from tensorfold.families.qwen3_5.cuda.glue import gated_norm, gdn_pre

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import (
    C1_GRAPH_PREFIX_CAPACITY,
    batch_graph_attention_plan,
    c1_graph_attention_plan,
    commit_paged_kv,
    paged_attention,
    partials,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_conv import ConvHistoryBatch
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
    TreeBatchPlan,
    TreeWindow,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
        BatchAcceptance,
        BatchCommitGraph,
        C1CommitGraph,
        KVCommitLayer,
    )

logger = init_logger(__name__)


def gdn_slot_bucket(slots: int) -> int:
    # TensorFold compiles the same tree kernel throughout each slot bucket.
    return max(2, 1 << (slots - 1).bit_length()) if slots else 0


def _paged_cache(layer: "Attention") -> torch.Tensor:
    cache = layer.kv_cache
    return cache if cache.dtype == torch.bfloat16 else cache.view(layer.impl.fp8_dtype)


@dataclass
class StateSlots:
    conv: torch.Tensor
    offset: torch.Tensor
    source: list[int]
    all_slots: list[list[int]]
    device_ids: torch.Tensor | None = None
    device_counts: torch.Tensor | None = None

    @classmethod
    def from_host(cls, meta: GDNAttentionMetadata, packed: list[int]) -> "StateSlots":
        slots, accepted = meta.spec_state_indices_tensor, meta.num_accepted_tokens
        assert slots is not None and accepted is not None
        state = cls(
            slots[:, 0].to(dtype=torch.int64, copy=True), accepted.long() - 1, [], []
        )
        state._assign_host(slots, packed)
        return state

    def _assign_host(self, slots: torch.Tensor, packed: list[int]) -> None:
        width = slots.shape[1] + 1
        rows = [packed[i : i + width - 1] for i in range(0, len(packed), width)]
        counts = packed[width - 1 :: width]
        if (
            any(not 1 <= n <= len(row) for row, n in zip(rows, counts, strict=True))
            or any(slot <= 0 for row in rows for slot in row)
            or len({slot for row in rows for slot in row}) != slots.numel()
        ):
            raise RuntimeError("Invalid or overlapping vLLM GDN state slots.")
        self.source = [row[n - 1] for row, n in zip(rows, counts, strict=True)]
        self.all_slots = rows

    def materialize_host(self) -> None:
        """Supply the legacy commit inputs only if the captured path was ineligible."""
        if self.device_ids is not None and not self.all_slots:
            assert self.device_counts is not None
            packed = (
                torch.cat((self.device_ids, self.device_counts[:, None]), dim=1)
                .flatten()
                .tolist()
            )
            self._assign_host(self.device_ids, packed)


def device_verify_slots(
    metadata: dict[str, GDNAttentionMetadata | TritonAttentionMetadata],
    verification_rows: int,
) -> dict[int, StateSlots]:
    groups = {
        id(meta): meta
        for meta in metadata.values()
        if isinstance(meta, GDNAttentionMetadata)
    }
    result = {}
    for key, meta in groups.items():
        ids, counts = meta.spec_state_indices_tensor, meta.num_accepted_tokens
        if (
            ids is None
            or counts is None
            or ids.dtype != torch.int32
            or counts.dtype != torch.int32
            or ids.shape[0] != 1
            or counts.numel() != 1
            or ids.shape[1] < verification_rows
            or ids.stride(1) != 1
        ):
            return {}
        result[key] = StateSlots(
            ids[:, 0].long(), counts.long() - 1, [], [], ids, counts
        )
    return result


@triton.jit
def _prepare_resident_recurrent(
    LAYOUT,
    TABLE,
    RECURRENT,
    GROUP_IDS,
    GROUP_COUNTS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPED: tl.constexpr,
):
    layer = tl.program_id(0)
    params = LAYOUT + (3 if GROUPED else 4) * layer
    base = tl.load(params)
    stride = tl.load(params + 1)
    if GROUPED:
        slot_group = tl.load(params + 2)
        ids_addr = tl.full((), 0, tl.int64)
        counts_addr = tl.full((), 0, tl.int64)
        for group in tl.static_range(len(GROUP_IDS)):
            match = slot_group == group
            ids_addr = tl.where(match, GROUP_IDS[group].to(tl.int64), ids_addr)
            counts_addr = tl.where(match, GROUP_COUNTS[group].to(tl.int64), counts_addr)
    else:
        ids_addr = tl.load(params + 2)
        counts_addr = tl.load(params + 3)
    ids = ids_addr.to(tl.pointer_type(tl.int32))
    counts = counts_addr.to(tl.pointer_type(tl.int32))
    source = base + stride * tl.load(ids + tl.load(counts) - 1).to(tl.int64)
    tl.store(TABLE + layer, source)
    output = RECURRENT + (WIDTH + 3) * layer
    tl.store(output, base)
    tl.store(output + 1, stride)
    tl.store(output + 2, source)
    i = tl.arange(0, BLOCK)
    pending = tl.load(ids + i, i < WIDTH, other=0).to(tl.int64)
    tl.store(output + 3 + i, pending, i < WIDTH)


def prepare_resident_recurrent(
    layers: list[tuple[torch.Tensor, StateSlots]],
    width: int,
    device: torch.device,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = []
    for state, slots in layers:
        ids, counts = slots.device_ids, slots.device_counts
        assert ids is not None and counts is not None
        params.extend(
            (
                state.data_ptr(),
                state.stride(0) * state.element_size(),
                ids.data_ptr(),
                counts.data_ptr(),
            )
        )
    layout = gdn.to_device(params, torch.int64, device)
    if out is None:
        table = torch.empty((len(layers), 1), dtype=torch.int64, device=device)
        recurrent = table.new_empty(len(layers) * (width + 3))
    else:
        table, recurrent = out
    _prepare_resident_recurrent[(len(layers),)](
        layout,
        table,
        recurrent,
        (),
        (),
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        GROUPED=False,
        num_warps=1,
    )
    return table, recurrent


class C1ResidentRecurrentLayout:
    """Per-layer cache descriptors for a captured C1 verification graph."""

    def __init__(self, layers: list[tuple[torch.Tensor, int]]) -> None:
        params = [
            value
            for state, group in layers
            for value in (
                state.data_ptr(),
                state.stride(0) * state.element_size(),
                group,
            )
        ]
        self.layers = len(layers)
        self.layout = gdn.to_device(params, torch.int64, layers[0][0].device)

    def refresh(
        self,
        slots: list[StateSlots],
        width: int,
        table: torch.Tensor,
        recurrent: torch.Tensor,
    ) -> None:
        ids = tuple(group.device_ids for group in slots)
        counts = tuple(group.device_counts for group in slots)
        _prepare_resident_recurrent[(self.layers,)](
            self.layout,
            table,
            recurrent,
            ids,
            counts,
            WIDTH=width,
            BLOCK=triton.next_power_of_2(width),
            GROUPED=True,
            num_warps=1,
        )


def read_verify_metadata(
    seq_lens: torch.Tensor,
    metadata: dict[str, GDNAttentionMetadata | TritonAttentionMetadata],
) -> tuple[list[int], dict[int, StateSlots]]:
    """Read the exact lengths and each shared GDN slot group together."""
    groups = {
        id(meta): meta
        for meta in metadata.values()
        if isinstance(meta, GDNAttentionMetadata)
    }
    parts = [seq_lens]
    for meta in groups.values():
        slots, accepted = meta.spec_state_indices_tensor, meta.num_accepted_tokens
        assert slots is not None and accepted is not None
        parts.append(torch.cat((slots, accepted.reshape(-1, 1)), dim=1).view(-1))
    host = torch.cat(parts).tolist()
    slot_groups = {}
    offset = seq_lens.numel()
    for key, part in zip(groups, parts[1:], strict=True):
        stop = offset + part.numel()
        slot_groups[key] = StateSlots.from_host(groups[key], host[offset:stop])
        offset = stop
    return host[: seq_lens.numel()], slot_groups


@dataclass
class ConvGroup:
    batch: ConvHistoryBatch
    slots: StateSlots
    names: tuple[str, ...]


def group_conv_histories(
    layers: dict[str, tuple[torch.Tensor, StateSlots, int]],
) -> list[ConvGroup]:
    by_layout: dict[tuple, list[str]] = {}
    for name, (cache, slots, keep) in layers.items():
        if cache.dtype != torch.bfloat16:
            return []
        shape = (id(slots), cache.shape, cache.stride(), keep)
        by_layout.setdefault(shape, []).append(name)
    groups = []
    for names in by_layout.values():
        _, slots, keep = layers[names[0]]
        caches = [layers[name][0] for name in names]
        groups.append(ConvGroup(ConvHistoryBatch(caches, keep), slots, tuple(names)))
    return groups


@dataclass
class GDNRecord:
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    qkv: torch.Tensor
    conv: torch.Tensor
    prefix: torch.Tensor
    recurrent: torch.Tensor
    slots: StateSlots
    name: str


class C1GraphPlanBuffers:
    """Pack changing C1 tree inputs into the CUDA Graph's resident buffers."""

    def __init__(
        self,
        gdn_plan: gdn.Plan,
        attn_plan: attention.Plan,
        conv_windows: dict[int, torch.Tensor],
    ):
        inputs = (
            gdn_plan.entries,
            attn_plan.streams,
            attn_plan.parents,
            *conv_windows.values(),
            torch.zeros_like(attn_plan.parents),
        )
        self.packed = torch.cat([tensor.flatten() for tensor in inputs])
        views = [
            flat.view_as(original)
            for flat, original in zip(
                self.packed.split([tensor.numel() for tensor in inputs]),
                inputs,
                strict=True,
            )
        ]
        entries, streams, parents, *windows, depths = views
        self.position_depths = depths
        self.gdn_plan = replace(gdn_plan, entries=entries)
        self.attn_plan = replace(attn_plan, streams=streams, parents=parents)
        self.conv_windows = dict(zip(conv_windows, windows, strict=True))

    def update(
        self,
        window: TreeWindow,
        plan: TreeBatchPlan,
        length: int,
        entries: list[int],
    ) -> None:
        width = len(window.parents)
        conv = [
            index
            for kept in self.conv_windows
            for row in plan.convolution_windows([window], kept)
            for index in row
        ]
        values = [
            *entries,
            0,
            width,
            length,
            cdiv(length + width, attention.CHUNK),
            *window.parents,
            *conv,
            *plan.depths,
        ]
        self.packed.copy_(
            torch.tensor(values, dtype=torch.int32).pin_memory(), non_blocking=True
        )
        attn = self.attn_plan
        attention._paths[(width,)](
            attn.parents, attn.paths, attn.depths, MAXD=attention.MAX_NODES, num_warps=1
        )


def _aliases(values: dict) -> tuple[tuple[str, ...], ...]:
    by_id: dict[int, list[str]] = {}
    for name, value in values.items():
        by_id.setdefault(id(value), []).append(name)
    return tuple(tuple(names) for names in by_id.values())


def eager_recurrent_table(
    records: list[GDNRecord],
    accepted: list[int],
    *,
    direct: bool,
) -> list[int]:
    """Stage pointers for ordinary GDN replay, with optional destinations."""
    destinations: list[torch.Tensor] = []
    if direct:
        (count,) = accepted
        states = [[r.recurrent[r.slots.source[0]] for r in records]]
        destinations = [
            r.recurrent[r.slots.all_slots[0][count - 1]] for r in records
        ]
    else:
        states = [[] for _ in accepted]
        old_states, new_states = [], []
        for record in records:
            for i, n in enumerate(accepted):
                old = record.slots.source[i]
                slot = record.slots.all_slots[i][n - 1]
                dest = record.recurrent[slot]
                states[i].append(dest)
                if old != slot:
                    old_states.append(record.recurrent[old])
                    new_states.append(dest)
        if new_states:
            torch._foreach_copy_(new_states, old_states)

    pointers = gdn.replay_table(
        [r.k for r in records],
        [r.v for r in records],
        [r.g for r in records],
        [r.beta for r in records],
        states,
    )
    if direct:
        pointers.extend(gdn.pointers(destinations))
    return pointers


class TensorFoldVerify:
    """Provisional model outputs and stable device inputs for one tree shape."""

    def __init__(
        self,
        windows: list[TreeWindow],
        plan: TreeBatchPlan,
        lengths: list[int],
        metadata: dict[str, GDNAttentionMetadata | TritonAttentionMetadata],
        layers: dict,
        device: torch.device,
        slot_groups: dict[int, StateSlots],
        c1_graph_bucket: bool = False,
        fused_conv: bool = False,
        c1_fixed_width: bool = False,
        round_fp8_tail: bool = False,
        c8_graph_chunks: int | None = None,
        batch_graph_chunks: int | None = None,
    ):
        self.device = device
        self.plan = plan
        self.layers = layers
        self.metadata = metadata
        self.round_fp8_tail = round_fp8_tail
        parents = [w.parents for w in windows]
        self.is_c1_graph_bucket = (
            c1_graph_bucket
            and len(windows) == 1
            and 0 <= lengths[0] <= C1_GRAPH_PREFIX_CAPACITY
        )
        self.gdn_plan = gdn.plan(parents, device)
        multi_request_chunks = c8_graph_chunks or batch_graph_chunks
        if self.gdn_plan.slots > 4:
            multi_request_chunks = None
        self.is_c8_graph_bucket = (
            c8_graph_chunks is not None and multi_request_chunks is not None
        )
        self.is_batch_graph_bucket = (
            batch_graph_chunks is not None and multi_request_chunks is not None
        )
        if self.is_c1_graph_bucket:
            slots = gdn_slot_bucket(self.gdn_plan.slots)
            self.gdn_plan = replace(
                self.gdn_plan, slots=max(4, slots) if c1_fixed_width else slots
            )
        elif multi_request_chunks is not None:
            self.gdn_plan = replace(
                self.gdn_plan, slots=gdn_slot_bucket(self.gdn_plan.slots)
            )
        capacity = (
            C1_GRAPH_PREFIX_CAPACITY
            if self.is_c1_graph_bucket
            else cdiv(max(lengths), attention.CHUNK) * attention.CHUNK
        )
        if multi_request_chunks is not None:
            capacity = multi_request_chunks * attention.CHUNK
        self.stream_ids = gdn.to_device(plan.streams, torch.int32, device)
        self.conv_windows: dict[int, torch.Tensor] = {}
        pointers: list[int] = []
        resident_layers: list[tuple[torch.Tensor, StateSlots]] = []
        self.gdn_records: list[GDNRecord] = []
        self.state_slots: dict[str, StateSlots] = {}
        self.block_tables: dict[str, torch.Tensor] = {}
        self.kv_records: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        table_groups: dict[tuple[int, int], torch.Tensor] = {}
        geometry = None
        for name, meta in metadata.items():
            layer = layers[name]
            if isinstance(meta, TritonAttentionMetadata):
                shape = (layer.num_heads, layer.num_kv_heads, layer.head_size)
                if geometry is not None and shape != geometry:
                    raise RuntimeError("TensorFold tree needs uniform attention heads.")
                geometry = shape
                page = layer.kv_cache.shape[2]
                group = (id(meta.block_table), page)
                if group not in table_groups:
                    columns = max(1, cdiv(capacity, page))
                    table_groups[group] = meta.block_table[
                        : len(windows), :columns
                    ].clone(memory_format=torch.contiguous_format)
                self.block_tables[name] = table_groups[group]
            else:
                slots = slot_groups[id(meta)]
                self.state_slots[name] = slots
                if slots.device_ids is None:
                    pointers.extend(
                        gdn.pointers([layer.kv_cache[1][i] for i in slots.source])
                    )
                else:
                    resident_layers.append((layer.kv_cache[1], slots))
                kept = layer.conv1d.weight.shape[-1] - 1
                if kept not in self.conv_windows:
                    indices = plan.convolution_windows(windows, kept)
                    self.conv_windows[kept] = gdn.to_device(
                        [i for row in indices for i in row], torch.int32, device
                    ).view(-1, kept + 1)
        if geometry is None:
            raise RuntimeError("TensorFold tree needs at least one attention layer.")
        h, hk, _ = geometry
        if self.is_c1_graph_bucket:
            self.attention_plan = c1_graph_attention_plan(
                parents[0], lengths[0], h // hk, device
            )
        elif multi_request_chunks is not None:
            self.attention_plan = batch_graph_attention_plan(
                parents, lengths, h // hk, multi_request_chunks, device
            )
        else:
            self.attention_plan = attention.plan(parents, lengths, h // hk, device)
        self.resident_recurrent = None
        self.resident_commit_order: torch.Tensor | None = None
        if resident_layers:
            assert not pointers and len(windows) == 1
            self.gdn_table, self.resident_recurrent = prepare_resident_recurrent(
                resident_layers, plan.starts[-1], device
            )
        else:
            self.gdn_table = gdn.to_device(pointers, torch.int64, device).view(
                len(self.state_slots), len(windows)
            )
        self.gdn_tables = dict(
            zip(self.state_slots, self.gdn_table.unbind(0), strict=True)
        )
        self.workspace: tuple[torch.Tensor, ...] | None = None
        self.c1_resident_layout: C1ResidentRecurrentLayout | None = None
        self.slot_groups = tuple(slot_groups.values())
        self.table_groups = tuple(table_groups.values())
        self.conv_groups: list[ConvGroup] = []
        self.conv_histories: dict[str, torch.Tensor] = {}
        self.commit_graph: C1CommitGraph | None = None
        self.batch_commit: BatchCommitGraph | None = None
        self.capture_batch_commit = False
        self.c1_buffers: C1GraphPlanBuffers | None = None
        if fused_conv and len(windows) == 1:
            conv_layers = {}
            dim_first = is_conv_state_dim_first()
            for name, slots in self.state_slots.items():
                layer = layers[name]
                cache = layer.kv_cache[0]
                if not dim_first:
                    cache = cache.transpose(-1, -2)
                conv_layers[name] = (cache, slots, layer.conv1d.weight.shape[-1] - 1)
            self.conv_groups = group_conv_histories(conv_layers)
            if not self.conv_groups:
                logger.info_once(
                    "TensorFold C1 convolution fusion requires BF16 caches."
                )

    def graph_inputs(self) -> tuple[torch.Tensor, ...]:
        a = self.attention_plan
        return (
            self.gdn_plan.entries,
            self.gdn_plan.starts,
            self.stream_ids,
            self.gdn_table,
            *self.conv_windows.values(),
            *(group.batch.pointers for group in self.conv_groups),
            *(
                tensor
                for slots in self.slot_groups
                for tensor in (slots.conv, slots.offset)
            ),
            *self.table_groups,
            a.rows,
            a.streams,
            a.items,
            a.paths,
            a.depths,
        )

    def graph_key(self) -> tuple:
        conv_layout = tuple(
            (group.names, group.batch.channels, group.batch.keep, group.batch.strides)
            for group in self.conv_groups
        )
        return (
            self.is_c8_graph_bucket,
            self.is_batch_graph_bucket,
            self.gdn_plan.slots,
            self.gdn_plan.max_rows,
            self.attention_plan.chunks,
            conv_layout,
            tuple(self.state_slots) if self.resident_recurrent is not None else (),
            tuple(t.shape for t in self.graph_inputs()),
        )

    def stage(self, current: "TensorFoldVerify") -> None:
        self.plan = current.plan
        self.metadata = current.metadata
        self.resident_recurrent = current.resident_recurrent
        for dest, source in zip(
            self.graph_inputs(), current.graph_inputs(), strict=True
        ):
            dest.copy_(source, non_blocking=True)
        for dest, source in zip(self.slot_groups, current.slot_groups, strict=True):
            dest.source = source.source
            dest.all_slots = source.all_slots
            dest.device_ids = source.device_ids
            dest.device_counts = source.device_counts

    def bind_c1_graph_inputs(self, *, grouped_resident: bool = False) -> None:
        buffers = C1GraphPlanBuffers(
            self.gdn_plan, self.attention_plan, self.conv_windows
        )
        self.c1_buffers = buffers
        self.gdn_plan = buffers.gdn_plan
        self.attention_plan = buffers.attn_plan
        self.conv_windows = buffers.conv_windows
        self.c1_block_bindings = _aliases(self.block_tables)
        self.c1_slot_bindings = _aliases(self.state_slots)
        self.c1_resident_layout = None
        if grouped_resident and self.resident_recurrent is not None:
            group_index = {
                name: i
                for i, names in enumerate(self.c1_slot_bindings)
                for name in names
            }
            self.c1_resident_layout = C1ResidentRecurrentLayout(
                [(self.layers[n].kv_cache[1], group_index[n]) for n in self.state_slots]
            )

    def stage_c1(
        self,
        window: TreeWindow,
        plan: TreeBatchPlan,
        length: int,
        entries: list[int],
        metadata: dict[str, GDNAttentionMetadata | TritonAttentionMetadata],
        slot_groups: dict[int, StateSlots],
    ) -> torch.Tensor | None:
        buffers = self.c1_buffers
        if (
            buffers is None
            or len(entries) != buffers.gdn_plan.entries.numel()
            or metadata.keys() != self.metadata.keys()
        ):
            return None

        resident = self.resident_recurrent
        block_sources, slot_sources = {}, {}
        for names in self.c1_block_bindings:
            group = [metadata[name] for name in names]
            if not all(isinstance(meta, TritonAttentionMetadata) for meta in group):
                return None
            table = group[0].block_table
            if any(meta.block_table is not table for meta in group):
                return None
            dest = self.block_tables[names[0]]
            source = table[:1, : dest.shape[1]]
            if source.shape != dest.shape:
                return None
            block_sources[names[0]] = source
        for names in self.c1_slot_bindings:
            group = [slot_groups.get(id(metadata[name])) for name in names]
            slots = group[0]
            if (
                slots is None
                or (slots.device_ids is not None) != (resident is not None)
                or any(item is not slots for item in group)
            ):
                return None
            for name in names:
                slot_sources[name] = slots

        self.plan = plan
        self.metadata = metadata
        buffers.update(window, plan, length, entries)
        if resident is not None and self.c1_resident_layout is not None:
            self.c1_resident_layout.refresh(
                [slot_sources[names[0]] for names in self.c1_slot_bindings],
                len(window.parents),
                self.gdn_table,
                resident,
            )
        else:
            layers = [
                (self.layers[name].kv_cache[1], slot_sources[name])
                for name in self.state_slots
            ]
            if resident is not None:
                prepare_resident_recurrent(
                    layers,
                    len(window.parents),
                    self.gdn_table.device,
                    out=(self.gdn_table, resident),
                )
            else:
                pointers = [
                    recurrent.data_ptr()
                    + recurrent.stride(0) * recurrent.element_size() * slots.source[0]
                    for recurrent, slots in layers
                ]
                self.gdn_table.flatten().copy_(
                    torch.tensor(pointers, dtype=torch.int64).pin_memory(),
                    non_blocking=True,
                )
        for name, source in block_sources.items():
            self.block_tables[name].copy_(source, non_blocking=True)
        for names in self.c1_slot_bindings:
            name = names[0]
            dest, source = self.state_slots[name], slot_sources[name]
            dest.conv.copy_(source.conv, non_blocking=True)
            dest.offset.copy_(source.offset, non_blocking=True)
            dest.source = source.source
            dest.all_slots = source.all_slots
            dest.device_ids = source.device_ids
            dest.device_counts = source.device_counts
        return buffers.position_depths

    def prepare_conv_history(self) -> None:
        for group in self.conv_groups:
            histories = group.batch.gather(group.slots.conv, group.slots.offset)
            self.conv_histories.update(
                zip(group.names, histories.unbind(0), strict=True)
            )

    def attention(
        self,
        name: str,
        layer: "Attention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        cache = _paged_cache(layer)
        if self.workspace is None:
            self.workspace = partials(query, self.attention_plan)
            logger.info_once(
                "TensorFold Tree-DFlash2 used %s paged tree attention.",
                "BF16" if cache.dtype == torch.bfloat16 else "FP8",
            )
        paged_attention(
            query,
            key,
            value,
            cache,
            self.block_tables[name],
            self.attention_plan,
            layer._k_scale,
            layer._v_scale,
            layer.impl.scale,
            output,
            self.workspace,
            round_fp8_tail=self.round_fp8_tail,
        )
        self.kv_records[name] = (key, value)

    def gdn(
        self,
        layer: "QwenGatedDeltaNetAttention",
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        z: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.state_slots[layer.prefix]
        conv, recurrent = layer.kv_cache
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        weight = layer.conv1d.weight.flatten(1).contiguous()
        kept = weight.shape[1] - 1
        batch_history = self.conv_histories.get(layer.prefix)
        if batch_history is not None:
            history = batch_history[None]
        else:
            history = conv.index_select(0, slots.conv)
            offsets = slots.offset[:, None] + torch.arange(kept, device=self.device)
            history = history.gather(2, offsets[:, None].expand(-1, conv.shape[1], -1))
            history = history.transpose(1, 2).contiguous()
        qkv = mixed_qkv.contiguous()
        q, k, v, g, beta = gdn_pre(
            qkv,
            history.flatten(0, 1),
            weight,
            self.conv_windows[kept],
            a.contiguous(),
            b.contiguous(),
            layer.A_log,
            layer.dt_bias,
            kh=layer.num_k_heads,
            vh=layer.num_v_heads,
            dk=layer.head_k_dim,
            stream_ids=self.stream_ids,
            nkeep=kept,
        )
        y = gdn.tree(
            q, k, v, g, beta, self.gdn_plan, table=self.gdn_tables[layer.prefix]
        )
        gated, sums = gated_norm(y, z.contiguous(), layer.norm.weight, layer.norm.eps)
        if output is not None:
            output.copy_(gated.view_as(output))
        self.gdn_records.append(
            GDNRecord(k, v, g, beta, qkv, conv, history, recurrent, slots, layer.prefix)
        )
        return gated, sums

    def _commit_kv_layers(self) -> dict[str, "KVCommitLayer"]:
        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
            KVCommitLayer,
        )

        result = {}
        for name, (key, value) in self.kv_records.items():
            layer = self.layers[name]
            result[name] = KVCommitLayer(
                key,
                value,
                _paged_cache(layer),
                layer._k_scale,
                layer._v_scale,
                self.metadata[name].slot_mapping,
            )
        return result

    def accept_and_commit_batch(
        self,
        windows: list[TreeWindow],
        predicted: torch.Tensor,
    ) -> "BatchAcceptance | None":
        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
            BatchCommitGraph,
        )

        self._check_commit_records()
        if self.batch_commit is None:
            if not (
                self.gdn_records
                and self.kv_records
                and getattr(gdn, "SUPPORTS_REPLAY_DESTINATIONS", False)
            ):
                return None
            shape = self.gdn_records[0].recurrent.shape[1:]
            for record in self.gdn_records:
                if (
                    record.recurrent.shape[1:] != shape
                    or record.recurrent.dtype != torch.float32
                    or not record.recurrent[0].is_contiguous()
                    or record.conv.dtype not in (torch.bfloat16, torch.float32)
                    or record.prefix.dtype != record.conv.dtype
                    or record.qkv.dtype != record.conv.dtype
                ):
                    return None
            self.batch_commit = BatchCommitGraph(
                self.gdn_records,
                self._commit_kv_layers(),
                len(windows),
                self.plan.starts[-1],
                capture=self.capture_batch_commit,
            )
        mappings = {name: self.metadata[name].slot_mapping for name in self.kv_records}
        return self.batch_commit.accept_and_commit(
            windows, self.plan, predicted, mappings
        )

    def bind_commit_graph(
        self,
        *,
        overlap: bool = False,
        device_accept: bool = False,
        direct_recurrent_commit: bool = False,
    ) -> None:
        from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_commit import (
            C1CommitGraph,
        )

        if self.gdn_records and self.kv_records and not self.conv_groups:
            shape = self.gdn_records[0].recurrent.shape[1:]
            if any(record.recurrent.shape[1:] != shape for record in self.gdn_records):
                return
            self.commit_graph = C1CommitGraph(
                self.gdn_records,
                self._commit_kv_layers(),
                self.plan.starts[-1],
                self.device,
                overlap=overlap,
                device_accept=device_accept,
                direct_recurrent_commit=direct_recurrent_commit,
            )
            if self.resident_recurrent is not None:
                # Metadata can be grouped by cache layout; the replay table
                # follows the actual order of layer visits during verification.
                by_name = {name: row for row, name in enumerate(self.state_slots)}
                order = [by_name[record.name] for record in self.gdn_records]
                if order != list(range(len(order))):
                    self.resident_commit_order = gdn.to_device(
                        order, torch.int64, self.device
                    )

    def _check_commit_records(self) -> None:
        if (
            len(self.gdn_records) != len(self.state_slots)
            or self.kv_records.keys() != self.block_tables.keys()
        ):
            raise RuntimeError("A Qwen layer bypassed TensorFold tree verification.")

    def accept_and_commit(self, tree: TreeWindow, predicted: torch.Tensor):
        self._check_commit_records()
        assert self.commit_graph is not None
        resident = self.resident_recurrent
        if resident is not None and self.resident_commit_order is not None:
            resident = (
                resident.view(len(self.gdn_records), -1)
                .index_select(0, self.resident_commit_order)
                .flatten()
            )
        return self.commit_graph.accept_and_commit(
            tree,
            predicted,
            {name: self.metadata[name].slot_mapping for name in self.kv_records},
            resident,
        )

    def commit(
        self, paths: list[list[int]], *, direct_recurrent_commit: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_commit_records()
        for slots in self.slot_groups:
            slots.materialize_host()
        if self.commit_graph is not None:
            assert len(paths) == 1
            mappings = {
                name: self.metadata[name].slot_mapping for name in self.kv_records
            }
            graph_rows = self.commit_graph.commit(self.plan, paths[0], mappings)
            if graph_rows is not None:
                return graph_rows
        accepted = [len(path) for path in paths]
        sources, destinations = self.plan.commit_rows(paths)
        src_rows = gdn.to_device(sources, torch.int64, self.device)
        dst_rows = gdn.to_device(destinations, torch.int64, self.device)
        for name, (k, v) in self.kv_records.items():
            layer = self.layers[name]
            slots = self.metadata[name].slot_mapping.index_select(0, dst_rows)
            commit_paged_kv(
                k,
                v,
                src_rows,
                slots,
                _paged_cache(layer),
                layer._k_scale,
                layer._v_scale,
            )

        records = self.gdn_records
        direct = direct_recurrent_commit and len(paths) == 1
        pointers = eager_recurrent_table(records, accepted, direct=direct)
        width = max(accepted)
        rows = [
            start + row
            for start, path in zip(self.plan.starts, paths)
            for row in [*path, *([0] * (width - len(path)))]
        ]
        replay_rows = gdn.to_device(rows, torch.int32, self.device).view(-1, width)
        counts = gdn.to_device(accepted, torch.int32, self.device)
        table = gdn.to_device(pointers, torch.int64, self.device)
        gdn.replay(
            table,
            len(records),
            len(paths),
            replay_rows,
            counts,
            records[0].k,
            records[0].v,
            in_place=True,
        )
        if direct:
            logger.info_once(
                "TensorFold wrote C1 recurrent state directly in ordinary commit."
            )

        if self.conv_groups:
            by_name = {record.name: record for record in records}
            for group in self.conv_groups:
                grouped_records = [by_name[name] for name in group.names]
                group.batch.commit(
                    [(r.conv, r.prefix, r.qkv) for r in grouped_records],
                    self.plan.convolution_commit_indices(paths, group.batch.keep),
                    group.slots.conv,
                    accepted[0] - 1,
                )
            logger.info_once("TensorFold fused the C1 GDN convolution history.")
            logger.info_once(
                "TensorFold fused C1 convolution across %d GDN cache groups.",
                len(self.conv_groups),
            )
            return src_rows, dst_rows

        by_kept: dict[int, list[GDNRecord]] = {}
        for record in records:
            by_kept.setdefault(record.prefix.shape[1], []).append(record)
        for kept, group in by_kept.items():
            picks = self.plan.convolution_commit_indices(paths, kept)
            indices = gdn.to_device(picks, torch.int64, self.device)
            offsets = (
                counts.long()[:, None] - 1 + torch.arange(kept, device=self.device)
            )
            # TensorFold gathers the last kept rows of [old history | accepted].
            # vLLM stores that window at accepted_count - 1 for the next round.
            if group[0].qkv.shape[0] <= 32:
                combined = torch.cat(
                    (
                        torch.stack([r.prefix.flatten(0, 1) for r in group]),
                        torch.stack([r.qkv for r in group]),
                    ),
                    dim=1,
                )
                histories = combined.index_select(1, indices).unbind(0)
            else:
                histories = [
                    torch.cat((r.prefix.flatten(0, 1), r.qkv)).index_select(0, indices)
                    for r in group
                ]
            for record, tail in zip(group, histories, strict=True):
                history = tail.view(len(paths), kept, -1)
                record.conv[record.slots.conv[:, None], :, offsets] = history
        return src_rows, dst_rows
