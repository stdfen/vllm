# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay accepted C1 state updates on vLLM-owned cache pages."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensorfold.cuda.kernels import gdn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_attention import commit_paged_kv

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_plan import (
        TreeBatchPlan,
        TreeWindow,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash2.tensorfold_verify import (
        GDNRecord,
    )

logger = init_logger(__name__)


@triton.jit
def _copy_recurrent_slots(
    SOURCES, DESTINATIONS, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    layer = tl.program_id(0)
    source = tl.load(SOURCES + layer)
    destination = tl.load(DESTINATIONS + layer)
    if source != destination:
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(source.to(tl.pointer_type(tl.float32)) + offset, offset < SIZE)
        tl.store(
            destination.to(tl.pointer_type(tl.float32)) + offset, value, offset < SIZE
        )


@triton.jit
def _device_accept(
    PREDICTIONS,
    TREE,
    RECURRENT,
    ROWS,
    COUNT,
    PICKS,
    SOURCES,
    DESTINATIONS,
    RECEIPT,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYERS: tl.constexpr,
    LAYER_BLOCK: tl.constexpr,
    KEPT: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
):
    order = tl.arange(0, BLOCK)
    ids = tl.load(TREE + order, order < WIDTH, other=-1)
    parents = tl.load(TREE + WIDTH + order, order < WIDTH, other=-2)
    path = tl.full((BLOCK,), 0, tl.int64)
    emitted = tl.full((BLOCK,), -1, tl.int64)
    row = tl.full((), 0, tl.int32)
    accepted = tl.full((), 0, tl.int32)
    while row < WIDTH:
        token = tl.load(PREDICTIONS + row)
        path = tl.where(order == accepted, row.to(tl.int64), path)
        emitted = tl.where(order == accepted, token, emitted)
        row = tl.min(tl.where((parents == row) & (ids == token), order, WIDTH), 0)
        accepted += 1
    tl.store(ROWS + order, path, order < WIDTH)
    tl.store(COUNT, accepted)
    tl.store(RECEIPT, accepted)
    tl.store(RECEIPT + 1 + order, path, order < WIDTH)
    tl.store(RECEIPT + 1 + WIDTH + order, emitted, order < WIDTH)

    layers = tl.arange(0, LAYER_BLOCK)
    record = RECURRENT + layers * (WIDTH + 3)
    base = tl.load(record, layers < LAYERS, other=0)
    stride = tl.load(record + 1, layers < LAYERS, other=0)
    source = tl.load(record + 2, layers < LAYERS, other=0)
    slot = tl.load(record + 3 + accepted - 1, layers < LAYERS, other=0)
    tl.store(SOURCES + layers, source, layers < LAYERS)
    tl.store(DESTINATIONS + layers, base + stride * slot, layers < LAYERS)

    offset = 0
    for group in tl.static_range(len(KEPT)):
        kept = KEPT[group]
        i = tl.arange(0, CONV_BLOCK)
        parent_index = accepted - kept + i
        # All packed paths live in this CTA. Read from the lane holding each row.
        ancestor = tl.gather(
            path, tl.minimum(tl.maximum(parent_index, 0), WIDTH - 1), 0
        )
        index = tl.where(parent_index < 0, kept + parent_index, kept + ancestor)
        tl.store(PICKS + offset + i, index, i < kept)
        offset += kept


@dataclass(frozen=True)
class KVCommitLayer:
    key: torch.Tensor
    value: torch.Tensor
    cache: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor
    slots: torch.Tensor


class C1CommitGraph:
    """A CUDA Graph bound to one captured target's provisional state tensors."""

    def __init__(
        self,
        records: list["GDNRecord"],
        kv_layers: dict[str, KVCommitLayer],
        width: int,
        device: torch.device,
        *,
        overlap: bool = False,
        device_accept: bool = False,
        direct_recurrent_commit: bool = False,
    ):
        if direct_recurrent_commit and not getattr(
            gdn, "SUPPORTS_REPLAY_DESTINATIONS", False
        ):
            raise ValueError(
                "Direct recurrent commit requires destination-slot replay."
            )
        self.records = tuple(records)
        self.width = width
        self.device = device
        self.direct_recurrent_commit = direct_recurrent_commit
        self.graph: torch.cuda.CUDAGraph | None = None
        self.replay_stream = torch.cuda.Stream(device=device) if overlap else None
        self.pending = False
        self.conv_groups: dict[int, list[GDNRecord]] = {}
        for record in self.records:
            self.conv_groups.setdefault(record.prefix.shape[1], []).append(record)

        names_by_mapping: dict[int, list[str]] = {}
        self.kv_layers = kv_layers
        for name, layer in kv_layers.items():
            names_by_mapping.setdefault(id(layer.slots), []).append(name)
        self.kv_groups = [
            (names, kv_layers[names[0]].slots.new_empty(width))
            for names in names_by_mapping.values()
        ]
        self.replay_inputs = gdn.pointers(
            [
                tensor
                for record in self.records
                for tensor in (record.k, record.v, record.g, record.beta)
            ]
        )
        sizes = [self.width, 1, *self.conv_groups, 6 * len(self.records)]
        self.inputs = torch.empty(sum(sizes), device=self.device, dtype=torch.int64)
        self.rows, self.count, *picks, pointers = self.inputs.split(sizes)
        self.conv_picks = dict(zip(self.conv_groups, picks, strict=True))
        layers = len(self.records)
        first_slots = pointers[4 * layers : 5 * layers]
        second_slots = pointers[5 * layers :]
        if direct_recurrent_commit:
            self.replay_table = pointers
            self.sources, self.destinations = first_slots, second_slots
        else:
            self.replay_table = pointers[: 5 * layers]
            self.destinations, self.sources = first_slots, second_slots
        self.state_size = self.records[0].recurrent[0].numel()
        self.dest_rows = torch.arange(self.width, device=self.device, dtype=torch.int64)
        self.conv_rows = {
            kept: torch.arange(kept, device=self.device, dtype=torch.int64)
            for kept in self.conv_groups
        }
        self.device_accept = device_accept
        if device_accept:
            num_inputs = 2 * width + (width + 3) * layers
            self.accept_inputs = self.inputs.new_empty(num_inputs)
            self.accept_receipt = self.inputs.new_empty(1 + 2 * width)
            self.host_receipt = torch.empty(
                1 + 2 * width, dtype=torch.int64, pin_memory=True
            )
            self.receipt_ready = torch.cuda.Event()
            self.replay_table[: 4 * layers].copy_(
                torch.tensor(self.replay_inputs, dtype=torch.int64, device=device)
            )

    def _write_state(self) -> None:
        active = self.dest_rows < self.count
        for names, slots in self.kv_groups:
            slots = torch.where(active, slots, -1)
            for name in names:
                layer = self.kv_layers[name]
                commit_paged_kv(
                    layer.key,
                    layer.value,
                    self.rows,
                    slots,
                    layer.cache,
                    layer.key_scale,
                    layer.value_scale,
                )

        first = self.records[0]
        if not self.direct_recurrent_commit:
            grid = (len(self.records), triton.cdiv(self.state_size, 1024))
            _copy_recurrent_slots[grid](
                self.sources,
                self.destinations,
                self.state_size,
                BLOCK=1024,
                num_warps=4,
            )
        gdn.replay(
            self.replay_table,
            len(self.records),
            1,
            self.rows.int()[None],
            self.count.int(),
            first.k,
            first.v,
            in_place=True,
        )

        for kept, records in self.conv_groups.items():
            combined = torch.cat(
                (
                    torch.stack([record.prefix.flatten(0, 1) for record in records]),
                    torch.stack([record.qkv for record in records]),
                ),
                dim=1,
            )
            histories = combined.index_select(1, self.conv_picks[kept]).unbind(0)
            offsets = self.count[:, None] - 1 + self.conv_rows[kept]
            for record, tail in zip(records, histories, strict=True):
                record.conv[record.slots.conv[:, None], :, offsets] = tail[None]

    def _stage_kv(
        self, slot_mappings: dict[str, torch.Tensor], required_rows: int
    ) -> bool:
        mappings = []
        for names, dest in self.kv_groups:
            current = slot_mappings[names[0]]
            prefix = current[: self.width]
            if (
                prefix.ndim != 1
                or prefix.numel() < required_rows
                or any(slot_mappings[name] is not current for name in names)
            ):
                return False
            mappings.append((dest, prefix))
        for dest, current in mappings:
            count = current.numel()
            dest[:count].copy_(current, non_blocking=True)
            if count < self.width:
                dest[count:].fill_(-1)
        return True

    def commit(
        self,
        plan: "TreeBatchPlan",
        path: list[int],
        slot_mappings: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        assert not self.pending
        accepted = len(path)
        if not self._stage_kv(slot_mappings, accepted):
            return None
        sources = [r.recurrent[r.slots.source[0]] for r in self.records]
        dests = [r.recurrent[r.slots.all_slots[0][accepted - 1]] for r in self.records]
        state_pointers = (
            (sources, dests) if self.direct_recurrent_commit else (dests, sources)
        )
        values = [
            *path,
            *([0] * (self.width - accepted)),
            accepted,
            *(
                index
                for kept in self.conv_groups
                for index in plan.convolution_commit_indices([path], kept)
            ),
            *self.replay_inputs,
            *gdn.pointers([slot for slots in state_pointers for slot in slots]),
        ]
        self.inputs.copy_(
            torch.tensor(values, dtype=torch.int64).pin_memory(), non_blocking=True
        )
        self._launch()
        return self.rows[:accepted], self.dest_rows[:accepted]

    def accept_and_commit(
        self,
        tree: "TreeWindow",
        predictions: torch.Tensor,
        slot_mappings: dict[str, torch.Tensor],
        resident_recurrent: torch.Tensor | None = None,
    ) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor, bool] | None:
        """Start cache writes while the scheduler's compact receipt reaches the host."""
        assert self.device_accept and not self.pending
        real_rows = len(tree.tokens)
        if (
            real_rows > self.width
            or (
                resident_recurrent is None
                and any(len(r.slots.all_slots[0]) < real_rows for r in self.records)
            )
            or not self._stage_kv(slot_mappings, real_rows)
        ):
            return None
        pad = self.width - real_rows
        values = [*tree.tokens, *([0] * pad), *tree.parents, *([-2] * pad)]
        if resident_recurrent is None:
            for record in self.records:
                base = record.recurrent.data_ptr()
                stride = record.recurrent.stride(0) * record.recurrent.element_size()
                slots = record.slots.all_slots[0][: self.width]
                values.extend(
                    (
                        base,
                        stride,
                        base + stride * record.slots.source[0],
                        *slots,
                        *([0] * (self.width - len(slots))),
                    )
                )
            resident_recurrent = self.accept_inputs[2 * self.width :]
        self.accept_inputs[: len(values)].copy_(
            torch.tensor(values, dtype=torch.int64).pin_memory(), non_blocking=True
        )
        block = triton.next_power_of_2(max(self.width, max(self.conv_groups)))
        _device_accept[(1,)](
            predictions,
            self.accept_inputs,
            resident_recurrent,
            self.rows,
            self.count,
            self.inputs[self.width + 1 :],
            self.sources,
            self.destinations,
            self.accept_receipt,
            WIDTH=self.width,
            BLOCK=block,
            LAYERS=len(self.records),
            LAYER_BLOCK=triton.next_power_of_2(len(self.records)),
            KEPT=tuple(self.conv_groups),
            CONV_BLOCK=block,
            num_warps=4,
        )
        self.host_receipt.copy_(self.accept_receipt, non_blocking=True)
        self.receipt_ready.record()
        replay = self.graph is not None
        # _launch orders any side-stream replay after the staged pointers and
        # selected rows. The host waits for the receipt while the commit runs.
        self._launch()
        self.receipt_ready.synchronize()
        receipt = self.host_receipt.tolist()
        accepted = receipt[0]
        return (
            receipt[1 : 1 + accepted],
            receipt[1 + self.width : 1 + self.width + accepted],
            self.rows[:accepted],
            self.dest_rows[:accepted],
            replay,
        )

    def _launch(self) -> None:
        if self.graph is None:
            # Warm the kernels by committing this round once. CUDA capture
            # records the same kernels without committing this round again.
            self._write_state()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            with torch.cuda.graph(graph, stream=stream):
                self._write_state()
            current.wait_stream(stream)
            self.graph = graph
        else:
            if self.replay_stream is None:
                self.graph.replay()
            else:
                self.replay_stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.replay_stream):
                    self.graph.replay()
                self.pending = True
                logger.info_once(
                    "TensorFold dispatched the C1 commit Graph on a side stream."
                )
            logger.info_once("TensorFold replayed the C1 accepted-state commit graph.")

    def wait(self) -> None:
        if self.pending:
            assert self.replay_stream is not None
            torch.cuda.current_stream(self.device).wait_stream(self.replay_stream)
            self.pending = False


# A capped batch has one independent path per request. Staging uses a fixed
# width so a target graph can also replay when the per-request widths change.
BATCH_COMMIT_WIDTH = 32


@triton.jit
def _batch_accept(
    PRED,
    TOKENS,
    PARENTS,
    STARTS,
    ROWS,
    COUNTS,
    RECEIPT,
    KV_ROWS,
    KV_ACTIVE,
    WIDTH: tl.constexpr,
):
    stream = tl.program_id(0)
    start = tl.load(STARTS + stream).to(tl.int32)
    stop = tl.load(STARTS + stream + 1).to(tl.int32)
    lane = tl.arange(0, WIDTH)
    ids = tl.load(TOKENS + start + lane, start + lane < stop, other=-1)
    parents = tl.load(PARENTS + start + lane, start + lane < stop, other=-2)
    path = tl.full((WIDTH,), 0, tl.int32)
    emitted = tl.full((WIDTH,), -1, tl.int64)
    row = tl.full((), 0, tl.int32)
    count = tl.full((), 0, tl.int32)
    while row < WIDTH:
        token = tl.load(PRED + start + row)
        path = tl.where(lane == count, row, path)
        emitted = tl.where(lane == count, token, emitted)
        row = tl.min(tl.where((parents == row) & (ids == token), lane, WIDTH), 0)
        count += 1
    tl.store(ROWS + stream * WIDTH + lane, start + path)
    tl.store(KV_ROWS + start + lane, start + path, start + lane < stop)
    tl.store(KV_ACTIVE + start + lane, lane < count, start + lane < stop)
    tl.store(COUNTS + stream, count)
    receipt = RECEIPT + stream * (1 + 2 * WIDTH)
    tl.store(receipt, count)
    tl.store(receipt + 1 + lane, path)
    tl.store(receipt + 1 + WIDTH + lane, emitted)


@triton.jit
def _batch_pack_accepted(
    ROWS, STARTS, COUNTS, SOURCE, DEST, WIDTH: tl.constexpr, STREAMS: tl.constexpr
):
    stream = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    streams = tl.arange(0, 8)
    previous = tl.load(COUNTS + streams, (streams < stream) & (streams < STREAMS), 0)
    base = tl.sum(previous, 0)
    count = tl.load(COUNTS + stream)
    start = tl.load(STARTS + stream)
    row = tl.load(ROWS + stream * WIDTH + lane)
    tl.store(SOURCE + base + lane, row, lane < count)
    tl.store(DEST + base + lane, start + lane, lane < count)


@triton.jit
def _batch_recurrent_pointers(
    SLOT_ROWS,
    LAYERS_INFO,
    COUNTS,
    TABLE,
    WIDTH: tl.constexpr,
    LAYERS: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    stream = tl.program_id(0)
    layer = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(LAYERS_INFO + layer * 3, layer < LAYERS, 0)
    stride = tl.load(LAYERS_INFO + layer * 3 + 1, layer < LAYERS, 0)
    group = tl.load(LAYERS_INFO + layer * 3 + 2, layer < LAYERS, 0)
    slots = SLOT_ROWS + (group * STREAMS + stream) * (WIDTH + 1)
    accepted = tl.load(COUNTS + stream)
    old = tl.load(slots, layer < LAYERS, 0)
    new = tl.load(slots + accepted, layer < LAYERS, 0)
    src = TABLE + 4 * LAYERS + stream * LAYERS + layer
    dst = src + STREAMS * LAYERS
    tl.store(src, base + stride * old, layer < LAYERS)
    tl.store(dst, base + stride * new, layer < LAYERS)


@triton.jit
def _batch_conv_commit(
    PTRS,
    COUNTS,
    ROWS,
    CONV_STRIDE: tl.constexpr,
    PREFIX_STRIDE: tl.constexpr,
    QKV_STRIDE: tl.constexpr,
    KEPT: tl.constexpr,
    CHANNELS: tl.constexpr,
    BF16: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    stream = tl.program_id(1)
    patch = tl.program_id(2)
    tiles = tl.cdiv(CHANNELS, BLOCK)
    col = (patch % tiles) * BLOCK + tl.arange(0, BLOCK)
    pos = patch // tiles
    dtype = tl.bfloat16 if BF16 else tl.float32
    row = PTRS + layer * 4
    conv = tl.load(row).to(tl.pointer_type(dtype))
    prefix = tl.load(row + 1).to(tl.pointer_type(dtype))
    qkv = tl.load(row + 2).to(tl.pointer_type(dtype))
    slots = tl.load(row + 3).to(tl.pointer_type(tl.int64))
    count = tl.load(COUNTS + stream)
    accepted_index = count - KEPT + pos
    if accepted_index < 0:
        offset = (
            stream * PREFIX_STRIDE[0]
            + (KEPT + accepted_index) * PREFIX_STRIDE[1]
            + col * PREFIX_STRIDE[2]
        )
        value = tl.load(prefix + offset, col < CHANNELS, other=0)
    else:
        source = tl.load(ROWS + stream * WIDTH + accepted_index)
        value = tl.load(
            qkv + source * QKV_STRIDE[0] + col * QKV_STRIDE[1],
            col < CHANNELS,
            other=0,
        )
    slot = tl.load(slots + stream)
    destination = (
        slot * CONV_STRIDE[0]
        + col * CONV_STRIDE[1]
        + (count - 1 + pos) * CONV_STRIDE[2]
    )
    tl.store(conv + destination, value, col < CHANNELS)


@dataclass
class BatchAcceptance:
    paths: list[list[int]]
    emitted: list[list[int]]
    source: torch.Tensor
    destination: torch.Tensor
    sampled_ids: torch.Tensor
    counts: torch.Tensor
    replayed: bool


class BatchCommitGraph:
    """Commit packed multi-request KV, GDN state and convolution history."""

    def __init__(
        self,
        records: list["GDNRecord"],
        kv_layers: dict[str, KVCommitLayer],
        streams: int,
        rows: int,
        *,
        capture: bool,
    ):
        assert 2 <= streams <= 8 and records and kv_layers
        self.device = records[0].k.device
        self.records = tuple(records)
        self.kv_layers = kv_layers
        self.streams = streams
        self.total_rows = rows
        self.capture = capture
        self.graph: torch.cuda.CUDAGraph | None = None
        width = BATCH_COMMIT_WIDTH
        groups = {}
        layer_info = []
        layouts = {}
        for r in records:
            group, _ = groups.setdefault(id(r.slots), (len(groups), r.slots))
            stride = r.recurrent.stride(0) * r.recurrent.element_size()
            layer_info.append((r.recurrent.data_ptr(), stride, group))
            layout = (
                r.conv.stride(),
                r.prefix.stride(),
                r.qkv.stride(),
                r.prefix.shape[1],
                r.qkv.shape[1],
                r.conv.dtype == torch.bfloat16,
            )
            layouts.setdefault(layout, []).append(
                (
                    r.conv.data_ptr(),
                    r.prefix.data_ptr(),
                    r.qkv.data_ptr(),
                    r.slots.conv.data_ptr(),
                )
            )
        self.state_groups = tuple(slots for _, slots in groups.values())
        self.layer_info = gdn.to_device(
            [value for item in layer_info for value in item], torch.int64, self.device
        ).view(-1, 3)
        self.conv_layouts = [
            (
                layout,
                gdn.to_device(
                    [value for row in pointers for value in row],
                    torch.int64,
                    self.device,
                ).view(-1, 4),
            )
            for layout, pointers in layouts.items()
        ]

        def empty(*shape, dtype=torch.int64):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.table = empty(len(records) * (4 + 2 * streams))
        self.table[: 4 * len(records)].copy_(
            gdn.to_device(
                gdn.pointers([t for r in records for t in (r.k, r.v, r.g, r.beta)]),
                torch.int64,
                self.device,
            )
        )
        self.staging = empty(
            2 * rows + streams + 1 + len(groups) * streams * (width + 1)
        )
        self.starts = self.staging[2 * rows : 2 * rows + streams + 1]
        self.slots = self.staging[2 * rows + streams + 1 :]
        self.rows = empty(streams, width, dtype=torch.int32)
        self.counts = empty(streams, dtype=torch.int32)
        self.source = empty(rows)
        self.destination = empty(rows)
        self.receipt = empty(streams, 1 + 2 * width)
        self.host_receipt = torch.empty(
            self.receipt.shape, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.receipt_ready = torch.cuda.Event()
        self.kv_rows = empty(rows)
        self.kv_active = empty(rows, dtype=torch.bool)
        kv_groups: dict[int, list[str]] = {}
        for name, layer in kv_layers.items():
            kv_groups.setdefault(id(layer.slots), []).append(name)
        self.kv_groups = [
            (names, kv_layers[names[0]].slots.new_empty(rows))
            for names in kv_groups.values()
        ]

    def _write_state(self):
        for names, mapping in self.kv_groups:
            slots = mapping.masked_fill(~self.kv_active, -1)
            for name in names:
                layer = self.kv_layers[name]
                commit_paged_kv(
                    layer.key,
                    layer.value,
                    self.kv_rows,
                    slots,
                    layer.cache,
                    layer.key_scale,
                    layer.value_scale,
                )

        layer_count = len(self.records)
        _batch_recurrent_pointers[(self.streams, triton.cdiv(layer_count, 32))](
            self.slots,
            self.layer_info,
            self.counts,
            self.table,
            BATCH_COMMIT_WIDTH,
            layer_count,
            self.streams,
            32,
            num_warps=1,
        )
        first = self.records[0]
        gdn.replay(
            self.table,
            layer_count,
            self.streams,
            self.rows,
            self.counts,
            first.k,
            first.v,
            in_place=True,
        )

        for layout, pointers in self.conv_layouts:
            conv, prefix, qkv, kept, channels, bf16 = layout
            if kept:
                block = 128
                grid = (
                    pointers.shape[0],
                    self.streams,
                    kept * triton.cdiv(channels, block),
                )
                _batch_conv_commit[grid](
                    pointers,
                    self.counts,
                    self.rows,
                    conv,
                    prefix,
                    qkv,
                    kept,
                    channels,
                    bf16,
                    BATCH_COMMIT_WIDTH,
                    block,
                    num_warps=4,
                )

    def accept_and_commit(
        self,
        windows: list["TreeWindow"],
        plan: "TreeBatchPlan",
        predicted: torch.Tensor,
        mappings: dict[str, torch.Tensor],
    ) -> BatchAcceptance | None:
        width, rows = BATCH_COMMIT_WIDTH, self.total_rows
        # Revalidate live aliases before writing any state.
        live_mappings = []
        for names, buffer in self.kv_groups:
            live = mappings[names[0]]
            if (
                live.ndim != 1
                or live.numel() < rows
                or any(mappings[name] is not live for name in names)
            ):
                return None
            live_mappings.append((buffer, live))
        values = [token for tree in windows for token in tree.tokens]
        values.extend(parent for tree in windows for parent in tree.parents)
        values.extend(plan.starts)
        for group in self.state_groups:
            for source, row, tree in zip(
                group.source, group.all_slots, windows, strict=True
            ):
                if min(len(row), width) < len(tree.tokens):
                    return None
                values.extend((source, *row[:width], *([0] * max(0, width - len(row)))))
        self.staging.copy_(
            torch.tensor(values, dtype=torch.int64).pin_memory(), non_blocking=True
        )
        for buffer, live in live_mappings:
            buffer.copy_(live[:rows], non_blocking=True)
        _batch_accept[(self.streams,)](
            predicted,
            self.staging,
            self.staging[rows:],
            self.starts,
            self.rows,
            self.counts,
            self.receipt,
            self.kv_rows,
            self.kv_active,
            width,
            num_warps=1,
        )
        _batch_pack_accepted[(self.streams,)](
            self.rows,
            self.starts,
            self.counts,
            self.source,
            self.destination,
            width,
            self.streams,
            num_warps=1,
        )
        self.host_receipt.copy_(self.receipt, non_blocking=True)
        self.receipt_ready.record()
        replayed = self.graph is not None
        if replayed:
            self.graph.replay()
        else:
            self._write_state()
            if self.capture:
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream(device=self.device)
                current = torch.cuda.current_stream(self.device)
                stream.wait_stream(current)
                with torch.cuda.graph(graph, stream=stream):
                    self._write_state()
                current.wait_stream(stream)
                self.graph = graph
        self.receipt_ready.synchronize()
        receipt = self.host_receipt.tolist()
        lengths = [r[0] for r in receipt]
        count = sum(lengths)
        return BatchAcceptance(
            [r[1 : 1 + n] for r, n in zip(receipt, lengths, strict=True)],
            [
                r[1 + width : 1 + width + n]
                for r, n in zip(receipt, lengths, strict=True)
            ],
            self.source[:count],
            self.destination[:count],
            self.receipt[:, 1 + width : 1 + width + max(lengths)].contiguous(),
            self.counts.clone(),
            replayed,
        )
