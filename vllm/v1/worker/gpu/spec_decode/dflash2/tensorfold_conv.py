# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch convolution history loads and stores for one TensorFold request."""

import torch
from tensorfold.cuda.kernels.gdn import to_device

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_conv(
    PTRS,
    SLOT,
    OFFSET,
    OUT,
    CHANNELS: tl.constexpr,
    KEEP: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    STRIDE_CHANNEL: tl.constexpr,
    STRIDE_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row = tl.program_id(0), tl.program_id(1)
    channel = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    cache = tl.load(PTRS + layer).to(tl.pointer_type(tl.bfloat16))
    slot = tl.load(SLOT).to(tl.int64)
    offset = tl.load(OFFSET).to(tl.int64) + row
    source = slot * STRIDE_SLOT + channel * STRIDE_CHANNEL + offset * STRIDE_OFFSET
    value = tl.load(cache + source, channel < CHANNELS, 0)
    tl.store(OUT + (layer * KEEP + row) * CHANNELS + channel, value, channel < CHANNELS)


@triton.jit(do_not_specialize=["OFFSET"])
def _commit_conv(
    PTRS,
    PICKS,
    SLOT,
    OFFSET,
    CHANNELS: tl.constexpr,
    KEEP: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    STRIDE_CHANNEL: tl.constexpr,
    STRIDE_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row = tl.program_id(0), tl.program_id(1)
    channel = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    cache = tl.load(PTRS + layer * 3).to(tl.pointer_type(tl.bfloat16))
    pick = tl.load(PICKS + row)
    if pick < KEEP:
        history = tl.load(PTRS + layer * 3 + 1).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(history + pick * CHANNELS + channel, channel < CHANNELS, 0)
    else:
        qkv = tl.load(PTRS + layer * 3 + 2).to(tl.pointer_type(tl.bfloat16))
        value = tl.load(qkv + (pick - KEEP) * CHANNELS + channel, channel < CHANNELS, 0)
    slot = tl.load(SLOT).to(tl.int64)
    dest = (
        slot * STRIDE_SLOT + channel * STRIDE_CHANNEL + (OFFSET + row) * STRIDE_OFFSET
    )
    tl.store(cache + dest, value, channel < CHANNELS)


class ConvHistoryBatch:
    """Layers sharing BF16 cache geometry and one decode request's state slots."""

    def __init__(self, caches: list[torch.Tensor], keep: int):
        self.device = caches[0].device
        self.channels = caches[0].shape[1]
        self.keep = keep
        self.strides = caches[0].stride()
        self.layers = len(caches)
        self.pointers = to_device(
            [cache.data_ptr() for cache in caches], torch.int64, self.device
        )
        self.grid = (self.layers, self.keep, triton.cdiv(self.channels, 128))

    def gather(self, slot: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (self.layers, self.keep, self.channels),
            dtype=torch.bfloat16,
            device=self.device,
        )
        _gather_conv[self.grid](
            self.pointers,
            slot,
            offset,
            out,
            self.channels,
            self.keep,
            *self.strides,
            BLOCK=128,
        )
        return out

    def commit(
        self,
        layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        picks: list[int],
        slot: torch.Tensor,
        offset: int,
    ) -> None:
        pointers = to_device(
            [tensor.data_ptr() for record in layers for tensor in record],
            torch.int64,
            self.device,
        )
        indices = to_device(picks, torch.int32, self.device)
        _commit_conv[self.grid](
            pointers,
            indices,
            slot,
            offset,
            self.channels,
            self.keep,
            *self.strides,
            BLOCK=128,
        )
