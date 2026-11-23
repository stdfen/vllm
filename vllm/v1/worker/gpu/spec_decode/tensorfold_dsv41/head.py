# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 TensorFold Markov rows with a GPU-resident frequent-token cache."""

import json
import os

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _first_argmax(a, ai, b, bi):
    take_a = (a > b) | ((a == b) & (ai < bi))
    take_a = take_a | ((a != a) & ((b == b) | (ai < bi)))
    return tl.where(take_a, a, b), tl.where(take_a, ai, bi)


@triton.jit
def _finish_argmax(values, indices, output, blocks: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    scores = tl.load(values + row * blocks + offsets, offsets < blocks, -float("inf"))
    ids = tl.load(indices + row * blocks + offsets, offsets < blocks, 2147483647)
    _, winner = tl.reduce((scores, ids), 0, _first_argmax)
    tl.store(output + row, winner)


@triton.jit
def _bias_rows(
    ids,
    slots,
    cached,
    embeddings,
    weights,
    output,
    base,
    partial_values,
    partial_ids,
    base_stride: tl.constexpr,
    GREEDY: tl.constexpr,
    vocab: tl.constexpr,
    rank: tl.constexpr,
    emb_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = cols < vocab
    token = tl.load(ids + row).to(tl.int64)
    slot = tl.load(slots + token)
    if slot >= 0:
        values = tl.load(cached + slot.to(tl.int64) * vocab + cols, valid, 0)
    else:
        k = tl.arange(0, rank)
        embed = tl.load(embeddings + row * emb_stride + k).to(tl.float16).to(tl.float32)
        weight = tl.load(
            weights + cols[:, None].to(tl.int64) * rank + k[None, :], valid[:, None], 0
        ).to(tl.float32)
        values = tl.sum(weight * embed[None, :], 1).to(tl.float16)
    if GREEDY:
        scores = tl.load(base + row * base_stride + cols, valid, -float("inf"))
        scores = scores.to(tl.float32) + values.to(tl.float32)
        scores = tl.where(valid, scores, -float("inf"))
        indices = tl.where(valid, cols, 2147483647)
        score, index = tl.reduce((scores, indices), 0, _first_argmax)
        offset = row * tl.cdiv(vocab, BLOCK) + tl.program_id(1)
        tl.store(partial_values + offset, score)
        tl.store(partial_ids + offset, index)
    else:
        tl.store(output + row * vocab + cols, values, valid)


class FP16BiasCache:
    """Use identical FP32 reductions for cached rows and cache misses."""

    def __init__(self, embedding, projection, token_ids=None, capacity=None):
        self.projection = projection
        self.vocab, self.rank = projection.shape
        if self.rank & (self.rank - 1):
            raise ValueError("TensorFold Markov rank must be a power of two")
        if capacity is None:
            capacity = int(os.environ.get("VLLM_TENSORFOLD_BIAS_CACHE_ROWS", "256"))
        if capacity < 0:
            raise ValueError("Negative TensorFold bias cache capacity")
        self.capacity = capacity
        self.slots = torch.full(
            (self.vocab,), -1, dtype=torch.int32, device=projection.device
        )
        if capacity and token_ids is None:
            path = os.environ.get(
                "VLLM_TENSORFOLD_CACHE_IDS", "/opt/ab/assets/markov_top_tokens.json"
            )
            with open(path) as stream:
                token_ids = json.load(stream)
        seen = set()
        selected = [
            int(t)
            for t in (token_ids or [])
            if 0 <= int(t) < self.vocab and not (int(t) in seen or seen.add(int(t)))
        ][:capacity]
        self.cached = torch.empty(
            (max(1, len(selected)), self.vocab),
            dtype=torch.float16,
            device=projection.device,
        )
        if selected:
            tokens = torch.tensor(selected, dtype=torch.int64, device=projection.device)
            self._run(tokens, F.embedding(tokens, embedding), self.cached)
            self.slots[tokens] = torch.arange(
                len(selected), device=projection.device, dtype=torch.int32
            )
        self.cached_rows = len(selected)

    def _run(self, tokens, embeddings, output):
        _bias_rows[(tokens.numel(), triton.cdiv(self.vocab, 64))](
            tokens,
            self.slots,
            self.cached,
            embeddings,
            self.projection,
            output,
            None,
            None,
            None,
            base_stride=0,
            GREEDY=False,
            vocab=self.vocab,
            rank=self.rank,
            emb_stride=embeddings.stride(0),
            BLOCK=64,
            num_warps=4,
        )
        return output

    def __call__(self, tokens, embeddings):
        if not self.capacity:
            return F.linear(embeddings.to(torch.float16), self.projection)
        output = torch.empty(
            (tokens.numel(), self.vocab),
            dtype=torch.float16,
            device=self.projection.device,
        )
        return self._run(tokens, embeddings, output)

    def greedy(self, tokens, embeddings, base):
        """Select exactly the FP32 base + FP16 bias argmax without a bias tensor."""
        if not self.capacity:
            return (base.float() + self(tokens, embeddings).float()).argmax(-1)
        rows = tokens.numel()
        blocks = triton.cdiv(self.vocab, 64)
        scores = torch.empty((rows, blocks), device=base.device, dtype=torch.float32)
        indices = torch.empty((rows, blocks), device=base.device, dtype=torch.int32)
        output = torch.empty(rows, device=base.device, dtype=torch.int64)
        _bias_rows[(rows, blocks)](
            tokens,
            self.slots,
            self.cached,
            embeddings,
            self.projection,
            None,
            base,
            scores,
            indices,
            base_stride=base.stride(0),
            GREEDY=True,
            vocab=self.vocab,
            rank=self.rank,
            emb_stride=embeddings.stride(0),
            BLOCK=64,
            num_warps=4,
        )
        _finish_argmax[(rows,)](
            scores,
            indices,
            output,
            blocks=blocks,
            BLOCK=triton.next_power_of_2(blocks),
        )
        return output
