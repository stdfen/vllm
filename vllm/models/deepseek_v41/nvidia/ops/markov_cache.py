# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark Markov bias with cached rows for frequent tokens (DeepSeek-V4.1).

The Markov bias of a draft position depends only on the previous token:
``bias(t) = markov_w2 @ markov_w1[t]``, a full-vocabulary row read through the
whole replicated 129,280 x 256 projection (66 MB) at every draft position. Rows
for the most frequent tokens are computed once at load and read back instead
(258 KB a row). Cached and computed rows come from the same kernel, so a token's
bias has the same bits either way. TensorFold's V4.1 drafter caches its bias rows
the same way (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
src/tensorfold/families/deepseek_v41/cuda/markov.py, Apache-2.0); this is our
implementation.

VLLM_DSPARK_MARKOV_CACHE: a JSON list of token ids to cache (most frequent
first); VLLM_DSPARK_MARKOV_CACHE_ROWS caps how many are used (default 4096).
"""

import json
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_BLOCK_V = 64


@triton.jit
def _markov_bias_kernel(
    ids_ptr,
    slot_ptr,
    cache_ptr,
    emb_ptr,
    w2_ptr,
    out_ptr,
    V,
    cache_stride,
    w2_stride,
    emb_stride,
    out_stride,
    R: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < V
    tok = tl.load(ids_ptr + b).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    if slot >= 0:
        row = tl.load(cache_ptr + slot.to(tl.int64) * cache_stride + cols, mask=mask)
        tl.store(out_ptr + b * out_stride + cols, row, mask=mask)
    else:
        r = tl.arange(0, R)
        e = tl.load(emb_ptr + b * emb_stride + r).to(tl.float32)
        w = tl.load(
            w2_ptr + cols[:, None].to(tl.int64) * w2_stride + r[None, :],
            mask=mask[:, None],
            other=0.0,
        )
        acc = tl.sum(w.to(tl.float32) * e[None, :], axis=1)
        tl.store(
            out_ptr + b * out_stride + cols,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def markov_bias_rows(
    token_ids: torch.Tensor,
    slot_of_token: torch.Tensor,
    cache: torch.Tensor,
    markov_embed: torch.Tensor,
    w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """out[b] = cache[slot_of_token[t_b]] if cached, else w2 @ markov_embed[b]."""
    num_rows, rank = markov_embed.shape
    vocab = w2.shape[0]
    assert w2.stride(1) == 1 and markov_embed.stride(1) == 1 and out.stride(1) == 1
    if num_rows == 0:
        return out
    grid = (num_rows, triton.cdiv(vocab, _BLOCK_V))
    _markov_bias_kernel[grid](
        token_ids,
        slot_of_token,
        cache,
        markov_embed,
        w2,
        out,
        vocab,
        cache.stride(0),
        w2.stride(0),
        markov_embed.stride(0),
        out.stride(0),
        R=rank,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
    )
    return out


class MarkovBiasCache:
    """Cached Markov bias rows for the tokens listed in VLLM_DSPARK_MARKOV_CACHE."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, out_dtype: torch.dtype):
        path = os.environ["VLLM_DSPARK_MARKOV_CACHE"]
        rows = int(os.environ.get("VLLM_DSPARK_MARKOV_CACHE_ROWS") or 4096)
        with open(path) as f:
            ids = [int(t) for t in json.load(f)]
        vocab, rank = w2.shape
        seen: set[int] = set()
        ids = [t for t in ids if 0 <= t < vocab and not (t in seen or seen.add(t))]
        ids = ids[:rows]
        dev = w2.device
        self.w2 = w2
        self.out_dtype = out_dtype
        self.slot_of_token = torch.full((vocab,), -1, dtype=torch.int32, device=dev)
        self.cache = torch.empty((len(ids), vocab), dtype=out_dtype, device=dev)
        if ids:
            tok = torch.tensor(ids, dtype=torch.int64, device=dev)
            # The rows come from the same kernel as uncached tokens (all slots -1
            # while filling), so a token's bias has the same bits either way.
            for s in range(0, len(ids), 256):
                t = tok[s : s + 256]
                markov_bias_rows(
                    t,
                    self.slot_of_token,
                    self.cache,
                    w1[t].contiguous(),
                    w2,
                    self.cache[s : s + 256],
                )
            self.slot_of_token[tok] = torch.arange(
                len(ids), dtype=torch.int32, device=dev
            )
        logger.info(
            "DSpark Markov bias: %d cached rows (%.2f GiB) from %s",
            len(ids),
            self.cache.numel() * self.cache.element_size() / 2**30,
            path,
        )

    def __call__(self, token_ids: torch.Tensor, markov_embed: torch.Tensor):
        out = torch.empty(
            (markov_embed.shape[0], self.w2.shape[0]),
            dtype=self.out_dtype,
            device=markov_embed.device,
        )
        return markov_bias_rows(
            token_ids,
            self.slot_of_token,
            self.cache,
            markov_embed.contiguous(),
            self.w2,
            out,
        )
