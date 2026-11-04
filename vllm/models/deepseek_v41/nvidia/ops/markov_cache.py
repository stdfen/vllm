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

``MarkovBiasCache.argmax_pair`` (VLLM_DSPARK_MARKOV_FUSED=1 in the local draft
argmax): the bias, the add to the base logits and the shard's max in two launches,
a (max, first argmax) per 128 columns then a per-row reduction to the (value,
global index) pair the all-gather sends, instead of the bias kernel, the add,
torch.max and the pair's stacking. Same arithmetic and rounding points, ties to
the lowest index as torch.max: the drafts are the unfused path's bit for bit.

``MarkovBiasCache.argmax_record`` (VLLM_DSPARK_CONF_FEAT_HEAD): the same two launches
also keep each row's second-largest value (a tie at the max counts twice, as
torch.topk) and its logsumexp over the shard, a [rows, 4] (max, global index,
second, logsumexp) record whose first two columns are argmax_pair's bit for bit.

``w2`` may be a contiguous row range of markov_w2 (one TP rank's vocabulary
shard, for the local draft argmax): rows are then that range's bias, still keyed
by the previous token over the whole vocabulary (``w1``'s rows).
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


_ARGMAX_BLOCK_V = 128


@triton.jit
def _markov_argmax_partial_kernel(
    ids_ptr,
    slot_ptr,
    cache_ptr,
    emb_ptr,
    w2_ptr,
    base_ptr,
    pval_ptr,
    pidx_ptr,
    psec_ptr,
    psum_ptr,
    V,
    cache_stride,
    w2_stride,
    emb_stride,
    base_stride,
    nblk,
    R: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUB_V: tl.constexpr,
    FEAT: tl.constexpr,
):
    b = tl.program_id(0)
    blk = tl.program_id(1)
    tok = tl.load(ids_ptr + b).to(tl.int64)
    slot = tl.load(slot_ptr + tok)
    best_v = float("-inf")
    best_i = V
    sec_v = float("-inf")  # FEAT: the block's second largest
    run_s = 0.0  # FEAT: sum of exp(x - best_v) over the block so far
    r = tl.arange(0, R)
    for s in tl.static_range(0, BLOCK_V, SUB_V):
        cols = blk * BLOCK_V + s + tl.arange(0, SUB_V)
        mask = cols < V
        if slot >= 0:
            bias = tl.load(
                cache_ptr + slot.to(tl.int64) * cache_stride + cols, mask=mask, other=0.0
            )
        else:
            # _markov_bias_kernel's arithmetic, rounded to the cached rows' dtype
            e = tl.load(emb_ptr + b * emb_stride + r).to(tl.float32)
            w = tl.load(
                w2_ptr + cols[:, None].to(tl.int64) * w2_stride + r[None, :],
                mask=mask[:, None],
                other=0.0,
            )
            bias = tl.sum(w.to(tl.float32) * e[None, :], axis=1).to(
                cache_ptr.dtype.element_ty
            )
        base = tl.load(base_ptr + b * base_stride + cols, mask=mask, other=0.0)
        # torch's add of the two: fp32 arithmetic, rounded to the logits' dtype
        x = (base.to(tl.float32) + bias.to(tl.float32)).to(base_ptr.dtype.element_ty)
        x = tl.where(mask, x.to(tl.float32), float("-inf"))
        m = tl.max(x, axis=0)
        i = tl.min(tl.where(x == m, cols, V), axis=0)
        if FEAT:
            n_m = tl.sum((x == m).to(tl.int32), axis=0)
            m2 = tl.where(n_m > 1, m, tl.max(tl.where(x == m, float("-inf"), x), axis=0))
            # the top two of (best_v, sec_v) and (m, m2)
            sec_v = tl.maximum(tl.minimum(best_v, m), tl.maximum(sec_v, m2))
            new_v = tl.maximum(best_v, m)
            sub_s = tl.sum(tl.where(x > float("-inf"), tl.exp(x - new_v), 0.0), axis=0)
            run_s = tl.where(
                new_v > float("-inf"), run_s * tl.exp(best_v - new_v) + sub_s, 0.0
            )
        take = m > best_v  # strictly: an earlier sub-block keeps a tie
        best_i = tl.where(take, i, best_i)
        best_v = tl.where(take, m, best_v)
    tl.store(pval_ptr + b * nblk + blk, best_v)
    tl.store(pidx_ptr + b * nblk + blk, best_i)
    if FEAT:
        tl.store(psec_ptr + b * nblk + blk, sec_v)
        tl.store(psum_ptr + b * nblk + blk, run_s)


@triton.jit
def _markov_argmax_final_kernel(
    pval_ptr, pidx_ptr, psec_ptr, psum_ptr, pair_ptr, nblk, lo,
    NB: tl.constexpr, W: tl.constexpr, FEAT: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.arange(0, NB)
    mask = j < nblk
    v = tl.load(pval_ptr + b * nblk + j, mask=mask, other=float("-inf"))
    i = tl.load(pidx_ptr + b * nblk + j, mask=mask, other=2147483647)
    m = tl.max(v, axis=0)
    k = tl.min(tl.where(v == m, i, 2147483647), axis=0)
    tl.store(pair_ptr + b * W, m)
    tl.store(pair_ptr + b * W + 1, (k + lo).to(tl.float32))
    if FEAT:
        n_m = tl.sum((v == m).to(tl.int32), axis=0)
        sec = tl.where(n_m > 1, m, tl.max(tl.where(v == m, float("-inf"), v), axis=0))
        sec = tl.maximum(sec, tl.max(tl.load(psec_ptr + b * nblk + j, mask=mask, other=float("-inf")), axis=0))
        ps = tl.load(psum_ptr + b * nblk + j, mask=mask, other=0.0)
        tot = tl.sum(tl.where(v > float("-inf"), ps * tl.exp(v - m), 0.0), axis=0)
        tl.store(pair_ptr + b * W + 2, sec)
        tl.store(pair_ptr + b * W + 3, m + tl.log(tot))


def markov_argmax_pair(
    token_ids: torch.Tensor,
    slot_of_token: torch.Tensor,
    cache: torch.Tensor,
    markov_embed: torch.Tensor,
    w2: torch.Tensor,
    base_local: torch.Tensor,
    vocab: int,
    lo: int,
    record: bool = False,
) -> torch.Tensor:
    """[rows, 2] fp32: (max, lo + first argmax) of base_local[:, :vocab] + bias per
    row, the bias as markov_bias_rows gives it; with `record`, [rows, 4]: the same
    two columns, then the row's second-largest value and its logsumexp."""
    num_rows, rank = markov_embed.shape
    assert w2.stride(1) == 1 and markov_embed.stride(1) == 1
    assert base_local.stride(1) == 1
    nblk = triton.cdiv(vocab, _ARGMAX_BLOCK_V)
    dev = base_local.device
    width = 4 if record else 2
    pval = torch.empty((num_rows, nblk), dtype=torch.float32, device=dev)
    pidx = torch.empty((num_rows, nblk), dtype=torch.int32, device=dev)
    psec = torch.empty((num_rows, nblk) if record else (1,), dtype=torch.float32, device=dev)
    psum = torch.empty((num_rows, nblk) if record else (1,), dtype=torch.float32, device=dev)
    pair = torch.empty((num_rows, width), dtype=torch.float32, device=dev)
    if num_rows == 0:
        return pair
    _markov_argmax_partial_kernel[(num_rows, nblk)](
        token_ids,
        slot_of_token,
        cache,
        markov_embed,
        w2,
        base_local,
        pval,
        pidx,
        psec,
        psum,
        vocab,
        cache.stride(0),
        w2.stride(0),
        markov_embed.stride(0),
        base_local.stride(0),
        nblk,
        R=rank,
        BLOCK_V=_ARGMAX_BLOCK_V,
        SUB_V=_BLOCK_V,
        FEAT=record,
        num_warps=4,
    )
    _markov_argmax_final_kernel[(num_rows,)](
        pval, pidx, psec, psum, pair, nblk, lo,
        NB=triton.next_power_of_2(nblk), W=width, FEAT=record, num_warps=1,
    )
    return pair


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
        vocab = w1.shape[0]                   # previous-token vocabulary (the keys)
        out_vocab = w2.shape[0]               # bias columns: the whole vocab or a shard
        seen: set[int] = set()
        ids = [t for t in ids if 0 <= t < vocab and not (t in seen or seen.add(t))]
        ids = ids[:rows]
        dev = w2.device
        self.w2 = w2
        self.out_dtype = out_dtype
        self.slot_of_token = torch.full((vocab,), -1, dtype=torch.int32, device=dev)
        self.cache = torch.empty((len(ids), out_vocab), dtype=out_dtype, device=dev)
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

    def argmax_pair(
        self,
        token_ids: torch.Tensor,
        markov_embed: torch.Tensor,
        base_local: torch.Tensor,
        vocab: int,
        lo: int,
    ) -> torch.Tensor:
        """(max, lo + argmax) of base_local[:, :vocab] + this cache's bias, per row."""
        return markov_argmax_pair(
            token_ids,
            self.slot_of_token,
            self.cache,
            markov_embed.contiguous(),
            self.w2,
            base_local,
            vocab,
            lo,
        )

    def argmax_record(
        self,
        token_ids: torch.Tensor,
        markov_embed: torch.Tensor,
        base_local: torch.Tensor,
        vocab: int,
        lo: int,
    ) -> torch.Tensor:
        """argmax_pair's two columns, then the second-largest value and the logsumexp
        of base_local[:, :vocab] + this cache's bias, per row: [rows, 4] fp32."""
        return markov_argmax_pair(
            token_ids,
            self.slot_of_token,
            self.cache,
            markov_embed.contiguous(),
            self.w2,
            base_local,
            vocab,
            lo,
            record=True,
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
