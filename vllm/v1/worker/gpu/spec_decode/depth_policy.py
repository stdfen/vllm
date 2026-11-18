# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round DSpark verify depth inside a fixed-shape verify window.

After each draft, every request gets the verify depth d in 0..k with the most
expected tokens per millisecond: expected tokens = 1 + the sum of the
confidence head's prefix survivals (running products of its per-position
acceptance probabilities), cost = the verify forward with d + 1 live rows plus
the drafter pass, from a measured table. The rule is TensorFold's DeepSeek-V4.1
depth choice (github.com/bertholomus/TensorFold, branch deepseek-v41-tp2,
src/tensorfold/families/deepseek_v41/cuda/multi.py ``_choose_k``, Apache-2.0),
written here for the V2 model runner.

The window keeps its shape, so captured CUDA graphs and attention backends are
untouched. Rows past a request's depth are marked as MoE padding (the router
gives them top-k ids of -1 and routed-expert kernels skip their weights) and
as invalid drafts (the rejection sampler never accepts them). Rows up to the
depth compute what a full window computes for them (attention is causal, MoE
and dense rows are independent), so the depth decides how many drafts can be
accepted and never which tokens come out.

Environment:
  VLLM_DSPARK_DEPTH_POLICY  "conf" (the rule above), "fixed:<d>" (every request
                            verifies d drafts) or unset / "0" (off)
  VLLM_DSPARK_ROUND_MS      verify forward ms for 1..k+1 live rows, comma separated
  VLLM_DSPARK_DRAFT_MS      the drafter pass in ms
  VLLM_DSPARK_DEPTH_FILE    a file holding a policy string, re-read every 64
                            verify steps (to sweep depths in one serve);
                            "conf:<ms,...>" also replaces the cost table (round
                            ms by depth 0..k, drafter included)
  VLLM_DSPARK_DEPTH_LOG     N > 0: every N verify steps, log the chosen depths and
                            the drafts verified / accepted
"""

import math
import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Our two-Spark serve (DeepSeek-V4.1-Flash, EXL3 routed experts): verify forward
# ms for 1..6 live rows and the drafter pass, until a measured table is given.
_DEFAULT_ROUND_MS = (38.0, 41.0, 44.0, 46.5, 48.5, 50.5)
_DEFAULT_DRAFT_MS = 4.5
_FILE_EVERY = 64


def _parse_policy(text: str | None) -> tuple[str, int]:
    p = (text or "").strip().lower()
    if p in ("", "0", "off", "none"):
        return "off", 0
    if p == "conf" or p.startswith("conf:"):
        return "conf", 0
    if p == "conf+feat" or p.startswith("conf+feat:"):
        if os.environ.get("VLLM_DSPARK_CONF_FEAT_HEAD"):
            raise ValueError("DSpark fitted feature head is not yet ported")
        return "conf", 0
    if p.startswith("fixed:"):
        return "fixed", int(p.split(":", 1)[1])
    raise ValueError(
        f"VLLM_DSPARK_DEPTH_POLICY={text!r}: expected 'conf', 'fixed:<d>' or '0'"
    )


def _parse_table(text: str | None) -> list[float] | None:
    """The round-cost table of a "conf:<ms,ms,...>" policy string (depth 0 first)."""
    p = (text or "").strip().lower()
    if not p.startswith(("conf:", "conf+feat:")):
        return None
    return [float(x) for x in p.split(":", 1)[1].split(",") if x.strip()]


def depth_policy_requested() -> bool:
    return _parse_policy(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))[0] != "off"


def round_cost_ms(num_steps: int) -> list[float]:
    """Verify forward + drafter ms for depth 0..num_steps (1..num_steps+1 rows)."""
    text = os.environ.get("VLLM_DSPARK_ROUND_MS", "")
    rows = [float(x) for x in text.split(",") if x.strip()] or list(_DEFAULT_ROUND_MS)
    if len(rows) != num_steps + 1:
        raise ValueError("DSpark round cost table must contain k+1 entries")
    draft_ms = float(os.environ.get("VLLM_DSPARK_DRAFT_MS") or _DEFAULT_DRAFT_MS)
    costs = [rows[d] + draft_ms for d in range(num_steps + 1)]
    if not math.isfinite(draft_ms) or draft_ms < 0 or any(
        not math.isfinite(c) or c <= 0 for c in rows + costs
    ):
        raise ValueError("DSpark costs must be finite and positive")
    return costs


@triton.jit
def _choose_depth_kernel(conf_ptr, conf_stride, cost_ptr, out_ptr, K: tl.constexpr):
    row = tl.program_id(0)
    best_d = 0
    best_v = tl.math.div_rn(1.0, tl.load(cost_ptr))
    survival = 1.0
    cumulative = 0.0
    for d in tl.static_range(1, K + 1):
        c = tl.load(conf_ptr + row * conf_stride + d - 1).to(tl.float32)
        survival *= tl.minimum(tl.maximum(c, 0.0), 1.0)
        cumulative += survival
        value = tl.math.div_rn(1.0 + cumulative, tl.load(cost_ptr + d))
        take = value > best_v
        best_d = tl.where(take, d, best_d)
        best_v = tl.where(take, value, best_v)
    tl.store(out_ptr + row, best_d)


@triton.jit
def _dead_rows_kernel(idx_ptr, pos_ptr, depth_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    idx = tl.load(idx_ptr + i, mask=mask, other=0)
    pos = tl.load(pos_ptr + i, mask=mask, other=0)
    dep = tl.load(depth_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + i, pos > dep, mask=mask)


def choose_depth(confidence_probs: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """[n, k] per-position acceptance probabilities -> [n] int32 depths in 0..k."""
    conf = confidence_probs.float().clamp(0.0, 1.0)
    survival = torch.cumprod(conf, dim=1)
    expected = torch.cat(
        (torch.ones_like(conf[:, :1]), 1.0 + torch.cumsum(survival, dim=1)), dim=1
    )
    # argmax keeps the first (shallowest) depth among equal rates.
    return torch.argmax(expected / cost_ms, dim=1).to(torch.int32)


class DSparkDepthPolicy:
    def __init__(
        self,
        num_speculative_steps: int,
        max_num_reqs: int,
        device: torch.device,
        has_confidence: bool,
    ):
        unsupported = ("VLLM_DSPARK_CONF_TEMPS", "VLLM_DSPARK_CONF_BIAS",
                       "VLLM_DSPARK_CONF_LOG", "VLLM_DSPARK_CONF_LOG_TAG")
        if any(os.environ.get(name) for name in unsupported):
            raise ValueError("DSpark confidence calibration/logging is not yet ported")
        self.k = num_speculative_steps
        self._has_conf = has_confidence
        self.mode, self.fixed = _parse_policy(
            os.environ.get("VLLM_DSPARK_DEPTH_POLICY")
        )
        if self.mode == "conf" and not has_confidence:
            raise ValueError(
                "VLLM_DSPARK_DEPTH_POLICY=conf needs the draft model's confidence "
                "head, which this checkpoint does not load."
            )
        if self.k <= 0 or not 0 <= self.fixed <= self.k:
            raise ValueError("DSpark requires k>0 and fixed depth within 0..k")
        cost = _parse_table(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))
        if cost is None:
            cost = round_cost_ms(self.k)
        self._validate_table(cost)
        self.cost_ms = torch.tensor(cost, dtype=torch.float32, device=device)
        self.depth = torch.full(
            (max_num_reqs,), self.k, dtype=torch.int32, device=device
        )
        self.file = os.environ.get("VLLM_DSPARK_DEPTH_FILE") or None
        self._reload_pending = False
        self.log_every = int(os.environ.get("VLLM_DSPARK_DEPTH_LOG") or 0)
        self.num_records = 0
        self._observed = 0
        # Logging counters (device): depth histogram, drafts verified, accepted.
        self._hist = torch.zeros(self.k + 1, dtype=torch.int64, device=device)
        self._live = torch.zeros((), dtype=torch.int64, device=device)
        self._kept = torch.zeros((), dtype=torch.int64, device=device)
        logger.info(
            "DSpark depth policy %s (k=%d), round cost ms by depth %s",
            self.describe(),
            self.k,
            [round(c, 2) for c in cost],
        )

    def _validate_table(self, table: list[float]) -> None:
        if len(table) != self.k + 1 or any(
            not math.isfinite(c) or c <= 0 for c in table
        ):
            raise ValueError("DSpark requires k+1 finite positive round costs")

    def describe(self) -> str:
        return f"fixed:{self.fixed}" if self.mode == "fixed" else self.mode

    def reset(self, req_index: int) -> None:
        self.depth[req_index] = self.k
        if self.file is not None:
            self._reload_pending = True

    def _maybe_reload(self) -> None:
        """Measurement knob: take the policy in VLLM_DSPARK_DEPTH_FILE. Rank 0 reads
        the file and broadcasts it, so every rank switches at the same step (ranks
        that chose different depths would accept different tokens)."""
        if self.file is None or (self.num_records % _FILE_EVERY and not self._reload_pending):
            return
        self._reload_pending = False
        from vllm.distributed.parallel_state import get_tp_group

        try:
            tp = get_tp_group()
        except AssertionError:  # no model parallel group (single-process tests)
            tp = None
        text = None
        if tp is None or tp.rank_in_group == 0:
            try:
                with open(self.file) as f:
                    text = f.read()
            except OSError:
                text = None
        if tp is not None and tp.world_size > 1:
            text = tp.broadcast_object(text, src=0)
        if text is None:
            return
        if text.strip().lower().startswith("conf+feat") and os.environ.get("VLLM_DSPARK_CONF_FEAT_HEAD"):
            raise ValueError("DSpark fitted feature head is not yet ported")
        if text.strip().lower().startswith("conf+feat"):
            logger.warning_once("DSpark conf+feat without a feature head uses raw confidence, as historical V36")
        try:
            mode, fixed = _parse_policy(text)
        except ValueError as e:
            logger.warning("DSpark depth file %s ignored: %s", self.file, e)
            return
        if mode == "conf" and not self._has_conf:
            logger.warning("DSpark depth file %s: no confidence head", self.file)
            return
        table = _parse_table(text)
        if not 0 <= fixed <= self.k:
            raise ValueError("DSpark depth file: fixed depth outside 0..k")
        if table is not None:
            self._validate_table(table)
            new = torch.tensor(table[: self.k + 1], dtype=torch.float32, device=self.cost_ms.device)
            if not torch.equal(new, self.cost_ms):
                self.cost_ms.copy_(new)
                logger.info("DSpark depth cost table -> %s", [round(c, 2) for c in table[: self.k + 1]])
        if (mode, fixed) != (self.mode, self.fixed):
            self.mode, self.fixed = mode, fixed
            logger.info("DSpark depth policy -> %s", self.describe())

    def record(
        self,
        confidence_probs: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Choose each request's depth for the drafts just proposed."""
        if num_reqs == 0:
            return
        self._maybe_reload()
        self.num_records += 1
        if self.mode == "off":
            depth = torch.full(
                (num_reqs,), self.k, dtype=torch.int32, device=self.depth.device
            )
        elif self.mode == "fixed":
            depth = torch.full(
                (num_reqs,),
                min(self.fixed, self.k),
                dtype=torch.int32,
                device=self.depth.device,
            )
        else:
            conf = confidence_probs[:num_reqs]
            if (os.environ.get("VLLM_DSPARK_DEPTH_FUSED", "0") == "1"
                and conf.is_cuda and conf.dtype == torch.float32
                and conf.stride(1) == 1 and conf.shape[1] == self.k):
                depth = torch.empty(num_reqs, dtype=torch.int32, device=conf.device)
                _choose_depth_kernel[(num_reqs,)](
                    conf, conf.stride(0), self.cost_ms, depth, K=self.k
                )
            else:
                depth = choose_depth(conf, self.cost_ms)
        # All TP ranks must reject at the same boundary even if confidence
        # rounding differs across devices. This runs outside captured graphs.
        from vllm.distributed.parallel_state import get_tp_group

        try:
            tp = get_tp_group()
        except AssertionError:
            tp = None
        if tp is not None and tp.world_size > 1:
            depth = tp.broadcast(depth.contiguous(), src=0)
        self.depth.index_copy_(0, idx_mapping[:num_reqs].long(), depth)

    def dead_rows(
        self, expanded_idx_mapping: torch.Tensor, expanded_local_pos: torch.Tensor
    ) -> torch.Tensor:
        """Logits rows past their request's depth: local position p holds draft p."""
        n = expanded_local_pos.numel()
        if (os.environ.get("VLLM_DSPARK_DEPTH_FUSED", "0") == "1"
            and expanded_local_pos.is_cuda and n > 0
            and expanded_idx_mapping.is_contiguous()
            and expanded_local_pos.is_contiguous()):
            out = torch.empty(n, dtype=torch.bool, device=expanded_local_pos.device)
            _dead_rows_kernel[(triton.cdiv(n, 256),)](
                expanded_idx_mapping, expanded_local_pos, self.depth, out, n, BLOCK=256
            )
            return out
        return expanded_local_pos > self.depth[expanded_idx_mapping.long()]

    def observe(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_drafts_per_req: np.ndarray | None,
        num_reqs: int,
    ) -> None:
        """Count drafts verified and accepted; log every VLLM_DSPARK_DEPTH_LOG steps."""
        if self.log_every <= 0 or num_reqs == 0 or num_drafts_per_req is None:
            return
        depth = self.depth[idx_mapping[:num_reqs].long()].long()
        drafts = torch.from_numpy(
            np.ascontiguousarray(num_drafts_per_req[:num_reqs], dtype=np.int64)
        ).to(depth.device, non_blocking=True)
        self._hist.index_add_(0, depth.clamp(max=self.k), torch.ones_like(depth))
        self._live += torch.minimum(depth, drafts).sum()
        self._kept += (num_sampled[:num_reqs].long() - 1).clamp(min=0).sum()
        self._observed += 1
        if self._observed % self.log_every == 0:
            hist = self._hist.tolist()
            live, kept, rounds = int(self._live), int(self._kept), sum(hist)
            logger.info(
                "DSpark depth %s: depths %s, drafts verified %d, accepted %d "
                "(%.1f%%), tokens per verify %.2f",
                self.describe(),
                " ".join(f"{d}:{n}" for d, n in enumerate(hist) if n),
                live,
                kept,
                100.0 * kept / max(live, 1),
                (kept + rounds) / max(rounds, 1),
            )
            self._hist.zero_()
            self._live.zero_()
            self._kept.zero_()
