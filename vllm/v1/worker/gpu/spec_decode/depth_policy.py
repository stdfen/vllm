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
  VLLM_DSPARK_DEPTH_POLICY  "conf" (the rule above), "conf+feat" (the rule on the
                            confidences fitted with the drafts' features, patch
                            0036's VLLM_DSPARK_CONF_FEAT_HEAD), "fixed:<d>" (every
                            request verifies d drafts) or unset / "0" (off)
  VLLM_DSPARK_ROUND_MS      verify forward ms for 1..k+1 live rows, comma separated
  VLLM_DSPARK_DRAFT_MS      the drafter pass in ms
  VLLM_DSPARK_DEPTH_FILE    a file holding a policy string, re-read every 64
                            verify steps and at the first verify step after a
                            request starts (to sweep depths or A/B policies in
                            one serve); "conf:<ms,...>" / "conf+feat:<ms,...>"
                            also replaces the cost table (round ms by depth 0..k,
                            drafter included)
  VLLM_DSPARK_DEPTH_LOG     N > 0: every N verify steps, log the chosen depths and
                            the drafts verified / accepted
  VLLM_DSPARK_CONF_TEMPS    "t1,...,tk": per-position temperatures for the
                            confidence head; the conf rule reads
                            sigmoid(logit(c_i) / t_i + b_i) instead of c_i (one
                            value covers every position, a short list repeats its
                            last value); unset = the raw confidences
  VLLM_DSPARK_CONF_BIAS     "b1,...,bk": per-position bias added after the
                            temperature (default 0)
  VLLM_DSPARK_CONF_LOG      a .jsonl path: rank 0 appends a header line (k, policy,
                            cost table, temperatures) and then one line per request
                            per verify round: the raw confidences (before any
                            temperature), the depth chosen, the drafts in the
                            window and the drafts accepted
  VLLM_DSPARK_CONF_LOG_TAG  a run tag (e.g. the domain) for the header line
"""

import atexit
import collections
import json
import os
import time

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
# torch.logit clamp for VLLM_DSPARK_CONF_TEMPS: fp32 confidences within 1e-6 of 0 or 1
# read as logit -/+13.8
_CONF_EPS = 1e-6


def _parse_policy(text: str | None) -> tuple[str, int]:
    p = (text or "").strip().lower()
    if p in ("", "0", "off", "none"):
        return "off", 0
    if p in ("conf", "conf+feat") or p.startswith(("conf:", "conf+feat:")):
        return "conf", 0
    if p.startswith("fixed:"):
        return "fixed", max(0, int(p.split(":", 1)[1]))
    raise ValueError(
        f"VLLM_DSPARK_DEPTH_POLICY={text!r}: expected 'conf', 'fixed:<d>' or '0'"
    )


def _parse_table(text: str | None) -> list[float] | None:
    """The round-cost table of a "conf:<ms,ms,...>" policy string (depth 0 first)."""
    p = (text or "").strip().lower()
    if not p.startswith(("conf:", "conf+feat:")):
        return None
    return [float(x) for x in p.split(":", 1)[1].split(",") if x.strip()]


def _wants_feat(text: str | None) -> bool:
    """A "conf+feat" policy string: the conf rule reads the feature-fitted confidences."""
    return (text or "").strip().lower().startswith("conf+feat")


def depth_policy_requested() -> bool:
    return _parse_policy(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))[0] != "off"


def round_cost_ms(num_steps: int) -> list[float]:
    """Verify forward + drafter ms for depth 0..num_steps (1..num_steps+1 rows)."""
    text = os.environ.get("VLLM_DSPARK_ROUND_MS", "")
    rows = [float(x) for x in text.split(",") if x.strip()] or list(_DEFAULT_ROUND_MS)
    while len(rows) < num_steps + 1:
        # Past the table: the last measured step per row.
        slope = rows[-1] - rows[-2] if len(rows) > 1 else 0.0
        rows.append(rows[-1] + slope)
    draft_ms = float(os.environ.get("VLLM_DSPARK_DRAFT_MS") or _DEFAULT_DRAFT_MS)
    return [rows[d] + draft_ms for d in range(num_steps + 1)]


def _per_position(name: str, k: int) -> list[float] | None:
    """A per-position list from a comma-separated env value: one value covers every
    position, a short list repeats its last value; unset or empty = None."""
    text = os.environ.get(name) or ""
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals:
        return None
    if len(vals) > k:
        raise ValueError(f"{name}={text!r}: {len(vals)} values for {k} draft positions")
    return vals + [vals[-1]] * (k - len(vals))


def calibrate(
    confidence_probs: torch.Tensor, inv_temp: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """[n, k] confidences -> sigmoid(logit(c) * inv_temp + bias), per position. The
    head stores sigmoid(its logit) in fp32, so this is a temperature on that logit,
    to fp32 rounding."""
    z = torch.logit(confidence_probs.float(), eps=_CONF_EPS)
    return torch.sigmoid(torch.addcmul(bias, z, inv_temp))


def _is_rank0() -> bool:
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


# VLLM_DSPARK_DEPTH_FUSED=1 (default 0): the conf rule and its write into the depth slots
# in one launch (_choose_depth_kernel) instead of ~13 small ones (clamp, two scans, cat,
# divide, argmax, casts, index_copy), and dead_rows in one (3 before); the next round's
# input prep runs these eagerly, ~4 us apart on GB10. The same rule in fp32 with a
# running product and sum (the scans' association may differ in the last bit, so a
# near-tie can pick the other depth; the depth never changes which tokens come out).
_DEPTH_FUSED = os.environ.get("VLLM_DSPARK_DEPTH_FUSED", "0") == "1"


@triton.jit
def _choose_depth_kernel(conf_ptr, conf_stride, idx_ptr, cost_ptr, depth_ptr, K: tl.constexpr):
    r = tl.program_id(0)
    best_d = 0
    best_v = tl.math.div_rn(1.0, tl.load(cost_ptr))
    surv = 1.0
    cum = 0.0
    for d in tl.static_range(1, K + 1):
        c = tl.load(conf_ptr + r * conf_stride + (d - 1)).to(tl.float32)
        surv = surv * tl.minimum(tl.maximum(c, 0.0), 1.0)
        cum = cum + surv
        v = tl.math.div_rn(1.0 + cum, tl.load(cost_ptr + d))  # IEEE, as torch's divide
        take = v > best_v  # strictly: the shallowest depth keeps a tie, as torch.argmax
        best_d = tl.where(take, d, best_d)
        best_v = tl.where(take, v, best_v)
    tl.store(depth_ptr + tl.load(idx_ptr + r), best_d.to(tl.int32))


@triton.jit
def _dead_rows_kernel(idx_ptr, pos_ptr, depth_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    idx = tl.load(idx_ptr + i, mask=m, other=0)
    pos = tl.load(pos_ptr + i, mask=m, other=0)
    dep = tl.load(depth_ptr + idx, mask=m, other=0)
    tl.store(out_ptr + i, pos > dep, mask=m)


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
        self.k = num_speculative_steps
        self._has_conf = has_confidence
        self.mode, self.fixed = _parse_policy(
            os.environ.get("VLLM_DSPARK_DEPTH_POLICY")
        )
        self.feat = _wants_feat(os.environ.get("VLLM_DSPARK_DEPTH_POLICY"))
        self._feat_missing_warned = False
        # a request started since the last read of VLLM_DSPARK_DEPTH_FILE: every rank
        # adds the same requests, so every rank reads (and broadcasts) at the same step
        self._reload_pending = False
        if self.mode == "conf" and not has_confidence:
            raise ValueError(
                "VLLM_DSPARK_DEPTH_POLICY=conf needs the draft model's confidence "
                "head, which this checkpoint does not load."
            )
        cost = round_cost_ms(self.k)
        self.cost_ms = torch.tensor(cost, dtype=torch.float32, device=device)
        self.depth = torch.full(
            (max_num_reqs,), self.k, dtype=torch.int32, device=device
        )
        self.file = os.environ.get("VLLM_DSPARK_DEPTH_FILE") or None
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
        # Confidence calibration (VLLM_DSPARK_CONF_TEMPS / _BIAS): only the conf
        # rule reads it, only the depth moves.
        self._temps = _per_position("VLLM_DSPARK_CONF_TEMPS", self.k)
        self._bias = _per_position("VLLM_DSPARK_CONF_BIAS", self.k)
        self.inv_temp: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        if self._temps is not None or self._bias is not None:
            self._temps = self._temps or [1.0] * self.k
            self._bias = self._bias or [0.0] * self.k
            if min(self._temps) <= 0:
                raise ValueError(f"VLLM_DSPARK_CONF_TEMPS={self._temps}: must be > 0")
            self.inv_temp = torch.tensor(
                [1.0 / t for t in self._temps], dtype=torch.float32, device=device
            )
            self.bias = torch.tensor(self._bias, dtype=torch.float32, device=device)
            logger.info(
                "DSpark confidence temperatures %s, bias %s", self._temps, self._bias
            )
        # VLLM_DSPARK_CONF_LOG: raw confidences by request slot, kept from the draft
        # until its verify round is observed.
        self._conf_log: _ConfLog | None = None
        log_path = os.environ.get("VLLM_DSPARK_CONF_LOG") or None
        if log_path and not has_confidence:
            logger.warning("VLLM_DSPARK_CONF_LOG ignored: no confidence head")
        elif log_path and _is_rank0():
            self._conf_raw = torch.zeros(
                (max_num_reqs, self.k), dtype=torch.float32, device=device
            )
            # VLLM_DSPARK_CONF_LOG_FEATURES=1: the drafts' margins and top-1
            # probabilities ride along, 2 values a position ("f" in each line)
            self._feats = os.environ.get("VLLM_DSPARK_CONF_LOG_FEATURES", "0") == "1"
            self._feat_raw = torch.zeros(
                (max_num_reqs, 2 * self.k), dtype=torch.float32, device=device
            )
            # VLLM_DSPARK_CONF_LOG_HIDDEN=1: the confidence head's inputs, kept by slot
            # like the confidences and written to <log>.hid (bf16, k x width a line)
            self._hid_raw: torch.Tensor | None = None
            self._conf_log = _ConfLog(
                log_path, self.k, max_num_reqs, device, feats=self._feats
            )
            self._log_header()

    def describe(self) -> str:
        if self.mode == "fixed":
            return f"fixed:{self.fixed}"
        return "conf+feat" if self.mode == "conf" and self.feat else self.mode

    def _log_header(self) -> None:
        if self._conf_log is not None:
            self._conf_log.header(
                k=self.k,
                policy=self.describe(),
                cost_ms=[round(c, 4) for c in self.cost_ms.tolist()],
                temps=self._temps,
                bias=self._bias,
                tag=os.environ.get("VLLM_DSPARK_CONF_LOG_TAG") or None,
            )

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
        try:
            mode, fixed = _parse_policy(text)
        except ValueError as e:
            logger.warning("DSpark depth file %s ignored: %s", self.file, e)
            return
        if mode == "conf" and not self._has_conf:
            logger.warning("DSpark depth file %s: no confidence head", self.file)
            return
        changed = False
        table = _parse_table(text)
        if table:
            while len(table) < self.k + 1:
                table.append(table[-1] + (table[-1] - table[-2] if len(table) > 1 else 0.0))
            new = torch.tensor(table[: self.k + 1], dtype=torch.float32, device=self.cost_ms.device)
            if not torch.equal(new, self.cost_ms):
                self.cost_ms.copy_(new)
                changed = True
                logger.info("DSpark depth cost table -> %s", [round(c, 2) for c in table[: self.k + 1]])
        feat = _wants_feat(text)
        if (mode, fixed, feat) != (self.mode, self.fixed, self.feat):
            self.mode, self.fixed, self.feat = mode, fixed, feat
            changed = True
            logger.info("DSpark depth policy -> %s", self.describe())
        if changed:
            self._log_header()

    def record(
        self,
        confidence_probs: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        feats: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        fitted: torch.Tensor | None = None,
    ) -> None:
        """Choose each request's depth for the drafts just proposed. `feats`
        ([reqs, k, 2], VLLM_DSPARK_CONF_LOG_FEATURES) is only logged; `fitted`
        ([reqs, k], patch 0036) replaces the confidences under "conf+feat"."""
        if num_reqs == 0:
            return
        self._maybe_reload()
        self.num_records += 1
        if self._conf_log is not None:
            self._conf_raw.index_copy_(
                0, idx_mapping[:num_reqs].long(), confidence_probs[:num_reqs].float()
            )
            if self._feats and feats is not None:
                self._feat_raw.index_copy_(
                    0,
                    idx_mapping[:num_reqs].long(),
                    feats[:num_reqs].reshape(num_reqs, -1).float(),
                )
            if hidden is not None:
                if self._hid_raw is None:
                    self._hid_raw = torch.zeros(
                        (self.depth.shape[0],) + tuple(hidden.shape[1:]),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                self._hid_raw.index_copy_(
                    0, idx_mapping[:num_reqs].long(), hidden[:num_reqs]
                )
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
            if self.feat:
                if fitted is not None:
                    conf = fitted[:num_reqs]
                elif not self._feat_missing_warned:
                    self._feat_missing_warned = True
                    logger.warning(
                        "DSpark depth policy conf+feat without fitted confidences "
                        "(VLLM_DSPARK_CONF_FEAT_HEAD unset): the raw ones are used"
                    )
            if self.inv_temp is not None:
                conf = calibrate(conf, self.inv_temp, self.bias)
            if (
                _DEPTH_FUSED
                and conf.dtype == torch.float32
                and conf.stride(1) == 1
                and idx_mapping.dtype == torch.int32
                and self.cost_ms.numel() == self.k + 1
                and conf.shape[1] == self.k
            ):
                _choose_depth_kernel[(num_reqs,)](
                    conf, conf.stride(0), idx_mapping, self.cost_ms, self.depth, K=self.k
                )
                return
            depth = choose_depth(conf, self.cost_ms)
        self.depth.index_copy_(0, idx_mapping[:num_reqs].long(), depth)

    def dead_rows(
        self, expanded_idx_mapping: torch.Tensor, expanded_local_pos: torch.Tensor
    ) -> torch.Tensor:
        """Logits rows past their request's depth: local position p holds draft p."""
        n = expanded_local_pos.numel()
        if (
            _DEPTH_FUSED
            and n > 0
            and expanded_idx_mapping.dtype == torch.int32
            and expanded_local_pos.dtype == torch.int32
            and expanded_idx_mapping.is_contiguous()
            and expanded_local_pos.is_contiguous()
        ):
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
        req_ids: list[str] | None = None,
    ) -> None:
        """Count drafts verified and accepted; log every VLLM_DSPARK_DEPTH_LOG steps.
        With VLLM_DSPARK_CONF_LOG, queue this verify round's lines (rank 0)."""
        if num_reqs == 0 or num_drafts_per_req is None:
            return
        if self._conf_log is not None and req_ids is not None:
            idx = idx_mapping[:num_reqs].long()
            self._conf_log.add(
                torch.cat(
                    (
                        self._conf_raw[idx],
                        self.depth[idx].unsqueeze(1).float(),
                        num_sampled[:num_reqs].unsqueeze(1).float(),
                    )
                    + ((self._feat_raw[idx],) if self._feats else ()),
                    dim=1,
                ),
                list(req_ids[:num_reqs]),
                np.array(num_drafts_per_req[:num_reqs], dtype=np.int64),
                hidden=None if self._hid_raw is None else self._hid_raw[idx],
            )
        if self.log_every <= 0:
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


class _ConfLog:
    """VLLM_DSPARK_CONF_LOG writer (rank 0). A verify step's rows (raw confidences,
    depth, tokens sampled) go to a pinned host ring with one async copy and an event;
    their lines are formatted once the event has completed and written in batches,
    so a step waits on the GPU only if all 8 slots still have copies in flight.
    Lines reach the file within about a second; the last step before the serve goes
    idle is written at the next step or at exit."""

    SLOTS = 8
    FLUSH_LINES = 256
    FLUSH_S = 1.0

    def __init__(
        self,
        path: str,
        k: int,
        max_num_reqs: int,
        device: torch.device,
        feats: bool = False,
    ):
        self.k = k
        self.feats = feats
        self.path = path
        self.file = open(path, "a")
        # the confidence head's inputs (VLLM_DSPARK_CONF_LOG_HIDDEN): <path>.hid, one
        # record of k x width bf16 a line in order; "h" in the line is its index
        self.hid_file = None
        self.hid_host: torch.Tensor | None = None
        self.hid_count = 0
        self.max_num_reqs = max_num_reqs
        self.cuda = device.type == "cuda"
        cuda = device.type == "cuda"
        self.host = torch.empty(
            (self.SLOTS, max_num_reqs, k + 2 + (2 * k if feats else 0)),
            dtype=torch.float32,
            pin_memory=cuda,
        )
        self.events = [torch.cuda.Event() if cuda else None for _ in range(self.SLOTS)]
        self.pending: collections.deque = collections.deque()
        self.lines: list[str] = []
        self.steps = 0
        self.flushed_at = time.monotonic()
        atexit.register(self.close)
        logger.info("DSpark confidence log -> %s", path)

    def header(self, **fields) -> None:
        self.drain(wait=True)  # earlier rounds keep their place before the header
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append(json.dumps({"hdr": 1, "time": stamp, **fields}))
        self.flush()

    def add(
        self,
        rows: torch.Tensor,
        req_ids: list[str],
        drafts: np.ndarray,
        hidden: torch.Tensor | None = None,
    ) -> None:
        if len(self.pending) == self.SLOTS:
            # ring full: the oldest copy holds this step's slot
            event = self.events[self.pending[0][0]]
            if event is not None:
                event.synchronize()
            self._pop()
        slot = self.steps % self.SLOTS
        event = self.events[slot]
        self.host[slot, : rows.shape[0]].copy_(rows, non_blocking=event is not None)
        if hidden is not None:
            if self.hid_host is None:
                self.hid_host = torch.empty(
                    (self.SLOTS, self.max_num_reqs) + tuple(hidden.shape[1:]),
                    dtype=hidden.dtype,
                    pin_memory=self.cuda,
                )
                self.hid_file = open(self.path + ".hid", "ab")
            self.hid_host[slot, : hidden.shape[0]].copy_(
                hidden, non_blocking=event is not None
            )
        if event is not None:
            event.record()
        self.pending.append((slot, rows.shape[0], self.steps, req_ids, drafts))
        self.steps += 1
        self.drain()

    def _pop(self) -> None:
        slot, n, step, req_ids, drafts = self.pending.popleft()
        k = self.k
        for rid, nd, row in zip(req_ids, drafts.tolist(), self.host[slot, :n].tolist()):
            if nd <= 0:
                continue  # nothing verified for this request this step
            c = ",".join(f"{x:.7g}" for x in row[:k])
            f = ""
            if self.feats:
                f = ',"f":[' + ",".join(f"{x:.6g}" for x in row[k + 2 :]) + "]"
            if self.hid_file is not None:
                j = req_ids.index(rid)
                self.hid_file.write(
                    self.hid_host[slot, j].contiguous().view(torch.int16).numpy().tobytes()
                )
                f += f',"h":{self.hid_count}'
                self.hid_count += 1
            self.lines.append(
                f'{{"q":{json.dumps(rid)},"s":{step},"c":[{c}],'
                f'"d":{int(row[k])},"n":{nd},"a":{int(row[k + 1]) - 1}{f}}}'
            )

    def drain(self, wait: bool = False) -> None:
        while self.pending:
            event = self.events[self.pending[0][0]]
            if event is not None and not event.query():
                if not wait:
                    break
                event.synchronize()
            self._pop()
        if len(self.lines) >= self.FLUSH_LINES or (
            self.lines and time.monotonic() - self.flushed_at >= self.FLUSH_S
        ):
            self.flush()

    def flush(self) -> None:
        if self.lines:
            self.file.write("\n".join(self.lines) + "\n")
            self.file.flush()
            if self.hid_file is not None:
                self.hid_file.flush()
            self.lines.clear()
        self.flushed_at = time.monotonic()

    def close(self) -> None:
        try:
            self.drain(wait=True)
            self.flush()
            self.file.close()
            if self.hid_file is not None:
                self.hid_file.close()
        except Exception:  # interpreter shutdown: CUDA may already be gone
            pass
