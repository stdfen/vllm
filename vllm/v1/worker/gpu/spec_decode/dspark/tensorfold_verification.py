# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prefix policy; shares vLLM's varlen verification ABI, not its allocator."""

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.dspark.prefix_policy import select_prefixes

logger = init_logger(__name__)


class TensorFoldVerificationManager(AdaptiveVerificationManager):
    """Choose exact host-known prefix lengths before selecting a target graph.

    Uses the latest completed draft's confidence, never the native allocator's
    stale estimate. The D2H event wait and TP broadcast are intentional costs of
    this experimental implementation and must be included in A-B-A timings.
    """

    host_compaction = True

    def __init__(self, *args, verify_ms=None, draft_ms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._verify_override = verify_ms
        self._draft_override = draft_ms
        self._prefix_confidence = CpuGpuBuffer(
            self.req_states.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=self.req_states.device,
        )
        self._prefix_confidence.np.fill(1.0)
        self._prefix_confidence.gpu.fill_(1.0)
        self._confidence_ready = torch.cuda.Event(blocking=True)
        self._confidence_pending = False
        self._prefix_batch: dict[str, tuple[int, int]] | None = None
        self._policy_rounds = 0
        self._proposed = 0
        self._verified = 0

    def _wait_confidence(self) -> None:
        if self._confidence_pending:
            with gpu_sync_allowed():
                self._confidence_ready.synchronize()
            self._confidence_pending = False

    def add_request(self, req_idx: int) -> None:
        self._wait_confidence()
        self._prefix_confidence.np[req_idx].fill(1.0)
        self._prefix_confidence.gpu[req_idx].fill_(1.0)

    def record_confidences(self, confidence_probs, input_batch) -> None:
        self._wait_confidence()
        self._prefix_confidence.gpu[input_batch.idx_mapping] = confidence_probs[
            : input_batch.num_reqs
        ]
        current = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current)
        with stream(self._copy_stream, current):
            self._prefix_confidence.copy_to_cpu()
            self._confidence_ready.record()
        self._confidence_pending = True

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        self._wait_confidence()
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        slots = np.array(
            [self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp
        )
        caps = np.array([len(draft_tokens.get(r, ())) for r in req_ids], np.int32)
        base = np.array(list(num_tokens_per_req.values()), np.int32) - caps
        sampling = (
            self.req_states.num_computed_tokens_np[slots] + base
            >= self.req_states.prefill_len.np[slots]
        )
        draft_table, verify_table = self.cost_tables
        if self._verify_override is not None:
            verify_table = np.array([0.0, *self._verify_override], np.float64)
        draft_ms = (
            self._draft_override
            if self._draft_override is not None
            else float(draft_table[len(req_ids)])
        )
        tp = get_tp_group()
        decision = None
        if tp.rank_in_group == 0:
            try:
                depths = select_prefixes(
                    self._prefix_confidence.np[slots],
                    caps,
                    base,
                    sampling,
                    verify_table,
                    draft_ms,
                    max(
                        0,
                        self._max_total_logits - len(req_ids) * self.num_bonus_tokens,
                    ),
                ).depths.tolist()
                decision = depths, None
            except ValueError as exc:
                decision = None, str(exc)
        if tp.world_size > 1:
            decision = tp.broadcast_object(decision, src=0)
        assert decision is not None
        depths, error = decision
        if error is not None:
            raise ValueError(f"TensorFold prefix selection failed: {error}")
        assert depths is not None
        self._prefix_batch = {
            r: (int(b), int(d)) for r, b, d in zip(req_ids, base, depths, strict=True)
        }
        self._policy_rounds += 1
        self._proposed += int(caps.sum())
        self._verified += sum(depths)
        if tp.rank_in_group == 0 and self._policy_rounds % 128 == 0:
            logger.info(
                "DSpark tensorfold policy: proposed=%d verified=%d rounds=%d",
                self._proposed,
                self._verified,
                self._policy_rounds,
            )
        return sum(b + d for b, d in self._prefix_batch.values())

    def compact_batch(
        self,
        num_draft_tokens_per_req,
        num_scheduled_tokens,
        cu_num_logits_np,
        *,
        req_ids=None,
    ):
        assert self._prefix_batch is not None and req_ids is not None
        base = np.array([self._prefix_batch[r][0] for r in req_ids], np.int32)
        depths = np.array([self._prefix_batch[r][1] for r in req_ids], np.int32)
        # These runner counts describe verified prefixes. Scheduler metrics
        # still count all proposed drafts, so their denominator stays unchanged.
        num_draft_tokens_per_req[:] = depths
        boundaries = np.empty(len(req_ids) + 1, dtype=cu_num_logits_np.dtype)
        boundaries[0] = 0
        np.cumsum(depths + self.num_bonus_tokens, out=boundaries[1:])
        return base + depths, boundaries

    def reallocate_drafts(self, req_ids, idx_mapping):
        batch, self._prefix_batch = self._prefix_batch, None
        assert batch is not None
        base = np.array([batch[r][0] for r in req_ids], np.int32)
        depths = np.array([batch[r][1] for r in req_ids], np.int32)
        n = len(req_ids)
        num_tokens = int((base + depths).sum())
        q = np.empty(n + 1, np.int32)
        q[0] = 0
        np.cumsum(base + depths, out=q[1:])
        logits = np.empty(n + 1, np.int32)
        logits[0] = 0
        np.cumsum(depths + self.num_bonus_tokens, out=logits[1:])
        async_tensor_h2d(q, out=self.query_start_loc[: n + 1])
        self.query_start_loc[n + 1 :].fill_(num_tokens)
        async_tensor_h2d(logits, out=self._cu_num_logits[: n + 1])
        # Counts are exact on the host; no admitted-count D2H is needed.
        return (
            self._cu_num_logits[: n + 1],
            self.query_start_loc,
            int(depths.sum()),
            None,
        )
