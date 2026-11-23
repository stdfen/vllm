# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned TensorFold request-lane port with a vLLM model-execution adapter.

DFlashSpeculator supplies only device input/KV staging and graph invocation.
No DSparkSpeculator, adaptive-verification manager, legacy depth policy,
native Markov cache, acceptance estimator or rejection sampler is used here.
The FP16 bias cache belongs exclusively to this independent lane.
The model loader supplies checkpoint tensors and ordinary model operators.
"""

import atexit
import copy
import json
import os
import time
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn.functional as F

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model
from vllm.v1.worker.gpu.spec_decode.tensorfold_dsv41.core import (
    CalibratedLaneState,
    MultiLaneRule,
)
from vllm.v1.worker.gpu.spec_decode.tensorfold_dsv41.head import FP16BiasCache


class TensorFoldDSV41Speculator(DFlashSpeculator):
    """Independent request lanes with C1 copy fast paths and batched drafting."""

    _speculator_name = "TensorFoldDSV41"
    tensorfold_full = True

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        if self.num_speculative_steps != 5:
            raise ValueError(
                "TensorFold full path reserves 5 drafts (6-row exact window)"
            )
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.enable_batch_sharded_sampling
            or vllm_config.lora_config
        ):
            raise ValueError(
                "TensorFold full path requires PP1/DP1/PCP1 and no LoRA/sharding"
            )
        if os.environ.get("VLLM_DSPARK_DEPTH_POLICY"):
            raise ValueError("Unset the legacy DSpark depth policy for TensorFold full")
        self.sample_from_anchor = True
        self.num_query_per_req = 5
        self.markov_steps = 5
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.draft_model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )
        self.feedback = torch.zeros(self.max_num_reqs, 2, 5, device=device)
        self.confidence_logits = self.feedback[:, 1]
        self.rule = MultiLaneRule()
        self.lanes = {}
        self.held = {}
        self.batch_plan = None
        self.target_output = None
        self.round_start = None
        self.depth_graphs = {}
        self.warming_up = False
        self._trace = None
        self._lifecycle_trace = None
        self._graph_trace = None
        self._resources = ExitStack()
        atexit.register(self._resources.close)
        path = os.environ.get("VLLM_TENSORFOLD_TRACE")
        if path and (
            get_tp_group().rank_in_group == 0
            or os.environ.get("VLLM_TENSORFOLD_TRACE_ALL_RANKS") == "1"
        ):
            self._graph_trace = self._resources.enter_context(
                open(  # noqa: SIM115
                    path + ".graphs.jsonl", "a", encoding="utf-8", buffering=1
                )
            )
            self._trace = self._resources.enter_context(
                open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
            )
            self._lifecycle_trace = self._resources.enter_context(
                open(  # noqa: SIM115
                    path + ".lifecycle.jsonl", "a", encoding="utf-8", buffering=1
                )
            )

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = load_dspark_model(target_model, self.vllm_config)
        if model.model.confidence_head is None:
            raise ValueError(
                "TensorFold full path requires checkpoint confidence weights"
            )
        if model.draft_id_to_target_id is not None:
            raise ValueError("TensorFold DSV4.1 requires a full vocabulary draft head")
        # tri_markov.zig: HEAD and the staged/cache bias are fp16; EMB is
        # checkpoint bf16. Preserve this conversion rather than inheriting
        # the original vLLM Markov head's bf16 projection behavior.
        self.markov_projection_fp16 = model.model.markov_head.markov_w2.weight.to(
            torch.float16
        ).contiguous()
        self.bias_cache = FP16BiasCache(
            model.model.markov_head.markov_w1.weight, self.markov_projection_fp16
        )
        self.confidence_weight = (
            model.model.confidence_head.proj.weight.float().contiguous()
        )
        bias = model.model.confidence_head.proj.bias
        self.confidence_bias = None if bias is None else bias.float().contiguous()
        return model

    def _num_graph_context_tokens(self, num_reqs):
        return min(num_reqs * 6, self.max_num_tokens)

    def init_cudagraph_manager(self, cudagraph_mode):
        # The query backbone always has five rows. Copies take the KV-only
        # branch, so target-prefix depths must not multiply draft captures.
        super().init_cudagraph_manager(cudagraph_mode)
        graph_config = copy.copy(self.vllm_config)
        graph_config.speculative_config = copy.copy(self.speculative_config)
        graph_config.speculative_config.dspark_implementation = "native"
        graph_config.speculative_config.dspark_verification_policy = "native"
        manager = type(self.query_cudagraph_manager)(
            graph_config,
            self.device,
            self.query_cudagraph_manager.cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )
        self.query_cudagraph_manager = manager
        self.depth_graphs = dict.fromkeys(range(1, 6), manager)
        # A mixed-prefill context can exceed the fixed context staging in a
        # query graph. Keep that KV insertion and query pass together eagerly;
        # the fixed-row query graphs are reserved for pure decode batches.
        self.eager_query_manager = type(manager)(
            graph_config,
            self.device,
            CUDAGraphMode.NONE,
            decode_query_len=5,
        )

    def capture(self):
        self.markov_steps = 5
        self.query_cudagraph_manager = self.depth_graphs[5]
        super().capture()

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    ):
        width = self.num_query_per_req
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        head_hidden = hidden[self.sample_indices[: num_reqs * width]]
        base = self.model.compute_logits(head_hidden).view(num_reqs, width, -1)
        markov = self.model.model.markov_head
        prev = self.input_buffers.input_ids[
            torch.arange(num_reqs, device=self.device) * width
        ]
        embeddings = []
        for step in range(self.markov_steps):
            # draft.zig _body + Markov: greedy low-rank transition from the
            # previous token. These are checkpoint matrix operations only.
            embedding = F.embedding(prev, markov.markov_w1.weight)
            ids = self.bias_cache.greedy(prev, embedding, base[:, step])
            self.draft_tokens[:num_reqs, step] = ids
            embeddings.append(embedding)
            prev = ids
        conf_in = torch.cat(
            (
                head_hidden.view(num_reqs, width, -1)[:, : self.markov_steps].float(),
                torch.stack(embeddings, dim=1).float(),
            ),
            dim=-1,
        )
        self.confidence_logits[:num_reqs, : self.markov_steps] = F.linear(
            conf_in, self.confidence_weight, self.confidence_bias
        ).squeeze(-1)
        self.feedback[:num_reqs, 0] = self.draft_tokens[:num_reqs].float()

    def record_target_output(self, batch, sampler_output, req_states):
        self.target_output = sampler_output
        self.target_states = req_states
        if self._trace is not None and not self.warming_up:
            self._graph_trace.write(
                json.dumps(
                    dict(
                        req_ids=batch.req_ids,
                        graph=getattr(batch, "tensorfold_target_graph", "unknown"),
                        rows=batch.num_tokens,
                        padded_rows=batch.num_tokens_after_padding,
                        requests=batch.num_reqs,
                        padded_requests=batch.num_reqs_after_padding,
                        prefill=bool(batch.has_prefill),
                        query_lengths=batch.num_scheduled_tokens.tolist(),
                    )
                )
                + "\n"
            )
            self.computed_before = req_states.num_computed_tokens.gpu[
                batch.idx_mapping
            ].clone()

    def set_warmup_mode(self, enabled):
        self.warming_up = enabled
        if not enabled:
            self.lanes.clear()
            self.held.clear()
            self.batch_plan = self.target_output = self.round_start = None
            self.rule = MultiLaneRule()

    def remove_request(self, req_id: str) -> None:
        self.lanes.pop(req_id, None)
        self.held.pop(req_id, None)
        if self.batch_plan is not None:
            self.batch_plan.pop(req_id, None)
        if self._lifecycle_trace is not None:
            self._lifecycle_trace.write(
                json.dumps(
                    {
                        "req": req_id,
                        "event": "release",
                        "live_lanes": len(self.lanes),
                        "held_requests": len(self.held),
                    }
                )
                + "\n"
            )

    def get_num_tokens(self, scheduled, drafts):
        if self.warming_up:
            return sum(scheduled.values())
        if len(scheduled) != 1:
            return self._get_batch_tokens(scheduled, drafts)
        self.round_start = time.perf_counter()
        req = next(iter(scheduled))
        base = scheduled[req] - len(drafts.get(req, ()))
        lane = self.lanes.get(req)
        held, confidence, kind = self.held.get(req, ([], [], "none"))
        cap = min(len(held), len(drafts.get(req, ())))
        depth = cap
        calibration = (
            {"trials": lane.trials.copy(), "residual": lane.residual.copy()}
            if self._trace is not None and lane is not None
            else None
        )
        if kind == "head" and lane is not None:
            depth = min(cap, self.rule.admit(lane, confidence[:cap]))
        tp = get_tp_group()
        if tp.world_size > 1:
            # Preserve the coordinator's message-queue fast path. A direct
            # CPU Gloo tensor broadcast adds cross-node latency on GB10.
            depth = tp.broadcast_object(depth if tp.rank_in_group == 0 else None, src=0)
        if lane is not None:
            lane.kind = kind if depth else "none"
        self.batch_plan = {
            req: (
                base,
                depth,
                len(held),
                kind if depth else "none",
                held,
                confidence,
                self.rule.overhead_ms(1),
                calibration,
                cap,
            )
        }
        return base + depth

    def _get_batch_tokens(self, scheduled, drafts):
        self.round_start = time.perf_counter()
        records, entries = {}, []
        for req, total in scheduled.items():
            base = total - len(drafts.get(req, ()))
            lane = self.lanes.setdefault(req, CalibratedLaneState())
            held, confidence, kind = self.held.get(req, ([], [], "none"))
            cap = min(len(held), len(drafts.get(req, ())))
            entries.append(
                (
                    base,
                    lane,
                    confidence[:cap] if kind == "head" else [],
                    cap if kind == "copy" else 0,
                )
            )
            calibration = (
                {"trials": lane.trials.copy(), "residual": lane.residual.copy()}
                if self._trace is not None
                else None
            )
            records[req] = (base, held, confidence, kind, cap, calibration)
        allocator = (
            self.rule.admit_uniform
            if len(entries) <= 4 and all(entry[0] == 1 for entry in entries)
            else self.rule.admit_batch
        )
        depths = allocator(entries, sum(scheduled.values()))
        tp = get_tp_group()
        grants = dict(zip(scheduled, depths))
        if tp.world_size > 1:
            grants = tp.broadcast_object(
                grants if tp.rank_in_group == 0 else None, src=0
            )
        self.batch_plan = {}
        for req, (base, held, confidence, kind, cap, calibration) in records.items():
            depth = grants[req]
            if not 0 <= depth <= cap:
                raise RuntimeError("TensorFold cross-rank depth exceeds held prefix")
            lane = self.lanes[req]
            lane.granted, lane.kind = depth, kind if depth else "none"
            self.batch_plan[req] = (
                base,
                depth,
                len(held),
                lane.kind,
                held,
                confidence,
                self.rule.overhead_ms(len(scheduled)),
                calibration,
                cap,
            )
        return sum(row[0] + row[1] for row in self.batch_plan.values())

    def compact_batch(self, req_ids, num_drafts):
        assert self.batch_plan is not None
        base = np.array([self.batch_plan[r][0] for r in req_ids], np.int32)
        depths = np.array([self.batch_plan[r][1] for r in req_ids], np.int32)
        num_drafts[:] = depths
        boundaries = np.empty(len(req_ids) + 1, np.int32)
        boundaries[0] = 0
        np.cumsum(depths + 1, out=boundaries[1:])
        return base + depths, boundaries

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        dummy_run = dummy_run or self.warming_up
        if input_batch.num_reqs != 1:
            options = dict(
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                temperature=temperature,
                seeds=seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )
            return self._propose_multi(input_batch, options)
        req = input_batch.req_ids[0] if not dummy_run else None
        output = self.target_output
        lane = None
        count = 1
        copied_ahead = []
        plan = self.batch_plan
        evidence = None
        copy_was_silent = False

        def launch(width, skip_head):
            self.markov_steps = width
            if self.depth_graphs:
                self.query_cudagraph_manager = (
                    self.eager_query_manager
                    if input_batch.has_prefill
                    else self.depth_graphs[width]
                )
            self.draft_tokens.fill_(-1)
            try:
                result = super(TensorFoldDSV41Speculator, self).propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    dp_sync_state=dp_sync_state,
                    dummy_run=dummy_run,
                    skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                    mm_inputs=mm_inputs,
                    is_profile=is_profile,
                    num_speculative_tokens=0 if count == 0 or skip_head else width,
                )
            finally:
                self.markov_steps = 5
            return result

        # A decode verification always emits a correction. Queue the neural
        # head before host feedback; existing copy lanes keep their KV-only pass.
        deferred = (
            not dummy_run
            and not input_batch.is_prefilling_np[0]
            and req in self.lanes
            and plan is not None
            and plan[req][3] != "copy"
        )
        early_result = launch(5, False) if deferred else None
        generated = None
        if not dummy_run:
            lane = self.lanes.setdefault(req, CalibratedLaneState())
            with gpu_sync_allowed():
                if deferred:
                    packet = (
                        torch.cat(
                            (
                                num_sampled[:1].float(),
                                output.sampled_token_ids[0, :6].float(),
                                self.feedback[0].reshape(-1),
                            )
                        )
                        .cpu()
                        .tolist()
                    )
                    generated = [packet[7:12], packet[12:17]]
                else:
                    packet = (
                        torch.cat(
                            (
                                num_sampled[:1].to(torch.int64),
                                output.sampled_token_ids[0, :6],
                            )
                        )
                        .cpu()
                        .tolist()
                    )
                count = int(packet[0])
                committed = [int(x) for x in packet[1 : count + 1]]
                if self._trace is not None and not input_batch.is_prefilling_np[0]:
                    slot = int(input_batch.idx_mapping_np[0])
                    before = int(self.computed_before[0].item())
                    after = int(self.target_states.num_computed_tokens.gpu[slot].item())
                    evidence = {
                        "input_window": input_batch.input_ids[
                            input_batch.logits_indices
                        ]
                        .cpu()
                        .tolist(),
                        "target_picks": output.tensorfold_target_picks.cpu().tolist(),
                        "committed_tokens": committed,
                        "kv_before": before,
                        "kv_after": after,
                    }
                    if after != before + count:
                        raise RuntimeError("TensorFold KV commit count mismatch")
                if input_batch.is_prefilling_np[0]:
                    offset = int(input_batch.num_computed_prefill_tokens_np[0])
                    amount = min(
                        int(input_batch.num_scheduled_tokens[0]),
                        int(input_batch.prefill_len_np[0]) - offset,
                    )
                    lane.context.extend(input_batch.input_ids[:amount].cpu().tolist())
            if not input_batch.is_prefilling_np[0] and plan is not None:
                depth = plan[req][1]
                accepted = max(0, count - 1)
                lane.proposed += plan[req][2]
                if plan[req][3] == "head":
                    lane.observe_confidence(plan[req][5], depth, accepted)
                lane.observe(depth, accepted, committed)
            else:
                lane.context.extend(committed)
            copy_was_silent = lane.lookup.silent_for > 0
            copied_ahead = lane.copy_proposal()
        width = 5 if deferred or lane is None else lane.head_depth(bool(copied_ahead))
        skip_head = bool(copied_ahead) and not deferred
        result = early_result if deferred else launch(width, skip_head)
        if not dummy_run:
            with gpu_sync_allowed():
                packet = (
                    (generated if deferred else self.feedback[0].cpu().tolist())
                    if count and not skip_head
                    else [[], []]
                )
                held = [int(x) for x in packet[0][:width]]
                confidence = packet[1][:width]
            copied = lane.copy_proposal() if copy_was_silent else copied_ahead
            kind = "head"
            if copied:
                held, confidence, kind = copied, [], "copy"
                result[0, : len(held)] = torch.tensor(held, device=self.device)
                result[0, len(held) :].fill_(-1)
            self.held[req] = held, confidence, kind
            if plan is not None and not input_batch.is_prefilling_np[0]:
                elapsed = 1000 * (time.perf_counter() - self.round_start)
                self.rule.observe_round(
                    plan[req][3], plan[req][1], elapsed, lane.rounds == 1
                )
                if self._trace is not None:
                    self._trace.write(
                        json.dumps(
                            {
                                "req": req,
                                "round": lane.rounds,
                                "kind": plan[req][3],
                                "proposed": plan[req][2],
                                "verified": plan[req][1],
                                "accepted": max(0, count - 1),
                                "output": count,
                                "head_queries": 0 if skip_head else 5,
                                "head_steps": 0 if skip_head else width,
                                "held_tokens": plan[req][4],
                                "confidence_logits": plan[req][5],
                                "overhead_before": plan[req][6],
                                "calibration_before": plan[req][7],
                                "available_drafts": plan[req][8],
                                "state": evidence,
                                "round_ms": elapsed,
                                "overhead_ms": self.rule.overhead_ms(1),
                            }
                        )
                        + "\n"
                    )
            self.batch_plan = None
        return result

    def _propose_multi(self, batch, options):
        """Request-keyed feedback with one batched host read per transfer."""
        n = batch.num_reqs
        self.markov_steps = 5
        if self.depth_graphs:
            self.query_cudagraph_manager = (
                self.eager_query_manager
                if batch.has_prefill or n > 4
                else self.depth_graphs[5]
            )

        def launch(skip):
            self.draft_tokens.fill_(-1)
            return super(TensorFoldDSV41Speculator, self).propose(
                batch, **options, num_speculative_tokens=0 if skip else 5
            )

        if options["dummy_run"]:
            return launch(False)
        plan = self.batch_plan
        output = self.target_output
        counts_gpu = options["num_sampled"][:n]
        deferred = (
            not batch.is_prefilling_np.any()
            and plan is not None
            and all(r in self.lanes and plan[r][3] != "copy" for r in batch.req_ids)
        )
        result = launch(False) if deferred else None
        with gpu_sync_allowed():
            parts = [
                counts_gpu[:, None].float(),
                output.sampled_token_ids[:n, :6].float(),
            ]
            if deferred:
                parts.append(self.feedback[:n].reshape(n, 10))
            packets = torch.cat(parts, dim=1).cpu().tolist()
            prompt_ids = batch.input_ids.cpu().tolist() if batch.has_prefill else None
        copied, evidence = {}, {}
        for i, req in enumerate(batch.req_ids):
            lane = self.lanes.setdefault(req, CalibratedLaneState())
            count = int(packets[i][0])
            committed = [int(x) for x in packets[i][1 : count + 1]]
            if batch.is_prefilling_np[i]:
                offset = int(batch.num_computed_prefill_tokens_np[i])
                amount = min(
                    int(batch.num_scheduled_tokens[i]),
                    int(batch.prefill_len_np[i]) - offset,
                )
                start = int(batch.query_start_loc_np[i])
                if len(lane.context) != offset:
                    raise RuntimeError("TensorFold prompt history offset mismatch")
                lane.context.extend(prompt_ids[start : start + amount])
                lane.context.extend(committed)
            elif plan is not None:
                p = plan[req]
                accepted = max(0, count - 1)
                lane.proposed += p[2]
                if p[3] == "head":
                    lane.observe_confidence(p[5], p[1], accepted)
                lane.observe(p[1], accepted, committed)
            else:
                lane.context.extend(committed)
            copied[req] = lane.copy_proposal()
            if self._trace is not None and not batch.is_prefilling_np[i]:
                with gpu_sync_allowed():
                    lo, hi = map(int, batch.cu_num_logits_np[i : i + 2])
                    slot = int(batch.idx_mapping_np[i])
                    before = int(self.computed_before[i].item())
                    after = int(self.target_states.num_computed_tokens.gpu[slot].item())
                    evidence[req] = dict(
                        input_window=batch.input_ids[batch.logits_indices[lo:hi]]
                        .cpu()
                        .tolist(),
                        target_picks=output.tensorfold_target_picks[lo:hi]
                        .cpu()
                        .tolist(),
                        committed_tokens=committed,
                        kv_before=before,
                        kv_after=after,
                        history_length=len(lane.context),
                        slot=slot,
                    )
                    if after != before + count:
                        raise RuntimeError("TensorFold batched KV commit mismatch")
        skip = all(
            copied[r] or not int(packets[i][0]) for i, r in enumerate(batch.req_ids)
        )
        tp = get_tp_group()
        if not deferred:
            if tp.world_size > 1:
                skip = tp.broadcast_object(
                    skip if tp.rank_in_group == 0 else None, src=0
                )
            result = launch(skip)
        with gpu_sync_allowed():
            feedback = (
                [p[7:17] for p in packets]
                if deferred
                else (
                    self.feedback[:n].reshape(n, 10).cpu().tolist()
                    if not skip
                    else None
                )
            )
        proposals = {}
        for i, req in enumerate(batch.req_ids):
            if not int(packets[i][0]):
                proposals[req] = ([], [], "none")
            elif copied[req]:
                proposals[req] = (copied[req], [], "copy")
            else:
                proposals[req] = (
                    [int(x) for x in feedback[i][:5]],
                    feedback[i][5:10],
                    "head",
                )
        # A single request-keyed broadcast prevents replica-specific CPU copy
        # histories from introducing different future target windows.
        if tp.world_size > 1:
            proposals = tp.broadcast_object(
                proposals if tp.rank_in_group == 0 else None, src=0
            )
        dense = []
        for req in batch.req_ids:
            self.held[req] = proposals[req]
            tokens = proposals[req][0]
            dense.append(tokens + [-1] * (5 - len(tokens)))
        result[:n] = torch.tensor(dense, dtype=result.dtype, device=self.device)
        if plan is not None:
            elapsed = 1000 * (time.perf_counter() - self.round_start)
            rows = sum(plan[r][1] + 1 for r in batch.req_ids)
            if not batch.is_prefilling_np.any():
                self.rule.observe_overhead(n, rows, elapsed)
            if self._trace is not None:
                for i, req in enumerate(batch.req_ids):
                    if batch.is_prefilling_np[i]:
                        continue
                    p, lane = plan[req], self.lanes[req]
                    self._trace.write(
                        json.dumps(
                            dict(
                                req=req,
                                batch_ids=batch.req_ids,
                                batch_size=n,
                                round=lane.rounds,
                                kind=p[3],
                                proposed=p[2],
                                verified=p[1],
                                accepted=max(0, int(packets[i][0]) - 1),
                                output=int(packets[i][0]),
                                held_tokens=p[4],
                                confidence_logits=p[5],
                                calibration_before=p[7],
                                available_drafts=p[8],
                                state=evidence[req],
                                round_ms=elapsed,
                            )
                        )
                        + "\n"
                    )
        self.batch_plan = None
        return result
