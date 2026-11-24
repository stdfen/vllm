# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify target-selected tree branches using the runner's ordinary forward."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.dflash2.tree import DraftTree, verify_paths

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

logger = init_logger(__name__)
SampleResult = tuple[SamplerOutput, torch.Tensor, torch.Tensor]


class TreeDFlash2:
    def __init__(self, speculator: "DFlash2Speculator", max_replays: int, mode: str):
        self.speculator = speculator
        self.max_replays = max_replays
        self.packed = None
        if mode != "replay":
            from vllm.v1.worker.gpu.spec_decode.dflash2.tree_packed import (
                PackedTreeVerifier,
            )

            self.packed = PackedTreeVerifier(use_graph=mode == "graph")
        self.eligible: set[str] = set()
        self.proposals: dict[str, DraftTree] = {}

    def add_request(self, req_id: str, params: SamplingParams | None) -> None:
        if (
            params is not None
            and params.temperature == 0.0
            and params.structured_outputs is None
            and params.thinking_token_budget is None
            and params.logprobs is None
            and params.prompt_logprobs is None
            and not params.trace_decode_token_ids
        ):
            # The packed verifier uses target argmax per tree node.
            if self.packed is not None and (
                params.min_tokens
                or params.allowed_token_ids is not None
                or params.logit_bias
                or params.bad_words
                or params.presence_penalty
                or params.frequency_penalty
                or params.repetition_penalty != 1.0
            ):
                return
            self.eligible.add(req_id)

    def remove_request(self, req_id: str) -> None:
        self.eligible.discard(req_id)
        self.proposals.pop(req_id, None)

    def record(self, req_ids: list[str], draft_width: int) -> None:
        for req_id in req_ids:
            self.proposals.pop(req_id, None)
        candidates = self.speculator.tree_candidates
        self.speculator.tree_candidates = None
        row_ids = [i for i, req_id in enumerate(req_ids) if req_id in self.eligible]
        if not draft_width or candidates is None or not row_ids:
            return
        token_ids, scores = candidates
        for i, ids, edges in zip(
            row_ids,
            token_ids[row_ids, :draft_width].tolist(),
            scores[row_ids, :draft_width].tolist(),
            strict=True,
        ):
            self.proposals[req_ids[i]] = DraftTree(ids, edges)

    @staticmethod
    def _save_gdn_sources(
        attn_metadata: dict[str, Any],
        layers: dict[str, Any],
        num_reqs: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Save the conv block and accepted recurrent slot each replay reads."""
        saved = []
        source_slots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_name, metadata in attn_metadata.items():
            if not isinstance(metadata, GDNAttentionMetadata):
                continue
            if id(metadata) not in source_slots:
                slots = metadata.spec_state_indices_tensor
                accepted = metadata.num_accepted_tokens
                if (
                    metadata.num_spec_decodes != num_reqs
                    or metadata.num_prefills
                    or metadata.num_decodes
                    or slots is None
                    or accepted is None
                ):
                    return []
                slots = slots[:num_reqs]
                columns = accepted[:num_reqs].long() - 1
                if bool(((columns < 0) | (columns >= slots.shape[1])).any()):
                    return []
                conv = slots[:, 0].long()
                recurrent = slots.gather(1, columns[:, None]).flatten().long()
                if bool(((conv <= 0) | (recurrent <= 0)).any()):
                    return []
                source_slots[id(metadata)] = conv, recurrent
            conv_state, recurrent_state = layers[layer_name].kv_cache
            for state, indices in zip(
                (conv_state, recurrent_state), source_slots[id(metadata)], strict=True
            ):
                saved.append((state, indices, state.index_select(0, indices)))
        return saved

    def verify(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        layers: dict[str, Any],
        forward: Callable[[], Any],
        sample: Callable[[torch.Tensor], SampleResult],
        positions: torch.Tensor,
        packed_forward: Callable[[torch.Tensor, torch.Tensor], Any],
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[Any, SampleResult | None]:
        draft_counts = input_batch.num_draft_tokens_per_req
        if (
            input_batch.has_prefill
            or input_batch.has_structured_output_reqs
            or not input_batch.num_draft_tokens
            or draft_counts is None
            or not (input_batch.num_scheduled_tokens == draft_counts + 1).all()
            or not all(req_id in self.eligible for req_id in input_batch.req_ids)
        ):
            return forward(), None

        trees = [self.proposals.get(req_id) for req_id in input_batch.req_ids]
        if not any(tree is not None for tree in trees):
            return forward(), None

        if self.packed is not None:
            result = self.packed.verify(
                trees,
                input_batch,
                positions,
                attn_metadata,
                packed_forward,
                compute_logits,
                self.speculator.draft_attn_layer_names,
            )
            return result if result is not None else (forward(), None)

        starts = input_batch.query_start_loc_np[: input_batch.num_reqs + 1]
        bounds = [(int(a) + 1, int(b)) for a, b in zip(starts[:-1], starts[1:])]
        tokens = input_batch.input_ids[: input_batch.num_tokens]
        original = tokens.tolist()
        paths = [original[a:b] for a, b in bounds]
        saved = self._save_gdn_sources(attn_metadata, layers, input_batch.num_reqs)
        if not saved:
            return forward(), None

        def run(
            selected_paths: list[list[int]], replay: bool
        ) -> tuple[tuple[Any, SampleResult], list[list[int]]]:
            if replay:
                for state, indices, snapshot in saved:
                    state.index_copy_(0, indices, snapshot)
                inputs = original.copy()
                for (a, b), path in zip(bounds, selected_paths, strict=True):
                    inputs[a:b] = path
                tokens.copy_(
                    torch.tensor(inputs, dtype=tokens.dtype, device=tokens.device)
                )
            # This forward overwrites the same provisional paged-KV positions.
            # The prefix is read-only and rejected positions stay beyond seq_len
            # after the normal scheduler/runner rejection accounting.
            output = forward()
            hidden = output[0] if isinstance(output, tuple) else output
            result = sample(hidden)
            sampler_output, num_sampled, _ = result
            outputs = sampler_output.sampled_token_ids.tolist()
            return (output, result), [
                row[:count]
                for row, count in zip(outputs, num_sampled.tolist(), strict=True)
            ]

        outcome, committed_branch = verify_paths(paths, trees, run, self.max_replays)
        if committed_branch:
            logger.info_once("Tree-DFlash2 verified and selected a sibling branch.")
        return outcome
