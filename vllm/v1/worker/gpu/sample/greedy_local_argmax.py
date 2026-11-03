# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy steps sampled from a vocab-parallel argmax, without the logits all-gather.

VLLM_GREEDY_LOCAL_ARGMAX=1 (default off). With a TP-sharded lm_head, every step
all-gathers the full logits (rows x vocabulary) before the sampler or the
rejection sampler reads them. When every request in the step is greedy and asks
for nothing that reads or edits the logits, those only need the argmax of each
row: the model computes it from its vocabulary shards instead
(``compute_logits_argmax``: each rank's max and argmax over its own columns,
then one all-gathered (value, index) pair per row, the reduction of vLLM's
LogitsProcessor.get_top_tokens), and the step's tokens come from those ids
through the plain sampler's greedy rule or the rejection sampler's
(rejection_sampler_utils.greedy_verify).

The tokens are the ones the full path picks, bit for bit. Stock greedy sampling
(gumbel_sample at temperature 0) and stock greedy verification
(rejection_sample) both take, per row, the first index of the maximum of the
gathered logits read as fp32: vocab-block argmaxes that keep the lower index on
equal values, then the first block among equal block maxima. The pair reduction
compares the same values (a rank's columns are its slice of the gathered row;
bf16 to fp32 is exact and keeps the order), keeps the first index on equal
values within a shard and the lower rank, i.e. the lower vocabulary index,
across shards, and never lets a padding column win. Rows holding NaN are the
exception (the stock kernels resolve those by their reduction order);
VLLM_COMPUTE_NANS_IN_LOGITS turns this path off.

The switch is a host-side decision per step from state every TP rank holds
identically (the batch's sampling params and grammar output), so all ranks join
the same collective. Sampling runs after the captured forward on both paths, so
a step can take either one with no graph recaptured. A step keeps the full path
when any request samples with a temperature, top-k, top-p or min-p, edits its
logits (penalties, logit bias, allowed token ids, min_tokens, bad words, a
thinking budget, custom logits processors), asks for logprobs or prompt
logprobs, or has structured output. The configuration keeps it always when the
logits are read elsewhere (LoRA, watermarking, a custom sampler, NaN counting,
sampling masks, trace replay, batch-sharded sampling, adaptive or synthetic
verification, probabilistic drafting, the acceptance estimator).

VLLM_GREEDY_LOCAL_ARGMAX_LOG=N: every N steps, log how many took this path (an
OpenAI request that leaves out temperature gets the model's generation-config
default, which may not be greedy).
"""

import os

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    gather_draft_sampled,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import greedy_verify

logger = init_logger(__name__)


def greedy_local_argmax_requested() -> bool:
    return os.environ.get("VLLM_GREEDY_LOCAL_ARGMAX") == "1"


class GreedyLocalArgmax:
    def __init__(
        self,
        model: nn.Module,
        sampler: Sampler,
        rejection_sampler: RejectionSampler | None,
        prompt_logprobs_worker: PromptLogprobsWorker,
        device: torch.device,
    ):
        self.model = model
        self.sampler = sampler
        self.rejection_sampler = rejection_sampler
        self.prompt_logprobs_worker = prompt_logprobs_worker
        # VLLM_GREEDY_LOCAL_ARGMAX_LOG=N: every N steps, log how many took this path.
        self.log_every = int(os.environ.get("VLLM_GREEDY_LOCAL_ARGMAX_LOG") or 0)
        self._steps = 0
        self._hits = 0
        if rejection_sampler is not None:
            # Compile the verify kernel now rather than on the first greedy round
            # (same argument dtypes and sampled stride as the serving calls).
            k = rejection_sampler.num_speculative_steps
            greedy_verify(
                torch.zeros(k + 1, dtype=torch.int64, device=device),
                torch.zeros(k + 1, dtype=torch.int32, device=device),
                torch.tensor([0, k + 1], dtype=torch.int32, device=device),
                k,
            )

    def applies(
        self, input_batch: InputBatch, grammar_output: GrammarOutput | None
    ) -> bool:
        """Every request greedy with nothing that reads or edits its logits. Host
        state only, the same on every TP rank."""
        hit = self._applies(input_batch, grammar_output)
        if self.log_every > 0:
            self._steps += 1
            self._hits += hit
            if self._steps % self.log_every == 0:
                logger.info(
                    "Greedy local argmax: %d of the last %d steps",
                    self._hits,
                    self.log_every,
                )
                self._hits = 0
        return hit

    def _applies(
        self, input_batch: InputBatch, grammar_output: GrammarOutput | None
    ) -> bool:
        if grammar_output is not None or input_batch.has_structured_output_reqs:
            return False
        if input_batch.num_reqs == 0 or input_batch.logits_indices.shape[0] == 0:
            return False
        idx = input_batch.idx_mapping_np
        sampler = self.sampler
        states = sampler.sampling_states
        return bool(
            (states.temperature.np[idx] == 0.0).all()
            and (states.top_k.np[idx] == states.vocab_size).all()
            and (states.top_p.np[idx] == 1.0).all()
            and (states.min_p.np[idx] == 0.0).all()
            # penalties, logit bias, allowed token ids, min_tokens, bad words, a
            # thinking budget, custom logits processors
            and not sampler.needs_logits_processing[idx].any()
            and (states.num_logprobs[idx] == NO_LOGPROBS).all()
            and sampler.logprob_token_ids_state.max_num_token_ids(idx) == 0
            # Prompt logprobs compute their own logits from the hidden states;
            # kept on the full path regardless.
            and not self.prompt_logprobs_worker.uses_prompt_logprobs[idx].any()
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        invalid_drafts: torch.Tensor | None,
    ) -> SamplerOutput:
        """What the sampler or the rejection sampler returns for this step, from
        the argmax of each logits row."""
        # [num_logits] int64
        target_argmax = self.model.compute_logits_argmax(
            hidden_states[input_batch.logits_indices]
        )
        prefill_len = self.sampler.req_states.prefill_len.gpu
        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            # Sampler: the argmax of each row.
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                prefill_len,
            )
            return SamplerOutput(
                sampled_token_ids=target_argmax.view(-1, 1),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
            )

        # Rejection sampler: drafts against the argmax of the row before them.
        draft_sampled, _ = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            prefill_len,
        )
        if invalid_drafts is not None:
            draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)
        sampled, num_sampled = greedy_verify(
            target_argmax,
            draft_sampled,
            input_batch.cu_num_logits,
            self.rejection_sampler.num_speculative_steps,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            prefill_len,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )


def maybe_create_greedy_local_argmax(
    model: nn.Module,
    sampler: Sampler | None,
    rejection_sampler: RejectionSampler | None,
    prompt_logprobs_worker: PromptLogprobsWorker | None,
    speculator: object | None,
    *,
    lora: bool,
    batch_sharded: bool,
    device: torch.device,
) -> GreedyLocalArgmax | None:
    """The fast path when VLLM_GREEDY_LOCAL_ARGMAX=1 and nothing in the
    configuration reads the full logits of a greedy step, else None."""
    if not greedy_local_argmax_requested():
        return None
    blockers: list[str] = []
    if not hasattr(model, "compute_logits_argmax"):
        blockers.append(f"{type(model).__name__} has no compute_logits_argmax")
    if type(sampler) is not Sampler or prompt_logprobs_worker is None:
        # GPUWatermarkSampler or a model state's custom sampler.
        blockers.append(f"sampler {type(sampler).__name__}")
    else:
        if sampler.compute_nans:
            blockers.append("VLLM_COMPUTE_NANS_IN_LOGITS")
        if sampler.return_sampling_mask:
            blockers.append("sampling masks")
        if sampler.trace_replay_state is not None:
            blockers.append("trace replay")
    if rejection_sampler is not None and (
        type(rejection_sampler) is not RejectionSampler
        or rejection_sampler.synthetic_conditional_rates is not None
        or rejection_sampler.enable_adaptive_verification
        or rejection_sampler.watermark_key is not None
    ):
        blockers.append("a custom, synthetic, adaptive or watermarking verifier")
    if getattr(speculator, "draft_logits", None) is not None:
        # The cached draft distribution narrows the verify vocabulary to the
        # draft's (rejection_sample takes the smaller width).
        blockers.append("probabilistic drafting")
    if getattr(speculator, "acceptance_estimator", None) is not None:
        blockers.append("the acceptance estimator")
    if lora:
        blockers.append("LoRA")
    if batch_sharded:
        blockers.append("batch-sharded sampling")
    if blockers:
        logger.warning("VLLM_GREEDY_LOCAL_ARGMAX ignored: %s.", "; ".join(blockers))
        return None
    assert sampler is not None and prompt_logprobs_worker is not None
    logger.info(
        "Greedy steps sample from a vocab-parallel argmax, without the logits "
        "all-gather (VLLM_GREEDY_LOCAL_ARGMAX)."
    )
    return GreedyLocalArgmax(
        model, sampler, rejection_sampler, prompt_logprobs_worker, device
    )
