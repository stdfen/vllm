# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorFold verifyFn + acceptPath/conclude; never invokes RejectionSampler."""

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.tensorfold_dsv41.sampling import choose


class TensorFoldSamplerOutput(SamplerOutput):
    tensorfold_target_picks: torch.Tensor


@triton.jit
def _accept(picks, inputs, indices, cu, out, counts, stride: tl.constexpr):
    req = tl.program_id(0)
    start = tl.load(cu + req)
    end = tl.load(cu + req + 1)
    n = end - start - 1
    kept = 0
    active = True
    for j in range(n):
        wanted = tl.load(picks + start + j)
        input_idx = tl.load(indices + start + j + 1)
        candidate = tl.load(inputs + input_idx)
        hit = active & (candidate == wanted)
        if hit:
            tl.store(out + req * stride + kept, candidate)
            kept += 1
        active = hit
    correction = tl.load(picks + start + kept)
    tl.store(out + req * stride + kept, correction)
    tl.store(counts + req, kept + 1)


def verify_target(logits, batch, sampler, req_states, max_drafts):
    """Sample every target row, then keep the first matching chain prefix."""
    state = sampler.sampling_states
    raw_logits = logits
    if np.any(sampler.needs_logits_processing[batch.idx_mapping_np]):
        logits = logits.float().clone()
        ctx = LogitsContext(
            expanded_idx_mapping=batch.expanded_idx_mapping,
            idx_mapping=batch.idx_mapping,
            idx_mapping_np=batch.idx_mapping_np,
            expanded_local_pos=batch.expanded_local_pos,
            input_ids=batch.input_ids[batch.logits_indices],
            pos=batch.positions[batch.logits_indices],
            seq_lens_upper_bound_np=batch.seq_lens_cpu_upper_bound.numpy(),
        )
        for processor in sampler.logits_processors:
            logits = processor.apply(logits, ctx)
        sampler.thinking_budget_state.apply(logits, ctx)
    temperatures = state.temperature.np[batch.idx_mapping_np]
    if np.all(temperatures == 0):
        picks = logits.argmax(dim=-1)
    else:
        # TensorFold draws only from top-k+8 candidates on the host. Retain
        # exact lower-id ties by lexsort; this slower path is not the C1/T0 perf arm.
        with gpu_sync_allowed():
            cpu_logits = logits.float().cpu().numpy()
            positions = (batch.positions[batch.logits_indices] + 1).cpu().tolist()
        ids = np.arange(logits.shape[-1])
        result = []
        for r in range(batch.num_reqs):
            slot = batch.idx_mapping_np[r]
            for row in range(batch.cu_num_logits_np[r], batch.cu_num_logits_np[r + 1]):
                order = np.lexsort((ids, -cpu_logits[row]))[:64]
                result.append(
                    choose(
                        cpu_logits[row, order].tolist(),
                        order.tolist(),
                        seed=int(state.seeds.np[slot]),
                        position=positions[row],
                        temperature=float(temperatures[r]),
                        top_k=(
                            20
                            if state.top_k.np[slot] >= logits.shape[-1]
                            else int(state.top_k.np[slot])
                        ),
                        top_p=float(state.top_p.np[slot]),
                        min_p=float(state.min_p.np[slot]),
                    )
                )
        picks = torch.tensor(result, dtype=torch.int64, device=logits.device)
    sampled = torch.full(
        (batch.num_reqs, max_drafts + 1), -1, dtype=torch.int64, device=logits.device
    )
    counts = torch.empty(batch.num_reqs, dtype=torch.int32, device=logits.device)
    _accept[(batch.num_reqs,)](
        picks,
        batch.input_ids,
        batch.logits_indices,
        batch.cu_num_logits,
        sampled,
        counts,
        max_drafts + 1,
    )
    counts, rejected = get_num_sampled_and_rejected(
        counts,
        batch.seq_lens,
        batch.cu_num_logits,
        batch.idx_mapping,
        req_states.prefill_len.gpu,
    )
    dimensions = sampler.get_logprobs_dims(batch.idx_mapping_np)
    logprobs = None
    if dimensions is not None:
        logprobs = compute_topk_scores(
            raw_logits,
            dimensions[0],
            picks,
            batch.cu_num_logits_np.tolist(),
            logits_mode=sampler.logprobs_mode in ("raw_logits", "processed_logits"),
        )
    output = TensorFoldSamplerOutput(sampled, logprobs, None, counts, rejected)
    output.tensorfold_target_picks = picks
    return output
