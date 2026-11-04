# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator: semi-autoregressive parallel drafting.

DSpark drafts a block of ``num_speculative_tokens`` tokens in one parallel pass
(reusing the DFlash machinery: context-KV precompute + a query-block forward),
then injects intra-block dependency with a lightweight sequential Markov head.

Differences from DFlash:
  * Anchor-as-first-prediction: each request emits exactly ``N =
    num_speculative_tokens`` query tokens (anchor + N-1 noise), NOT ``1 + N``.
    Every query position is a prediction (the anchor predicts the first draft
    token), so we sample at all N positions and ``sample_pos = query_pos + 1``
    (standard next-token), whereas DFlash's masks sit AT the predicted position.
    This is the ``sample_from_anchor`` path in the shared prepare-inputs kernel.
    Speculators-format checkpoints instead use the DFlash ``1 + N`` fill-in
    layout (anchor is the bonus token).
  * Sequential Markov sampling: instead of DFlash's single parallel sample, we
    sample left-to-right, adding a prefix-dependent Markov bias derived from the
    previously sampled token at each step.

CUDA graphs (FULL, mirroring DFlash) cover the whole draft step: the parallel
backbone forward AND the sequential Markov sampling.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.depth_policy import depth_policy_requested
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


def conf_head_override(
    head: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    num_reqs: int,
    n_spec: int,
) -> torch.Tensor:
    """VLLM_DSPARK_CONF_HEAD: per draft position i, sigmoid(x . w[i] + b[i]) over x =
    [hidden, Markov embedding] in fp32 as the checkpoint's head reads it, plus, with
    w1 [k, H, D] / b1 [k, H] / w2 [k, H] (/ b2 [k]), a residual MLP
    w2[i] . gelu(w1[i] x + b1[i]); returns [num_reqs * n_spec] probabilities."""
    x = torch.cat([hidden, markov_embed], dim=-1).float().view(num_reqs, n_spec, -1)
    z = (x * head["w"][None, :n_spec]).sum(-1) + head["b"][None, :n_spec]
    if "w1" in head:
        u = torch.einsum("rpd,phd->rph", x, head["w1"][:n_spec]) + head["b1"][None, :n_spec]
        z = z + (torch.nn.functional.gelu(u) * head["w2"][None, :n_spec]).sum(-1)
        if "b2" in head:
            z = z + head["b2"][None, :n_spec]
    return torch.sigmoid(z).flatten()


def conf_feat_head(
    confidence: torch.Tensor, feats: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """VLLM_DSPARK_CONF_FEAT_HEAD: per draft position i, sigmoid(w[i, 0] + w[i, 1] z_i
    + w[i, 2] m_i + w[i, 3] logit(p1_i)) over the head's logit z_i = logit(c_i), the
    draft's (top-1 - top-2) logit margin m_i and its top-1 probability p1_i, logits
    clamped at 1e-6 as scripts/dspark/fit_draft_features.py fits them. confidence
    [reqs, k], feats [reqs, k, 2], w [k, 4] -> [reqs, k]."""
    z = torch.logit(confidence.float(), eps=1e-6)
    lp = torch.logit(feats[..., 1], eps=1e-6)
    return torch.sigmoid(w[:, 0] + w[:, 1] * z + w[:, 2] * feats[..., 0] + w[:, 3] * lp)


class DSparkSpeculator(DFlashSpeculator):
    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Whether to sample from the anchor position. When True, uses anchor-as-first
        # (N slots, each position predicts the next token). When False, uses 1+N
        # fill-in block (anchor is a bonus token).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # DSpark consumes mean-pooled target aux hidden states at the target
        # layers, combined to hidden_size via main_proj. Store that combined
        # main_x (hidden_size wide). DSpark does not use the same pre-allocated buffer
        # that DeepSeek-V4's MTP uses.
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )

        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # Reduced-vocab probabilistic drafting only; set in load_draft_model.
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )

        self.use_confidence_head: bool = False
        self._markov_by_ids: bool = False
        self._local_argmax: bool = False
        self._feat_head: torch.Tensor | None = None
        self._feat_rec: torch.Tensor | None = None
        self.draft_fitted_conf: torch.Tensor | None = None
        self._log_feats: bool = False
        self.draft_feats = torch.zeros(
            self.max_num_reqs, self.num_speculative_steps, 2,
            dtype=torch.float32, device=device,
        )
        self._log_hidden: bool = False
        self.conf_inputs: torch.Tensor | None = None
        self._conf_head: dict[str, torch.Tensor] | None = None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # Reduced draft vocab: probabilistic rejection sampling indexes draft
        # logits by target id, so precompute the draft->target column map and a
        # scratch buffer to scatter logits into target vocab before sampling.
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            # -inf once; the per-step scatter overwrites the draft->target
            # columns. Kept separate from draft_logits to avoid aliasing.
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        # The per-round depth policy (depth_policy.py) reads the confidences too.
        self.use_confidence_head = (
            self.enable_adaptive_verification or depth_policy_requested()
        ) and model.model.confidence_head is not None
        if self.use_confidence_head:
            # The acceptance estimator is not needed when a trained confidence head
            # is available.
            self.use_acceptance_estimator = False
        self._markov_by_ids = hasattr(model, "markov_bias_for_ids")
        # The local argmax never materializes full draft logits: greedy drafting only, and
        # no acceptance estimator (it reads the full logits; the confidence head replaces it).
        self._local_argmax = (
            getattr(model, "supports_local_argmax", lambda: False)()
            and self.draft_logits is None
            and not self.use_acceptance_estimator
            and self._draft_topk is None
        )
        if self._local_argmax:
            logger.info("DSpark: greedy drafts argmax per vocabulary shard (local argmax)")
        self._log_feats = (
            self._local_argmax
            and os.environ.get("VLLM_DSPARK_CONF_LOG_FEATURES", "0") == "1"
            and bool(os.environ.get("VLLM_DSPARK_CONF_LOG"))
        )
        if self._log_feats:
            logger.info("DSpark: draft margins and top-1 probabilities logged per round")
        self._log_hidden = (
            self.use_confidence_head
            and os.environ.get("VLLM_DSPARK_CONF_LOG_HIDDEN", "0") == "1"
            and bool(os.environ.get("VLLM_DSPARK_CONF_LOG"))
        )
        path = os.environ.get("VLLM_DSPARK_CONF_HEAD")
        if path and self.use_confidence_head:
            import numpy as np

            z = np.load(path)
            head = {key: torch.tensor(z[key], dtype=torch.float32, device=self.device)
                    for key in z.files}
            k = self.num_speculative_steps
            if head["w"].shape[0] < k or ("w1" in head) != ("w2" in head):
                raise ValueError(f"VLLM_DSPARK_CONF_HEAD={path}: needs w [>= {k}, D] (+ w1, b1, w2)")
            self._conf_head = head
            logger.info(
                "DSpark confidence head from %s (%s)", path,
                "linear + MLP" if "w1" in head else "linear",
            )
        fpath = os.environ.get("VLLM_DSPARK_CONF_FEAT_HEAD")
        if fpath:
            if not (
                self._local_argmax
                and self.use_confidence_head
                and getattr(model, "supports_feature_records", lambda: False)()
            ):
                raise ValueError(
                    f"VLLM_DSPARK_CONF_FEAT_HEAD={fpath}: needs the local draft argmax, "
                    "VLLM_DSPARK_MARKOV_FUSED=1 and the confidence head"
                )
            import numpy as np

            w = np.load(fpath)["w"]
            k = self.num_speculative_steps
            if w.ndim != 2 or w.shape[0] < k or w.shape[1] != 4:
                raise ValueError(
                    f"VLLM_DSPARK_CONF_FEAT_HEAD={fpath}: needs w [>= {k}, 4], got {w.shape}"
                )
            self._feat_head = torch.tensor(w[:k], dtype=torch.float32, device=self.device)
            self._feat_rec = torch.zeros(
                self.max_num_reqs, k, model.lm_head.tp_size, 4,
                dtype=torch.float32, device=self.device,
            )
            self.draft_fitted_conf = torch.zeros(
                self.max_num_reqs, k, dtype=torch.float32, device=self.device
            )
            logger.info("DSpark confidence from the head + draft features in %s", fpath)
        if self._log_hidden:
            width = self.draft_model_config.get_hidden_size() + int(
                getattr(self.draft_model_config.hf_config, "dspark_markov_rank", 0) or 0
            )
            self.conf_inputs = torch.zeros(
                self.max_num_reqs, self.num_speculative_steps, width,
                dtype=torch.bfloat16, device=self.device,
            )
            logger.info("DSpark: the confidence head's inputs (%d a position) logged", width)
        return model

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._maybe_predict_acceptance(logits, idx_map, self._step_cols[step])
        if self.draft_logits is None:
            draft_ids = logits.argmax(dim=-1)
            return self.model.map_draft_to_target(draft_ids)

        # Probabilistic sampling and rejection operate in target-vocabulary
        # space. A reduced draft vocabulary is scattered into its target rows.
        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        sampler = (
            gumbel_sample
            if self.draft_watermarker is None
            else self.draft_watermarker.sample
        )
        return sampler(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        # Sequential Markov sampling over the backbone's output hidden states.
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        if self._local_argmax:
            # this rank's vocabulary shard of the base logits, never gathered
            base_logits = self.model.compute_draft_logits_local(sample_hidden)
        else:
            # Draft-vocab logits; sampled ids are remapped to target vocab below.
            base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            if self._local_argmax:
                draft_sampled_i = self.model.draft_argmax_local(
                    base_logits[:, i],
                    prev,
                    markov_embed,
                    feat_out=self.draft_feats[:num_reqs, i]
                    if self._log_feats and self._feat_rec is None
                    else None,
                    rec_out=self._feat_rec[:num_reqs, i]
                    if self._feat_rec is not None
                    else None,
                )
                self.draft_tokens[:num_reqs, i] = draft_sampled_i
                prev = draft_sampled_i
                continue
            if self._markov_by_ids:
                bias = self.model.markov_bias_for_ids(prev, markov_embed)
            else:
                bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            markov_cat = torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1)
            if self._conf_head is not None:
                confidence = self._confidence_override(
                    sample_hidden, markov_cat, num_reqs, n_spec
                )
            else:
                confidence = self.model.compute_confidence(sample_hidden, markov_cat)
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )
            if self._feat_head is not None:
                feats = self.model.draft_features(self._feat_rec[:num_reqs])
                self.draft_fitted_conf[:num_reqs] = conf_feat_head(
                    confidence.view(num_reqs, n_spec), feats, self._feat_head
                )
                if self._log_feats:
                    self.draft_feats[:num_reqs] = feats
            if self._log_hidden and self.conf_inputs is not None:
                dt = self.conf_inputs.dtype
                self.conf_inputs[:num_reqs] = torch.cat(
                    [sample_hidden.to(dt), markov_cat.to(dt)], dim=-1
                ).view(num_reqs, n_spec, -1)

    def _confidence_override(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        num_reqs: int,
        n_spec: int,
    ) -> torch.Tensor:
        return conf_head_override(self._conf_head, hidden, markov_embed, num_reqs, n_spec)

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Apply the sequential Markov head only to top-k base-logit candidates.

        Candidate selection is done once for all draft positions. At each
        sequential step, the selected logits are corrected in place and every
        other entry is set to ``-inf``. The normal dense sampling and rejection
        paths then consume that truncated distribution unchanged.
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # Reuse the dense backbone output as the normal sampler's input. Fill
        # once for all positions, then scatter only the corrected candidates
        # during the sequential loop.
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Full draft step (captured under CUDA graph): parallel backbone forward
        # then sequential Markov sampling over its hidden state outputs.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
