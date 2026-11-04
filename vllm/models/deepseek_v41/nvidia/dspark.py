# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the
DSV4.1 DSpark, which reuses the target model's architecture similarly to MTP.
Its weights ship in the target checkpoint under the ``mtp.{0,1,2}.*`` prefix.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import os
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)

from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from .model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
    _linear_scale_param_name,
    _use_sequence_parallel,
    make_deepseek_v4_expert_params_mapping,
    maybe_init_gemm_rs,
    prepare_mega_gate_routing_metadata,
)

logger = init_logger(__name__)


def draft_logit_features(logits: torch.Tensor, gather, tp: int) -> torch.Tensor:
    """[rows, 2] fp32 (top-1 - top-2 logit margin, top-1 probability) over the whole
    vocabulary from this rank's shard of the draft logits; `gather` all-gathers a
    [rows, 3] tensor across the tp ranks along the last dim (rank order)."""
    lf = logits.float()
    local = torch.cat(
        [lf.topk(2, dim=-1).values, torch.logsumexp(lf, -1, keepdim=True)], -1
    )
    g = gather(local).view(-1, tp, 3)
    top = g[:, :, :2].reshape(g.shape[0], -1).topk(2, dim=-1).values
    lse = torch.logsumexp(g[:, :, 2], dim=-1)
    return torch.stack([top[:, 0] - top[:, 1], torch.exp(top[:, 0] - lse)], dim=-1)


def draft_features_from_records(rec: torch.Tensor) -> torch.Tensor:
    """[..., tp, 4] gathered (max, index, second, logsumexp) records, one per rank's
    vocabulary shard (MarkovBiasCache.argmax_record) -> [..., 2] fp32 (top-1 - top-2
    logit margin, top-1 probability) over the whole vocabulary: draft_logit_features
    from the records instead of the logits."""
    top = torch.cat([rec[..., 0], rec[..., 2]], dim=-1).topk(2, dim=-1).values
    lse = torch.logsumexp(rec[..., 3], dim=-1)
    return torch.stack([top[..., 0] - top[..., 1], torch.exp(top[..., 0] - lse)], dim=-1)


# VLLM_DSPARK_MARKOV_FUSED=1: the local draft argmax's Markov bias, add and max in
# two launches (ops/markov_cache.py argmax_pair), the same drafts bit for bit.
_MARKOV_FUSED = os.environ.get("VLLM_DSPARK_MARKOV_FUSED", "0") == "1"
# VLLM_DSPARK_CONTEXT_KV_GROUPED=1: the context KV's wkv for every draft layer in one
# EXL3 launch (vllm_exl3_moe.dense.context_kv_group) instead of each layer's whole
# fused wq_a|wkv with the wq_a columns dropped.
_CONTEXT_KV_GROUPED = os.environ.get("VLLM_DSPARK_CONTEXT_KV_GROUPED", "0") == "1"


def _context_kv_group(layers) -> tuple | None:
    """(run, widths) over the draft layers' fused wq_a|wkv modules, or None (the plugin
    is missing, or its modules are not EXL3 with flat parts)."""
    try:
        from vllm_exl3_moe.dense import context_kv_group
    except ImportError:
        return None
    return context_kv_group([layer.attn.fused_wqa_wkv for layer in layers], part=1)

# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")

# The drafter's own lm_head (VLLM_DSPARK_OWN_HEAD=1): checkpoint pieces
# mtp.head.{trellis,suh,svh,mul1|mcg}, the target head's head.* convention (patch 0007)
# under mtp., loaded into lm_head. The head is built under OWN_HEAD_PREFIX, the name a
# quantization config keys its method on (vllm_exl3_moe: exl3_moe.drafter_head_k).
_OWN_HEAD_PIECES_RE = re.compile(r"^mtp\.head\.(trellis|suh|svh|mul1|mcg)$")
OWN_HEAD_PREFIX = "mtp_head"


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.use_native_mega_moe = (
            vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        )
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        # Shared with the target (aliased by the speculator's loading utility).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # VLLM_DSPARK_MAIN_PROJ_ROWPAR=1: main_proj row-parallel over the concatenated
        # target taps. Each rank multiplies its share of the taps by the matching input
        # rows (half the weight bytes a rank at TP=2) and one all-reduce sums the
        # partials; replicated otherwise. The split must keep whole 32-column blocks
        # (the MXFP8 [32, 32] scales) and needs TP > 1 without sequence parallelism.
        main_in = config.hidden_size * len(self.target_layer_ids)
        tp_size = get_tensor_model_parallel_world_size()
        want_rowpar = os.environ.get("VLLM_DSPARK_MAIN_PROJ_ROWPAR", "0") == "1"
        self.main_proj_rowpar = (
            want_rowpar
            and tp_size > 1
            and not self.use_sequence_parallel
            and main_in % (32 * tp_size) == 0
        )
        if want_rowpar and not self.main_proj_rowpar:
            logger.warning(
                "VLLM_DSPARK_MAIN_PROJ_ROWPAR=1 needs TP > 1 without sequence "
                "parallelism and a 32-column-block input split; main_proj stays "
                "replicated"
            )
        if self.main_proj_rowpar:
            self.main_proj = RowParallelLinear(
                main_in,
                config.hidden_size,
                bias=False,
                input_is_parallel=False,
                reduce_results=True,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "main_proj"),
            )
            logger.info(
                "DSpark main_proj row-parallel over %d ranks "
                "(VLLM_DSPARK_MAIN_PROJ_ROWPAR=1)",
                tp_size,
            )
        else:
            self.main_proj = ReplicatedLinear(
                main_in,
                config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "main_proj"),
            )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        current_vllm_config = get_current_vllm_config()
        # The target model already holds the GEMM-RS workspace (same hidden
        # size, same TP group); this only re-checks and binds the draft layers.
        run_gemm_rs = maybe_init_gemm_rs(
            current_vllm_config, self.use_sequence_parallel
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                    run_gemm_rs=run_gemm_rs,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Heads: final norm, and the Markov + confidence heads.
        # Loaded from the "final" MTP layer weights (mtp.*) in the target
        # checkpoint. v4.1 has no learned hc_head: the hc copies are
        # collapsed with a pre-mix derived from the last layer's hc_ffn
        # projection at the end of forward().
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)].
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is a per-layer list (each entry is the context
        slot mapping for that layer's kv-cache group, since the hybrid manager may
        place draft layers in different groups). ``None`` (or a ``None`` entry)
        runs the projection to reserve workspace but writes nothing (profiling).
        """
        grouped = (
            _context_kv_group(self.layers)
            if _CONTEXT_KV_GROUPED and 0 < main_x.shape[0] <= 16
            else None
        )
        kv_all = grouped[0](main_x) if grouped is not None else None
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.attn
            if kv_all is not None:
                # this layer's wkv columns of the one launch over every layer's wkv
                lo = sum(grouped[1][:i])
                kv = kv_all[:, lo : lo + grouped[1][i]]
            else:
                # Optimized DSV4 MLA path: wkv part of the fused wq_a|wkv projection
                # (q_lora part discarded), then RoPE/quant/insert via the fused op.
                qr_kv, _ = attn.fused_wqa_wkv(main_x)
                kv = qr_kv[..., attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, inputs_embeds
                )
            inputs_embeds = sp_shard(inputs_embeds)
            input_ids = sp_shard(input_ids)
        mega_gate_metadata = None
        if self.use_native_mega_moe:
            mega_gate_metadata = prepare_mega_gate_routing_metadata(
                input_ids,
                has_hash_routing=False,
                image_sentinel_base_id=IMAGE_SENTINEL_BASE_ID
                if getattr(self.config, "vision_n_layers", 0) > 0
                else None,
            )
        # Expand to hc_mult copies for hyper-connections ([T, H] -> [T, hc, H]).
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                mega_gate_metadata=mega_gate_metadata,
            )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference forward_head applies (hc_pre with the
        # last block's ffn pre-mix). Return the PRE-norm head hidden;
        # compute_logits applies self.norm.
        assert pre_mix is not None
        hidden_states = hc_collapse_triton(hidden_states, pre_mix)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        return hidden_states


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Insert normalized context KV without constructing an unused query."""
    # The bound cache already is [num_blocks, block_size, bytes_per_token], so
    # the record width needs no per-layer attribute -- one whose name the
    # attention layer is free to change (kv_bytes_per_token has to become
    # swa_bytes_per_token once the sliding-window and compressed records can
    # differ). block_size stays the layer's, so the op's size(1) check still
    # cross-checks the two rather than restating the tensor.
    cache = attn.swa_cache_layer.kv_cache
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        kv,
        cache,
        slot_mapping,
        positions,
        attn.rotary_emb.cos_sin_cache,
        attn.swa_cache_layer.block_size,
        attn._flashinfer_fp8_kv_scale if cache.dtype == torch.float8_e4m3fn else None,
        attn.kv_mxfp8,
    )


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model aliases the target's. The exception is opt-in: with
    # VLLM_DSPARK_OWN_HEAD=1 on a checkpoint that carries the drafter's own head
    # (mtp.head.*), __init__ builds it and sets keeps_own_lm_head, and load_dspark_model
    # keeps it. has_own_lm_head stays False: it asks for _should_share's weight
    # comparison, and a quantized head (EXL3: trellis pieces) has no .weight to compare.
    has_own_embed_tokens = False
    has_own_lm_head = False
    keeps_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.linear_scale_name = _linear_scale_param_name(
            vllm_config, getattr(self.config, "expert_dtype", "fp4")
        )
        self.pad_shared_expert = getattr(
            self.quant_config, "weight_block_size", None
        ) is not None and not _use_sequence_parallel(vllm_config)
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # The drafter's own head (VLLM_DSPARK_OWN_HEAD=1), if the checkpoint has one.
        self.lm_head = self._own_lm_head(prefix)
        if self.lm_head is None:
            # Shared with the target (aliased by the speculator's load utility).
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # Cached Markov bias rows (ops/markov_cache.py), built after loading when
        # VLLM_DSPARK_MARKOV_CACHE names a token list.
        self._markov_cache = None
        # VLLM_DSPARK_LOCAL_ARGMAX=1: greedy drafts argmax each rank's vocabulary shard of
        # (base logits + Markov bias) and all-gather only (value, index) per row, instead of
        # all-gathering the full draft logits and adding a full-vocab bias on every rank.
        self._markov_cache_local = None

    def _own_lm_head(self, prefix: str) -> ParallelLMHead | None:
        """VLLM_DSPARK_OWN_HEAD=1: the drafter's own lm_head, from the checkpoint's
        mtp.head.* pieces (another quantization of the target's [hidden -> vocab]
        matrix, e.g. EXL3 K4 under a K6 target head: the drafter only proposes, the
        target verifies), when the quant config gives OWN_HEAD_PREFIX a method. None
        otherwise, which keeps the shared head. Built like the placeholder it replaces
        (same vocab, padding and TP group), so its vocabulary shard is the target
        head's: _vocab_shard, the local draft argmax and the Markov cache read the
        same columns."""
        if os.environ.get("VLLM_DSPARK_OWN_HEAD") != "1":
            return None
        head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, OWN_HEAD_PREFIX),
        )
        unquantized = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        if type(head.quant_method) in unquantized:
            # no method: the checkpoint declares no drafter head
            logger.warning(
                "VLLM_DSPARK_OWN_HEAD=1, but the quant config has no method for %s "
                "(no drafter head in this checkpoint): the drafter shares the "
                "target's lm_head",
                head.prefix,
            )
            return None
        self.keeps_own_lm_head = True
        logger.info(
            "DSpark drafter: its own lm_head (%s, mtp.head.*), not the target's",
            type(head.quant_method).__name__,
        )
        return head

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm collapsed head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_bias_for_ids(
        self, token_ids: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        if self._markov_cache is None:
            return self.markov_bias(markov_embed)
        return self._markov_cache(token_ids, markov_embed)

    def _maybe_build_markov_cache(self) -> None:
        if not os.environ.get("VLLM_DSPARK_MARKOV_CACHE"):
            return
        from .ops.markov_cache import MarkovBiasCache

        head = self.model.markov_head
        w1, w2 = head.markov_w1.weight, head.markov_w2.weight
        if w2.dim() != 2 or w2.dtype not in (torch.bfloat16, torch.float16):
            logger.warning("DSpark Markov cache off: markov_w2 is %s", w2.dtype)
            return
        probe = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=w2.device)
        ref = self.markov_bias(self.markov_embed(probe))
        if ref.shape != (probe.numel(), w2.shape[0]):
            logger.warning("DSpark Markov cache off: bias shape %s", tuple(ref.shape))
            return
        shard = self._vocab_shard()
        if os.environ.get("VLLM_DSPARK_LOCAL_ARGMAX") == "1" and shard is not None:
            lo, hi = shard
            cache = MarkovBiasCache(w1, w2[lo:hi], ref.dtype)
            got = cache(probe, self.markov_embed(probe))
            ref = ref[:, lo:hi]
            err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
            logger.info(
                "DSpark Markov cache, vocab columns %d-%d (local draft argmax): "
                "max rel diff vs the GEMM path %.2e", lo, hi, err
            )
            self._markov_cache_local = cache
            return
        cache = MarkovBiasCache(w1, w2, ref.dtype)
        got = cache(probe, self.markov_embed(probe))
        err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
        logger.info("DSpark Markov cache: max rel diff vs the GEMM path %.2e", err)
        self._markov_cache = cache

    def _vocab_shard(self) -> tuple[int, int] | None:
        if getattr(self.lm_head, "tp_size", 1) <= 1:
            return None
        si = self.lm_head.shard_indices
        return si.org_vocab_start_index, si.org_vocab_end_index

    def supports_local_argmax(self) -> bool:
        return self._markov_cache_local is not None

    def draft_features(self, rec: torch.Tensor) -> torch.Tensor:
        return draft_features_from_records(rec)

    def supports_feature_records(self) -> bool:
        return _MARKOV_FUSED and self._markov_cache_local is not None

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(
            self.lm_head, self.model.norm(hidden_states), skip_gather=True
        )

    def draft_argmax_local(
        self,
        base_local: torch.Tensor,
        token_ids: torch.Tensor,
        markov_embed: torch.Tensor,
        feat_out: torch.Tensor | None = None,
        rec_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self._markov_cache_local is not None
        shard = self._vocab_shard()
        assert shard is not None
        lo, hi = shard
        if feat_out is not None:
            bias = self._markov_cache_local(token_ids, markov_embed)
            logits = base_local[:, : hi - lo] + bias
            vals, idx = logits.max(dim=-1)
            pair = torch.stack([vals.float(), (idx + lo).float()], dim=-1)
            feat_out.copy_(draft_logit_features(
                logits,
                lambda t: tensor_model_parallel_all_gather(t, dim=-1),
                self.lm_head.tp_size,
            ))
        elif rec_out is not None and base_local.stride(-1) == 1:
            pair = self._markov_cache_local.argmax_record(
                token_ids, markov_embed, base_local, hi - lo, lo
            )
        elif rec_out is not None:
            bias = self._markov_cache_local(token_ids, markov_embed)
            lf = (base_local[:, : hi - lo] + bias).float()
            vals, idx = lf.max(dim=-1)
            pair = torch.stack([
                vals, (idx + lo).float(), lf.topk(2, dim=-1).values[:, 1],
                torch.logsumexp(lf, dim=-1),
            ], dim=-1)
        elif _MARKOV_FUSED and base_local.stride(-1) == 1:
            pair = self._markov_cache_local.argmax_pair(
                token_ids, markov_embed, base_local, hi - lo, lo
            )
        else:
            bias = self._markov_cache_local(token_ids, markov_embed)
            logits = base_local[:, : hi - lo] + bias
            vals, idx = logits.max(dim=-1)
            pair = torch.stack([vals.float(), (idx + lo).float()], dim=-1)
        tp = self.lm_head.tp_size
        gathered = tensor_model_parallel_all_gather(pair, dim=-1).view(
            -1, tp, pair.shape[-1]
        )
        if rec_out is not None:
            rec_out.copy_(gathered)
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=best).squeeze(-1).to(torch.int64)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target,
        except a drafter head of its own (``keeps_own_lm_head``: ``mtp.head.*``).
        """
        first_layer = self.model.layers[0]
        use_native_mega_moe = first_layer.ffn.use_native_mega_moe
        # Draft MoE layers use the dspark_* expert counts, not the
        # backbone's (see DeepseekV4MoE and the reference
        # ModelArgs.get_moe_config).
        n_draft_experts = (
            getattr(self.config, "dspark_n_routed_experts", 0)
            or self.config.n_routed_experts
        )
        if use_native_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(n_draft_experts)
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=n_draft_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_confidence_head = False

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True
            grouped_match = re.search(
                r"\.slice\.(\d+)\.(trellis|suh|svh|mul1|mcg)$", name
            )
            if grouped_match:
                slice_id = int(grouped_match.group(1))
                canonical_group = (
                    name[: grouped_match.start()] + "." + grouped_match.group(2)
                )
                mapped_group = canonical_group
                # EXL3 wo_a may expose real ``slice.<id>`` submodules (the
                # current plugin) or a legacy flat parameter whose loader
                # consumes the slice id.  Accept both checkpoint contracts.
                if name in params_dict:
                    mapped_group = name
                    param = params_dict[mapped_group]
                    loaded = param.weight_loader(param, loaded_weight)
                else:
                    param = params_dict[mapped_group]
                    loaded = param.weight_loader(param, loaded_weight, slice_id)
                if loaded is not False:
                    loaded_params.add(mapped_group)
                    # The own-head completeness check uses the canonical flat
                    # parameter names even when the current EXL3 plugin exposes
                    # physical ``slice.<id>`` parameters.
                    loaded_params.add(canonical_group)
                continue

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else f".{self.linear_scale_name}"
                )
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = DeepseekV4Model._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/markov_head/confidence_head) load directly —
            # otherwise e.g. "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                try:
                    param.weight_loader(param, loaded_weight, stacked_shard_id)
                except AssertionError as exc:
                    raise ValueError(
                        "DSpark stacked weight shape mismatch for "
                        f"{name!r} (checkpoint={tuple(loaded_weight.shape)}, "
                        f"parameter={tuple(param.shape)}, shard={stacked_shard_id})"
                    ) from exc
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        if self.keeps_own_lm_head:
            # the quant config declared the drafter's head, so its tensors must all be
            # here: drafting from an unloaded head would only cost acceptance, silently
            missing = {
                f"lm_head.{n}" for n, _ in self.lm_head.named_parameters()
            } - loaded_params
            if missing:
                raise ValueError(
                    "VLLM_DSPARK_OWN_HEAD=1: the drafter's own lm_head has a quant "
                    "method but the checkpoint has no mtp.head.* tensors for "
                    f"{sorted(missing)}"
                )
        self.process_weights_after_loading()
        self._maybe_build_markov_cache()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _finalize_moe(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()

    def process_weights_after_loading(self) -> None:
        self._finalize_moe()

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint draft name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        own = _OWN_HEAD_PIECES_RE.match(name)
        if own is not None:
            return f"lm_head.{own.group(1)}" if self.keeps_own_lm_head else None
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None
        # The checkpoint calls the Markov head's factors ``embed``/``head``;
        # DSparkMarkovHead registers them as ``markov_w1``/``markov_w2``.
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix(
                "markov_head.embed."
            )
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix(
                "markov_head.head."
            )
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "markov_head.",
            "confidence_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
