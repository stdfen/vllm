# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whenever you add an architecture to this page, please also update
`tests/models/registry.py` with example HuggingFace models for it.
"""

import importlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch.nn as nn
import transformers

from vllm import envs
from vllm.config import (
    ModelConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)
from vllm.logger import init_logger
from vllm.logging_utils import logtime
from vllm.tasks import ScoreType
from vllm.transformers_utils.dynamic_module import try_get_class_from_dynamic_module
from vllm.utils.hashing import safe_hash

if TYPE_CHECKING:
    from vllm.config.model import AttnTypeStr
    from vllm.config.pooler import SequencePoolingType, TokenPoolingType
else:
    AttnTypeStr = Any
    SequencePoolingType = Any
    TokenPoolingType = Any


from .interfaces import (
    has_inner_state,
    has_noops,
    is_attention_free,
    is_hybrid,
    requires_raw_input_tokens,
    supports_hc_sp,
    supports_mamba_prefix_caching,
    supports_multimodal,
    supports_multimodal_encoder_tp_data,
    supports_multimodal_raw_input_only,
    supports_pp,
    supports_replayssm,
    supports_transcription,
)
from .interfaces_base import (
    get_attn_type,
    get_default_seq_pooling_type,
    get_default_tok_pooling_type,
    get_score_type,
    is_pooling_model,
    is_text_generation_model,
)

logger = init_logger(__name__)

_TEXT_GENERATION_MODELS = {
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek_v2", "DeepseekForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "DeepseekV32ForCausalLM": (
        "vllm.models.deepseek_v32",
        "DeepseekV32ForCausalLM",
    ),
    "DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
    "Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),
    "Qwen4ExpForCausalLM": (
        "vllm.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
    ),
    "Glm4ForCausalLM": ("glm4", "Glm4ForCausalLM"),
    "Glm4MoeForCausalLM": ("glm4_moe", "Glm4MoeForCausalLM"),
    "Glm4MoeLiteForCausalLM": ("glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
    "GlmMoeDsaForCausalLM": ("vllm.models.deepseek_v32", "GlmMoeDsaForCausalLM"),
    "Glm5NextForCausalLM": ("vllm.models.glm5next", "Glm5NextForCausalLM"),
    "KimiLinearForCausalLM": (
        "vllm.models.kimi_k3",
        "KimiLinearForCausalLM",
    ),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),
    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),
}

_EMBEDDING_MODELS = {
    "DeepseekV3BidirectionalModel": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    **{
        # Multiple models share the same architecture, so we include them all
        k: (mod, arch)
        for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
        if arch == "LlamaForCausalLM"
    },
    "Qwen2Model": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "VoyageQwen3BidirectionalEmbedModel": (
        "voyage",
        "VoyageQwen3BidirectionalEmbedModel",
    ),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
}

_LATE_INTERACTION_MODELS = {
    "ColQwen3": ("colqwen3", "ColQwen3Model"),
    "OpsColQwen3Model": ("colqwen3", "ColQwen3Model"),
    "ColQwen3_5": ("colqwen3_5", "ColQwen3_5Model"),
    "Qwen3VLNemotronEmbedModel": ("colqwen3", "ColQwen3Model"),
}

_REWARD_MODELS = {
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "Qwen2ForProcessRewardModel": ("qwen2_rm", "Qwen2ForProcessRewardModel"),
}

_TOKEN_CLASSIFICATION_MODELS = {
    "Qwen3ASRForcedAlignerForTokenClassification": (
        "qwen3_asr_forced_aligner",
        "Qwen3ASRForcedAlignerForTokenClassification",
    ),
}

_SEQUENCE_CLASSIFICATION_MODELS = {}

_MULTIMODAL_MODELS = {
    "DeepseekVLV2ForCausalLM": ("deepseek_vl2", "DeepseekVLV2ForCausalLM"),
    "DeepseekOCRForCausalLM": ("deepseek_ocr", "DeepseekOCRForCausalLM"),
    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),
    "DeepseekV4ForConditionalGeneration": (
        "vllm.models.deepseek_v4",
        "DeepseekV4ForConditionalGeneration",
    ),
    "DeepseekV41ForCausalLM": (
        "vllm.models.deepseek_v41",
        "DeepseekV41ForCausalLM",
    ),
    "GlmAsrForConditionalGeneration": ("glmasr", "GlmAsrForConditionalGeneration"),
    "GLM4VForCausalLM": ("glm4v", "GLM4VForCausalLM"),
    "Glm4vForConditionalGeneration": ("glm4_1v", "Glm4vForConditionalGeneration"),
    "Glm4vMoeForConditionalGeneration": ("glm4_1v", "Glm4vMoeForConditionalGeneration"),
    "GlmOcrForConditionalGeneration": ("glm_ocr", "GlmOcrForConditionalGeneration"),
    "Glm5NextForConditionalGeneration": (
        "vllm.models.glm5next",
        "Glm5NextForConditionalGeneration",
    ),
    "KimiVLForConditionalGeneration": ("kimi_vl", "KimiVLForConditionalGeneration"),
    "KimiK25ForConditionalGeneration": ("kimi_k25", "KimiK25ForConditionalGeneration"),
    "KimiK3ForConditionalGeneration": (
        "vllm.models.kimi_k3",
        "KimiK3ForConditionalGeneration",
    ),
    "MoonshotKimiaForCausalLM": ("kimi_audio", "KimiAudioForConditionalGeneration"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
    "Qwen2_5_VLForConditionalGeneration": (
        "qwen2_5_vl",
        "Qwen2_5_VLForConditionalGeneration",
    ),
    "Qwen2AudioForConditionalGeneration": (
        "qwen2_audio",
        "Qwen2AudioForConditionalGeneration",
    ),
    "Qwen2_5OmniModel": (
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ),
    "Qwen3ASRForConditionalGeneration": (
        "qwen3_asr",
        "Qwen3ASRForConditionalGeneration",
    ),
    "Qwen3ASRRealtimeGeneration": ("qwen3_asr_realtime", "Qwen3ASRRealtimeGeneration"),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
    "Qwen3VLMoeForConditionalGeneration": (
        "qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
    ),
    "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    "Qwen3_5MoeForConditionalGeneration": (
        "qwen3_5",
        "Qwen3_5MoeForConditionalGeneration",
    ),
    "Qwen4ExpForConditionalGeneration": (
        "vllm.models.qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
    ),
}

_SPECULATIVE_DECODING_MODELS = {
    "ExtractHiddenStatesModel": ("extract_hidden_states", "ExtractHiddenStatesModel"),
    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),
    "DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"),
    "DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
    "DSparkV41DraftModel": (
        "vllm.models.deepseek_v41",
        "DSparkDeepseekV4ForCausalLM",
    ),
    "Qwen3DSparkModel": ("qwen3_dspark", "Qwen3DSparkForCausalLM"),
    "Qwen3OmniDSparkModel": ("qwen3_dspark", "Qwen3DSparkForCausalLM"),
    "K3DSparkModel": (
        "vllm.models.kimi_k3.nvidia.dspark_mla",
        "K3DSparkForCausalLM",
    ),
    "PEagleDraftModel": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "PeagleLlamaForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "Eagle3Qwen2_5vlForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "Eagle3Qwen3vlForCausalLM": ("llama_eagle3", "Eagle3LlamaForCausalLM"),
    "Eagle3Qwen3ForCausalLM": ("qwen3_eagle3", "Eagle3Qwen3ForCausalLM"),
    "PeagleQwen3ForCausalLM": ("qwen3_eagle3", "Eagle3Qwen3ForCausalLM"),
    "Eagle3DeepseekV2ForCausalLM": ("deepseek_eagle3", "Eagle3DeepseekV2ForCausalLM"),
    "Eagle3DeepseekV3ForCausalLM": ("deepseek_eagle3", "Eagle3DeepseekV2ForCausalLM"),
    "EagleDeepSeekMTPModel": ("deepseek_eagle", "EagleDeepseekV3ForCausalLM"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "DeepseekV32MTPModel": ("vllm.models.deepseek_v32", "DeepseekV32MTP"),
    "DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
    "Glm4MoeMTPModel": ("glm4_moe_mtp", "Glm4MoeMTP"),
    "Glm4MoeLiteMTPModel": ("glm4_moe_lite_mtp", "Glm4MoeLiteMTP"),
    "GlmOcrMTPModel": ("glm_ocr_mtp", "GlmOcrMTP"),
    "Glm5NextMTPModel": ("vllm.models.glm5next", "Glm5NextMTP"),
    "Qwen3NextMTP": ("qwen3_next_mtp", "Qwen3NextMTP"),
    "Qwen4ExpMTP": ("vllm.models.qwen4_exp", "Qwen4ExpMTP"),
    "Qwen3_5MTP": ("qwen3_5_mtp", "Qwen3_5MTP"),
    "Qwen3_5MoeMTP": ("qwen3_5_mtp", "Qwen3_5MoeMTP"),
    "KimiK3MTPModel": ("vllm.models.kimi_k3", "KimiK3MTP"),
    # Temporarily disabled.
    # # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.
    # "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
}

_TRANSFORMERS_SUPPORTED_MODELS = {
    "GlmForCausalLM": ("transformers", "TransformersForCausalLM"),
}

_TRANSFORMERS_BACKEND_MODELS = {
    # Text generation models
    "TransformersForCausalLM": ("transformers", "TransformersForCausalLM"),
    "TransformersMoEForCausalLM": ("transformers", "TransformersMoEForCausalLM"),
    # Multimodal models
    "TransformersMultiModalForCausalLM": (
        "transformers",
        "TransformersMultiModalForCausalLM",
    ),
    "TransformersMultiModalMoEForCausalLM": (
        "transformers",
        "TransformersMultiModalMoEForCausalLM",
    ),
    # Embedding models
    "TransformersEmbeddingModel": ("transformers", "TransformersEmbeddingModel"),
    "TransformersMoEEmbeddingModel": ("transformers", "TransformersMoEEmbeddingModel"),
    "TransformersMultiModalEmbeddingModel": (
        "transformers",
        "TransformersMultiModalEmbeddingModel",
    ),
    # Sequence classification models
    "TransformersForSequenceClassification": (
        "transformers",
        "TransformersForSequenceClassification",
    ),
    "TransformersMoEForSequenceClassification": (
        "transformers",
        "TransformersMoEForSequenceClassification",
    ),
    "TransformersMultiModalForSequenceClassification": (
        "transformers",
        "TransformersMultiModalForSequenceClassification",
    ),
}

_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_LATE_INTERACTION_MODELS,
    **_REWARD_MODELS,
    **_TOKEN_CLASSIFICATION_MODELS,
    **_SEQUENCE_CLASSIFICATION_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_TRANSFORMERS_SUPPORTED_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}

# Keep checkpoint admission independent of mutable plugin registrations. Generic
# Transformers classes are implementation targets, not checkpoint identities.
_FOUR_FAMILY_ARCHITECTURES = frozenset(_VLLM_MODELS) - frozenset(
    _TRANSFORMERS_BACKEND_MODELS
)


# A shared drafter identity is not an independently serviceable checkpoint.
# This pair is evidenced by the upstream PEagle Qwen3-8B configuration.
_SHARED_DRAFT_TARGETS = {
    "PeagleLlamaForCausalLM": frozenset({"Qwen3ForCausalLM"}),
    "PEagleDraftModel": frozenset({"Qwen3ForCausalLM"}),
    "DFlashDraftModel": frozenset({"Qwen3ForCausalLM"}),
    "DFlash2DraftModel": frozenset({"Qwen3_5ForConditionalGeneration"}),
}

# A family may reuse another family's registered implementation when the
# checkpoint architecture is explicitly evidenced. Kimi-K2.5 EAGLE3 ships a
# DeepSeek-V3-shaped draft, so only that exact pair is allowed.
_CROSS_FAMILY_DRAFT_TARGETS = {
    "Eagle3DeepseekV2ForCausalLM": frozenset({"KimiK25ForConditionalGeneration"}),
    "Eagle3DeepseekV3ForCausalLM": frozenset({"KimiK25ForConditionalGeneration"}),
}


def _is_target_repacked_eagle_draft(
    architecture: str, model_config: ModelConfig
) -> bool:
    """Admit only a draft identity that its approved target will repack.

    Some Qwen-VL and Kimi EAGLE3 checkpoints expose the intermediate
    ``LlamaForCausalLMEagle3`` identity before ``EAGLEConfig`` derives the
    target-dependent draft architecture. This is not a Llama checkpoint
    admission: the target is still checked independently, and the final draft
    architecture is checked again after ``update_arch_``.
    """
    return (
        architecture == "LlamaForCausalLMEagle3"
        and getattr(model_config, "runner", None) == "draft"
        and getattr(model_config, "speculative_method", None) in ("eagle", "eagle3")
        and bool(getattr(model_config, "spec_target_architectures", ()))
    )


def _validate_four_family_architectures(
    architectures: list[str], model_config: ModelConfig
) -> None:
    rejected = [
        arch
        for arch in architectures
        if arch not in _FOUR_FAMILY_ARCHITECTURES
        and not _is_target_repacked_eagle_draft(arch, model_config)
    ]
    for arch in architectures:
        if arch in _SHARED_DRAFT_TARGETS:
            targets = getattr(model_config, "spec_target_architectures", ())
            if (
                getattr(model_config, "runner", None) != "draft"
                or not targets
                or not set(targets) <= _SHARED_DRAFT_TARGETS[arch]
            ):
                rejected.append(arch)
    if rejected or model_config.model_impl == "terratorch":
        raise ValueError(
            f"Model architectures {architectures} are not approved for this build. "
            "Supported families: GLM, DeepSeek, Qwen, Kimi. "
            "Plugin registration does not grant checkpoint admission."
        )


# This variable is used as the args for subprocess.run(). We
# can modify  this variable to alter the args if needed. e.g.
# when we use par format to pack things together, sys.executable
# might not be the target we want to run.
_FOUR_FAMILY_ARCH_BY_FAMILY = {
    "GLM": frozenset(
        {
            "ChatGLMModel",
            "GLM4VForCausalLM",
            "GlmAsrForConditionalGeneration",
            "Glm4vForConditionalGeneration",
            "Glm4vMoeForConditionalGeneration",
            "GlmOcrForConditionalGeneration",
            "ChatGLMForConditionalGeneration",
            "GlmForCausalLM",
            "Glm4ForCausalLM",
            "Glm4MoeForCausalLM",
            "Glm4MoeLiteForCausalLM",
            "GlmMoeDsaForCausalLM",
            "Glm5NextForCausalLM",
            "Glm5NextForConditionalGeneration",
            "Glm4MoeMTPModel",
            "Glm4MoeLiteMTPModel",
            "GlmOcrMTPModel",
            "Glm5NextMTPModel",
        }
    ),
    "DeepSeek": frozenset(
        {
            "DeepseekVLV2ForCausalLM",
            "DeepseekOCRForCausalLM",
            "DeepseekOCR2ForCausalLM",
            "DeepseekV4ForConditionalGeneration",
            "DeepseekForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "DeepseekV4ForCausalLM",
            "DeepseekV41ForCausalLM",
            "DeepseekV3BidirectionalModel",
            "DSparkDraftModel",
            "DSparkV41DraftModel",
            "DeepSeekMTPModel",
            "DeepseekV32MTPModel",
            "DeepSeekV4MTPModel",
            "EagleDeepSeekMTPModel",
            "Eagle3DeepseekV2ForCausalLM",
            "Eagle3DeepseekV3ForCausalLM",
        }
    ),
    "Qwen": frozenset(
        {
            "ColQwen3",
            "OpsColQwen3Model",
            "ColQwen3_5",
            "Qwen3VLNemotronEmbedModel",
            "Qwen2ForRewardModel",
            "Qwen2ForProcessRewardModel",
            "Qwen3ASRForcedAlignerForTokenClassification",
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2_5OmniModel",
            "Qwen2_5OmniForConditionalGeneration",
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen3ASRForConditionalGeneration",
            "Qwen3ASRRealtimeGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
            "Qwen3OmniDSparkModel",
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen4ExpForCausalLM",
            "Qwen3NextForCausalLM",
            "Qwen2Model",
            "VoyageQwen3BidirectionalEmbedModel",
            "Qwen3NextMTP",
            "Qwen4ExpMTP",
            "Qwen3_5MTP",
            "Qwen3_5MoeMTP",
            "Qwen3DSparkModel",
            "Eagle3Qwen3ForCausalLM",
            "PeagleQwen3ForCausalLM",
            "Eagle3Qwen2_5vlForCausalLM",
            "Eagle3Qwen3vlForCausalLM",
        }
    ),
    "Kimi": frozenset(
        {
            "KimiVLForConditionalGeneration",
            "KimiK25ForConditionalGeneration",
            "MoonshotKimiaForCausalLM",
            "KimiLinearForCausalLM",
            "KimiK3ForConditionalGeneration",
            "KimiK3MTPModel",
            "K3DSparkModel",
        }
    ),
}
_FOUR_FAMILY_ARCH_TO_FAMILY = {
    arch: family
    for family, architectures in _FOUR_FAMILY_ARCH_BY_FAMILY.items()
    for arch in architectures
}


def _four_family_for_arch(arch: str) -> str | None:
    """Resolve an explicitly classified, registered target or draft identity."""
    for table in (
        _TEXT_GENERATION_MODELS,
        _EMBEDDING_MODELS,
        _LATE_INTERACTION_MODELS,
        _REWARD_MODELS,
        _TOKEN_CLASSIFICATION_MODELS,
        _SEQUENCE_CLASSIFICATION_MODELS,
        _MULTIMODAL_MODELS,
        _SPECULATIVE_DECODING_MODELS,
    ):
        if arch in table:
            if arch in _FOUR_FAMILY_ARCH_TO_FAMILY:
                return _FOUR_FAMILY_ARCH_TO_FAMILY[arch]
            return None
    return None


def validate_four_family_speculation(
    target_architectures: str | list[str],
    draft_architectures: str | list[str],
    *,
    method: str | None = None,
) -> None:
    """Bind a draft to an admitted target and reject cross-family drafts."""
    targets = (
        [target_architectures]
        if isinstance(target_architectures, str)
        else list(target_architectures)
    )
    drafts = (
        [draft_architectures]
        if isinstance(draft_architectures, str)
        else list(draft_architectures)
    )
    target_repacked_eagle = (
        method in ("eagle", "eagle3")
        and drafts == ["LlamaForCausalLMEagle3"]
        and bool(targets)
    )
    rejected = [
        arch
        for arch in targets + drafts
        if arch not in _FOUR_FAMILY_ARCHITECTURES and not target_repacked_eagle
    ]
    families = {_four_family_for_arch(arch) for arch in targets}
    if not drafts or rejected or len(families) != 1 or None in families:
        raise ValueError(
            f"Draft architectures {drafts} are not approved for target {targets}. "
            "Supported families: GLM, DeepSeek, Qwen, Kimi. "
            "Draft and target must match one admitted family."
        )
    if "DeepseekV41ForCausalLM" in targets and any("MTP" in arch for arch in drafts):
        raise ValueError(
            "DeepSeek V4.1 has no classic-MTP draft; use its DSpark checkpoint."
        )
    family = families.pop()
    for arch in drafts:
        if arch == "LlamaForCausalLMEagle3":
            # A target-bound repack may still expose this intermediate identity
            # here; update_arch_ revalidates the served draft architecture.
            continue
        if arch == "ExtractHiddenStatesModel":
            if method == "extract_hidden_states" and drafts == [arch]:
                continue
            raise ValueError(
                "ExtractHiddenStatesModel requires the extract_hidden_states "
                "method and an admitted target."
            )
        if arch in _SHARED_DRAFT_TARGETS:
            if set(targets) <= _SHARED_DRAFT_TARGETS[arch]:
                continue
            raise ValueError(
                f"Shared draft {arch} is not approved for target {targets}."
            )
        cross_family_targets = _CROSS_FAMILY_DRAFT_TARGETS.get(arch)
        if cross_family_targets is not None:
            if method in ("eagle", "eagle3") and set(targets) <= cross_family_targets:
                continue
            raise ValueError(
                f"Cross-family draft {arch} is not approved for target {targets}."
            )
        expected = _four_family_for_arch(arch)
        # Shared implementations need their actual target, not a name guess:
        # Qwen PEagle and Eagle3-VL reuse the generic llama_eagle3 module.
        if expected != family:
            raise ValueError(
                f"Draft architectures {drafts} do not belong to the approved "
                f"{family} family for target {targets}."
            )


_SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]

_PREVIOUSLY_SUPPORTED_MODELS = {
    "MotifForCausalLM": "0.10.2",
    "Phi3SmallForCausalLM": "0.9.2",
    "Phi4FlashForCausalLM": "0.10.2",
    "Phi4MultimodalForCausalLM": "0.12.0",
    "JAISLMHeadModel": "0.22.0",
    "ErnieModel": "0.23.0",
    "ErnieForSequenceClassification": "0.23.0",
    "ErnieForTokenClassification": "0.23.0",
    "InternLM2VEForCausalLM": "0.23.0",
    "QWenLMHeadModel": "0.23.0",
    "QwenVLForConditionalGeneration": "0.23.0",
    "InternLMForCausalLM": "0.23.0",
    # encoder-decoder models except whisper
    # have been removed for V0 deprecation.
    "DonutForConditionalGeneration": "0.10.2",
    "MllamaForConditionalGeneration": "0.10.2",
    "XverseForCausalLM": "0.23.0",
    "Dots1ForCausalLM": "0.23.0",
    "BambaForCausalLM": "0.23.0",
    "MiniMaxForCausalLM": "0.23.0",
    "MiniMaxText01ForCausalLM": "0.23.0",
    "MiniMaxM1ForCausalLM": "0.23.0",
    "MiniMaxVL01ForConditionalGeneration": "0.23.0",
    "BaiChuanForCausalLM": "0.23.0",
    "BaichuanForCausalLM": "0.23.0",
    "AquilaModel": "0.24.0",
    "AquilaForCausalLM": "0.24.0",
    "Grok1ModelForCausalLM": "0.24.0",
    "Grok1ForCausalLM": "0.24.0",
    "TarsierForConditionalGeneration": "0.24.0",
    "Tarsier2ForConditionalGeneration": "0.23.0",  # last version with Transformers v4
    "MantisForConditionalGeneration": "0.24.0",
    "MusicFlamingoForConditionalGeneration": "0.24.0",
    "AyaVisionForConditionalGeneration": "0.24.0",
    "TeleChatForCausalLM": "0.25.0",
    "PersimmonForCausalLM": "0.25.0",
    "FuyuForCausalLM": "0.25.0",
    "Plamo2ForCausalLM": "0.26.0",
    "OuroForCausalLM": "0.26.0",
    "ArcticForCausalLM": "0.28.0",
    "ChameleonForConditionalGeneration": "0.28.0",
    "Cheers": "0.28.0",
    "CheersForConditionalGeneration": "0.28.0",
    "Fairseq2LlamaForCausalLM": "0.28.0",
    "FireRedLIDForConditionalGeneration": "0.28.0",
    "GritLM": "0.28.0",
    "HCXVisionForCausalLM": "0.28.0",
    "MPTForCausalLM": "0.28.0",
    "MptForCausalLM": "0.28.0",
    "RWForCausalLM": "0.28.0",
    "StableLMEpochForCausalLM": "0.28.0",
    # Superseded by the generic "Terratorch" architecture
    "PrithviGeoSpatialMAE": "0.28.0",
}

_OOT_SUPPORTED_MODELS = {
    "BartModel": "https://github.com/vllm-project/bart-plugin",
    "BartForConditionalGeneration": "https://github.com/vllm-project/bart-plugin",
    "Florence2ForConditionalGeneration": "https://github.com/vllm-project/bart-plugin",
    "MBartForConditionalGeneration": "https://github.com/vllm-project/bart-plugin",
    "Nemotron3_5AsrForRNNT": "https://github.com/Sohaib-Ahmed21/vllm-nemotron-asr-plugin",
}


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    attn_type: AttnTypeStr
    default_seq_pooling_type: SequencePoolingType
    default_tok_pooling_type: TokenPoolingType
    score_type: ScoreType
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    supports_hc_sp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_replayssm: bool
    supports_transcription: bool
    supports_transcription_only: bool
    supported_video_pruning_methods: tuple[str, ...]
    supports_mm_device_do_normalize: bool

    @staticmethod
    def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            architecture=model.__name__,
            is_text_generation_model=is_text_generation_model(model),
            is_pooling_model=is_pooling_model(model),
            default_seq_pooling_type=get_default_seq_pooling_type(model),
            default_tok_pooling_type=get_default_tok_pooling_type(model),
            attn_type=get_attn_type(model),
            score_type=get_score_type(model),
            supports_multimodal=supports_multimodal(model),
            supports_multimodal_raw_input_only=supports_multimodal_raw_input_only(
                model
            ),
            requires_raw_input_tokens=requires_raw_input_tokens(model),
            supports_multimodal_encoder_tp_data=supports_multimodal_encoder_tp_data(
                model
            ),
            supports_pp=supports_pp(model),
            supports_hc_sp=supports_hc_sp(model),
            has_inner_state=has_inner_state(model),
            is_attention_free=is_attention_free(model),
            is_hybrid=is_hybrid(model),
            supports_mamba_prefix_caching=supports_mamba_prefix_caching(model),
            supports_replayssm=supports_replayssm(model),
            supports_transcription=supports_transcription(model),
            supports_transcription_only=(
                supports_transcription(model) and model.supports_transcription_only
            ),
            has_noops=has_noops(model),
            supported_video_pruning_methods=getattr(
                model, "supported_video_pruning_methods", ()
            ),
            supports_mm_device_do_normalize=getattr(
                model, "supports_mm_device_do_normalize", False
            ),
        )


class _BaseRegisteredModel(ABC):
    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """Represents a model that has already been imported in the main process."""

    interfaces: _ModelInfo
    model_cls: type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """Represents a model that has not been imported in the main process."""

    module_name: str
    class_name: str

    @staticmethod
    def _get_cache_dir() -> Path:
        return Path(envs.VLLM_CACHE_ROOT) / "modelinfos"

    def _get_cache_filename(self) -> str:
        cls_name = f"{self.module_name}-{self.class_name}".replace(".", "-")
        return f"{cls_name}.json"

    @staticmethod
    def _get_modelinfo_module_hash(model_path: Path) -> str:
        if model_path.name == "__init__.py":
            # Package entry points often re-export classes implemented in
            # submodules, so include the package contents in the cache key.
            module_paths = sorted(model_path.parent.rglob("*.py"))
            root_path = model_path.parent
        else:
            module_paths = [model_path]
            root_path = model_path.parent

        hasher = safe_hash(b"", usedforsecurity=False)
        for path in module_paths:
            hasher.update(path.relative_to(root_path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
        return hasher.hexdigest()

    def _load_modelinfo_from_cache(self, module_hash: str) -> _ModelInfo | None:
        try:
            try:
                modelinfo_path = self._get_cache_dir() / self._get_cache_filename()
                with open(modelinfo_path, encoding="utf-8") as file:
                    mi_dict = json.load(file)
            except FileNotFoundError:
                logger.debug(
                    "Cached model info file for class %s.%s not found",
                    self.module_name,
                    self.class_name,
                )
                return None

            if mi_dict["hash"] != module_hash:
                logger.debug(
                    "Cached model info file for class %s.%s is stale",
                    self.module_name,
                    self.class_name,
                )
                return None

            # file not changed, use cached _ModelInfo properties
            return _ModelInfo(**mi_dict["modelinfo"])
        except Exception:
            logger.debug(
                "Cached model info for class %s.%s error. ",
                self.module_name,
                self.class_name,
            )
            return None

    def _save_modelinfo_to_cache(self, mi: _ModelInfo, module_hash: str) -> None:
        """Save dictionary json file to cache."""
        from vllm.model_executor.model_loader.weight_utils import atomic_writer

        try:
            modelinfo_dict = {
                "hash": module_hash,
                "modelinfo": asdict(mi),
            }
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            modelinfo_path = cache_dir / self._get_cache_filename()
            with atomic_writer(modelinfo_path, encoding="utf-8") as f:
                json.dump(modelinfo_dict, f, indent=2)
        except Exception:
            logger.exception("Error saving model info cache.")

    @logtime(logger=logger, msg="Registry inspect model class")
    def inspect_model_cls(self) -> _ModelInfo:
        # Modules registered with a non-default location (e.g. the
        # hardware-isolated ``vllm.models.<name>`` layout) live outside
        # ``vllm/model_executor/models``. Resolve the module spec directly
        # so the file-hash cache stays warm for them.
        model_path: Path | None = None
        if self.module_name.startswith("vllm.model_executor.models."):
            model_path = Path(__file__).parent / f"{self.module_name.split('.')[-1]}.py"
        if model_path is None or not model_path.exists():
            try:
                spec = importlib.util.find_spec(self.module_name)
            except (ImportError, ValueError):
                spec = None
            model_path = Path(spec.origin) if spec is not None and spec.origin else None
        module_hash = None

        if model_path is not None and model_path.exists():
            module_hash = self._get_modelinfo_module_hash(model_path)

            mi = self._load_modelinfo_from_cache(module_hash)
            if mi is not None:
                logger.debug(
                    "Loaded model info for class %s.%s from cache",
                    self.module_name,
                    self.class_name,
                )
                return mi
            else:
                logger.debug(
                    "Cache model info for class %s.%s miss. Loading model instead.",
                    self.module_name,
                    self.class_name,
                )

        # Performed in another process to avoid initializing CUDA
        mi = _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls())
        )
        logger.debug(
            "Loaded model info for class %s.%s", self.module_name, self.class_name
        )

        # save cache file
        if module_hash is not None:
            self._save_modelinfo_to_cache(mi, module_hash)

        return mi

    def load_model_cls(self) -> type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    from vllm.platforms import current_platform

    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> _ModelInfo | None:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'", model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        """Register an external model to be used in vLLM.

        `model_cls` can be either:

        - A [`torch.nn.Module`][] class directly referencing the model.
        - A string in the format `<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if not isinstance(model_arch, str):
            msg = f"`model_arch` should be a string, not a {type(model_arch)}"
            raise TypeError(msg)

        if model_arch in self.models:
            logger.debug(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls,
            )

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
            model = _RegisteredModel.from_model_cls(model_cls)
        else:
            msg = (
                "`model_cls` should be a string or PyTorch model class, "
                f"not a {type(model_cls)}"
            )
            raise TypeError(msg)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        for arch in architectures:
            if arch in _PREVIOUSLY_SUPPORTED_MODELS:
                previous_version = _PREVIOUSLY_SUPPORTED_MODELS[arch]

                raise ValueError(
                    f"Model architecture {arch} was supported in vLLM until "
                    f"v{previous_version}, and is not supported anymore. "
                    "Please use an older version of vLLM if you want to "
                    "use this model architecture."
                )
            if arch in _OOT_SUPPORTED_MODELS:
                plugin_url = _OOT_SUPPORTED_MODELS[arch]

                raise ValueError(
                    f"Model architecture {arch} is not supported in-tree anymore. "
                    f"Please install the plugin at {plugin_url} if you want to "
                    "use this model architecture."
                )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> _ModelInfo | None:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _try_resolve_transformers(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> str | None:
        if architecture in _TRANSFORMERS_BACKEND_MODELS:
            return architecture

        auto_map: dict[str, str] = (
            getattr(model_config.hf_config, "auto_map", None) or dict()
        )

        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        for prefix in ("AutoConfig", "AutoModel"):
            for name, module in auto_map.items():
                if name.startswith(prefix):
                    try_get_class_from_dynamic_module(
                        module,
                        model_config.model,
                        revision=model_config.revision,
                        code_revision=model_config.code_revision,
                        trust_remote_code=model_config.trust_remote_code,
                        warn_on_fail=False,
                    )

        model_module = getattr(transformers, architecture, None)

        if model_module is None:
            for name, module in auto_map.items():
                if name.startswith("AutoModel"):
                    model_module = try_get_class_from_dynamic_module(
                        module,
                        model_config.model,
                        revision=model_config.revision,
                        code_revision=model_config.code_revision,
                        trust_remote_code=model_config.trust_remote_code,
                        warn_on_fail=True,
                    )
                    if model_module is not None:
                        break
            else:
                if model_config.model_impl != "transformers":
                    return None

                raise ValueError(
                    f"Cannot find model module. {architecture!r} is not a "
                    "registered model in the Transformers library (only "
                    "relevant if the model is meant to be in Transformers) "
                    "and 'AutoModel' is not present in the model config's "
                    "'auto_map' (relevant if the model is custom)."
                )

        assert issubclass(model_module, transformers.PreTrainedModel)
        transformers_model_cls: type[transformers.PreTrainedModel] = model_module
        if not (
            transformers_model_cls.is_backend_compatible()
            or transformers_model_cls._can_set_attn_implementation()
        ):
            if model_config.model_impl != "transformers":
                return None

            raise ValueError(
                f"The Transformers implementation of {architecture!r} "
                "is not compatible with vLLM."
            )

        return model_config._get_transformers_backend_cls()

    def _normalize_arch(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> str:
        if architecture in self.models:
            return architecture
        if _is_target_repacked_eagle_draft(architecture, model_config):
            # The checkpoint's intermediate EAGLE identity selects the family
            # implementation; EAGLEConfig replaces the served draft architecture.
            for candidate in ("PeagleLlamaForCausalLM", "Eagle3Qwen2_5vlForCausalLM"):
                if candidate in self.models:
                    return candidate

        # This may be called in order to resolve runner_type and convert_type
        # in the first place, in which case we consider the default match
        match = try_match_architecture_defaults(
            architecture,
            runner_type=getattr(model_config, "runner_type", None),
            convert_type=getattr(model_config, "convert_type", None),
        )
        if match:
            suffix, _ = match

            # Get the name of the base model to convert
            for repl_suffix, _ in iter_architecture_defaults():
                base_arch = architecture.replace(suffix, repl_suffix)
                if base_arch in self.models:
                    return base_arch

        return architecture

    def inspect_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[_ModelInfo, str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        _validate_four_family_architectures(architectures, model_config)

        # Require transformers impl
        if model_config.model_impl == "transformers":
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)
        elif model_config.model_impl == "terratorch":
            model_info = self._try_inspect_model_cls("Terratorch")
            assert model_info is not None
            return (model_info, "Terratorch")

        # Fallback to transformers impl (after resolving convert_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)

        for arch in architectures:
            normalized_arch = self._normalize_arch(arch, model_config)
            model_info = self._try_inspect_model_cls(normalized_arch)
            if model_info is not None:
                return (model_info, arch)

        # Fallback to transformers impl (before resolving runner_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_info = self._try_inspect_model_cls(arch)
                if model_info is not None:
                    return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[type[nn.Module], str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        _validate_four_family_architectures(architectures, model_config)

        # Require transformers impl
        if model_config.model_impl == "transformers":
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)
        elif model_config.model_impl == "terratorch":
            arch = "Terratorch"
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        # Fallback to transformers impl (after resolving convert_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
            and getattr(model_config, "convert_type", "none") == "none"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)

        for arch in architectures:
            normalized_arch = self._normalize_arch(arch, model_config)
            model_cls = self._try_load_model_cls(normalized_arch)
            if model_cls is not None:
                return (model_cls, arch)

        # Fallback to transformers impl (before resolving runner_type)
        if (
            all(arch not in self.models for arch in architectures)
            and model_config.model_impl == "auto"
        ):
            arch = self._try_resolve_transformers(architectures[0], model_config)
            if arch is not None:
                model_cls = self._try_load_model_cls(arch)
                if model_cls is not None:
                    return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_text_generation_model

    def is_pooling_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_pooling_model

    def is_multimodal_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_multimodal

    def is_multimodal_raw_input_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_multimodal_raw_input_only

    def is_pp_supported_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_pp

    def is_hc_sp_supported_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        """Check support for HC SP without changing MoE parallelism."""
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_hc_sp

    def model_has_inner_state(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.has_inner_state

    def is_attention_free_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_attention_free

    def is_hybrid_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_hybrid

    def is_noops_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.has_noops

    def is_transcription_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_transcription

    def is_transcription_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_transcription_only


def _resolve_module_name(mod_relname: str) -> str:
    # Allow registry entries to point at fully-qualified module paths (e.g.
    # ``vllm.models.deepseek_v4``) for models that live outside the legacy
    # ``vllm.model_executor.models`` flat layout.
    if mod_relname.startswith("vllm."):
        return mod_relname
    return f"vllm.model_executor.models.{mod_relname}"


ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=_resolve_module_name(mod_relname),
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
    }
)

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        import cloudpickle

        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(
            _SUBPROCESS_COMMAND, input=input_bytes, capture_output=True
        )

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(
                f"Error raised in subprocess:\n{returned.stderr.decode()}"
            ) from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


def _run() -> None:
    # Setup plugins
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
