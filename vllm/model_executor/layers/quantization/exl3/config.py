# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from typing import Any

import torch

from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class Exl3Config(QuantizationConfig):
    """Configuration for pre-quantized EXL3 checkpoints.

    Runtime methods are imported lazily so checkpoint metadata can be parsed and
    validated on CPU-only hosts without loading the EXL3 CUDA extension.
    """

    def __init__(
        self,
        bits: float | int = 2.9,
        codebook: str = "mul1",
        scope: str = "dsv41_packed",
        tensor_bits: dict[str, int] | None = None,
        head_bits: int = 6,
        mtp_bits: int = 4,
        num_hidden_layers: int = 40,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.avg_bits = float(bits)
        self.bits = self.avg_bits
        self.codebook = str(codebook).lower()
        self.scope = str(scope)
        self.tensor_bits = tensor_bits or {}
        self.head_bits = int(head_bits)
        self.mtp_bits = int(mtp_bits)
        self.num_hidden_layers = int(num_hidden_layers)
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mul1", "mcg"):
            raise ValueError(
                f"EXL3 codebook must be 'mul1' or 'mcg', but got {self.codebook!r}"
            )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skipped = {
            "avg_bits",
            "bits",
            "codebook",
            "quant_method",
            "scope",
            "tensor_storage",
            "head_bits",
            "mtp_bits",
        }
        storage = config.get("tensor_storage") or {}
        tensor_bits = {
            _normalize_prefix(prefix): int(entry["bits_per_weight"])
            for prefix, entry in storage.items()
            if entry.get("quant_format") == "exl3"
            and entry.get("bits_per_weight") is not None
        }
        return cls(
            bits=config.get("avg_bits", config.get("bits", 2.9)),
            codebook=config.get("codebook", "mul1"),
            scope=config.get("scope", "dsv41_packed"),
            tensor_bits=tensor_bits,
            head_bits=config.get("head_bits", 6),
            mtp_bits=config.get("mtp_bits", 4),
            num_hidden_layers=config.get("num_hidden_layers", 40),
            **{key: value for key, value in config.items() if key not in skipped},
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        if str((hf_quant_cfg or {}).get("quant_method", "")).lower() == "exl3":
            return "exl3"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, RoutedExperts):
            from .moe import Exl3MoEMethod

            return Exl3MoEMethod(layer.moe_config, self, prefix)
        if isinstance(layer, ParallelLMHead):
            from .linear import Exl3LinearMethod

            return Exl3LinearMethod(self, prefix)
        if isinstance(layer, LinearBase) and not should_skip_exl3_linear(prefix):
            from .linear import Exl3LinearMethod

            return Exl3LinearMethod(self, prefix)
        return None

    def bits_for_prefix(self, prefix: str) -> int:
        normalized = _normalize_prefix(prefix)
        candidates = _checkpoint_prefix_candidates(normalized)
        matched_bits = {
            self.tensor_bits[candidate]
            for candidate in candidates
            if candidate in self.tensor_bits
        }
        if len(matched_bits) == 1:
            return matched_bits.pop()
        if len(matched_bits) > 1:
            raise ValueError(
                f"fused EXL3 module {prefix!r} combines different bit-widths: "
                f"{sorted(matched_bits)}"
            )
        grouped = self.grouped_slices_for_prefix(prefix)
        if grouped:
            grouped_bits = {
                self.tensor_bits[item] for item in grouped if item in self.tensor_bits
            }
            if not grouped_bits and self._is_remapped_mtp_prefix(normalized):
                return self.head_bits
            if len(grouped_bits) != 1:
                raise ValueError(
                    f"grouped EXL3 module {prefix!r} combines different "
                    f"bit-widths: {sorted(grouped_bits)}"
                )
            return grouped_bits.pop()
        if normalized in ("head", "lm_head") or normalized.endswith(
            (".head", ".lm_head")
        ):
            return self.head_bits
        if normalized.startswith("mtp.") or ".mtp." in normalized:
            return self.mtp_bits
        if self._is_remapped_mtp_prefix(normalized):
            # In the Flash-1017 EXL3 checkpoint, remapped DSpark attention,
            # shared-expert and main projection linears use the 6-bit layout;
            # routed experts are handled separately by bits_for_moe_prefix.
            return self.head_bits
        raise ValueError(f"no EXL3 bit-width metadata for {prefix!r}")

    def _is_remapped_mtp_prefix(self, normalized: str) -> bool:
        """Recognize vLLM's remapped names for in-checkpoint DSpark weights.

        The checkpoint stores DSpark tensors outside ``tensor_storage`` and
        declares their common width through ``mtp_bits``.  vLLM constructs the
        drafter as ``model.main_proj`` plus layers appended after the target's
        hidden layers, so those runtime prefixes cannot be resolved by the
        ordinary per-tensor metadata lookup.
        """
        if normalized == "main_proj" or normalized.endswith(".main_proj"):
            return True
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", normalized)
        return match is not None and int(match.group(1)) >= self.num_hidden_layers

    def grouped_slices_for_prefix(self, prefix: str) -> tuple[str, ...]:
        normalized = _normalize_prefix(prefix)
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", normalized)
        if layer_match and int(layer_match.group(1)) >= self.num_hidden_layers:
            mtp_layer = int(layer_match.group(1)) - self.num_hidden_layers
            suffix = layer_match.group(2)
            if suffix == "attn.wo_a":
                # Flash-1017 stores the wide DSpark wo_a projection as eight
                # independently packed 1024-column slices, just like target
                # layers, but those MTP entries are absent from tensor_storage.
                return tuple(
                    f"mtp.{mtp_layer}.attn.wo_a.slice.{index}" for index in range(8)
                )
        checkpoint = normalized.removeprefix("model.")
        slice_root = f"{checkpoint}.slice."
        indexed: list[tuple[int, str]] = []
        for candidate in self.tensor_bits:
            if not candidate.startswith(slice_root):
                continue
            suffix = candidate.removeprefix(slice_root)
            if suffix.isdigit():
                indexed.append((int(suffix), candidate))
        indexed.sort()
        if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError(f"non-contiguous EXL3 grouped slices for {prefix!r}")
        return tuple(candidate for _, candidate in indexed)

    def bits_for_moe_prefix(self, prefix: str) -> int:
        """Resolve one layer's routed experts without guessing from avg_bits."""
        normalized = _normalize_prefix(prefix)
        if (
            normalized.startswith("mtp.")
            or ".mtp." in normalized
            or self._is_remapped_mtp_prefix(normalized)
        ):
            return self.mtp_bits
        checkpoint = normalized.removeprefix("model.")
        root = checkpoint.removesuffix(".experts") + ".experts."
        matched = {
            bits
            for candidate, bits in self.tensor_bits.items()
            if candidate.startswith(root)
            and candidate.rsplit(".", 1)[-1] in ("w1", "w2", "w3")
        }
        if not matched:
            raise ValueError(f"no EXL3 expert bit-width metadata for {prefix!r}")
        if len(matched) != 1:
            raise ValueError(
                f"EXL3 experts for {prefix!r} have mixed bit-widths: {sorted(matched)}"
            )
        return matched.pop()


def _normalize_prefix(prefix: str) -> str:
    normalized = prefix.replace("-", ".").lower()
    for root in ("language_model.", "model.language_model."):
        if normalized.startswith(root):
            normalized = normalized[len(root) :]
    return normalized


def _checkpoint_prefix_candidates(prefix: str) -> tuple[str, ...]:
    checkpoint = prefix.removeprefix("model.")
    candidates = [prefix, checkpoint]
    aliases = {
        ".attn.fused_wqa_wkv": (".attn.wq_a", ".attn.wkv"),
        ".compressor.fused_wkv_wgate": (".compressor.wkv", ".compressor.wgate"),
        ".shared_experts.gate_up_proj": (
            ".shared_experts.w1",
            ".shared_experts.w3",
        ),
        ".shared_experts.down_proj": (".shared_experts.w2",),
    }
    for model_suffix, checkpoint_suffixes in aliases.items():
        if checkpoint.endswith(model_suffix):
            root = checkpoint.removesuffix(model_suffix)
            candidates.extend(root + suffix for suffix in checkpoint_suffixes)
    return tuple(dict.fromkeys(candidates))


def should_skip_exl3_linear(prefix: str) -> bool:
    name = _normalize_prefix(prefix)
    if any(
        token in name
        for token in (
            "vision",
            "visual",
            "aligner",
            "markov_head",
            "confidence_head",
            "weights_proj",
        )
    ):
        return True
    if name == "embed" or name.endswith(".embed"):
        return "engram" not in name
    return bool(name.endswith(".gate") or ".ffn.gate" in name or ".mlp.gate" in name)
