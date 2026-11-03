# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-memory Engram row reads from native safetensors tables."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def resolve_engram_tensor_names(
    weight_map: dict[str, str], tensor_prefix: str
) -> tuple[str, str]:
    """Resolve the model wrapper prefix without accepting fuzzy matches."""
    prefixes = [tensor_prefix]
    for wrapper in (
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
    ):
        if tensor_prefix.startswith(wrapper):
            prefixes.append(tensor_prefix.removeprefix(wrapper))
    pairs = [
        (f"{prefix}.weight", f"{prefix}.scale")
        for prefix in dict.fromkeys(prefixes)
        if f"{prefix}.weight" in weight_map and f"{prefix}.scale" in weight_map
    ]
    if len(pairs) != 1:
        raise ValueError(
            f"expected one Engram weight/scale pair for {tensor_prefix!r}, "
            f"found {pairs}"
        )
    return pairs[0]


class SafetensorsEngramStore:
    """Keep file-backed tensor views and materialize only requested rows."""

    def __init__(
        self,
        table_dir: str | Path,
        tensor_prefix: str,
        vocab_start: int,
        vocab_end: int,
        dim: int,
        block_size: int,
    ) -> None:
        root = Path(table_dir)
        index_path = root / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        try:
            weight_name, scale_name = resolve_engram_tensor_names(
                weight_map, tensor_prefix
            )
        except ValueError as exc:
            raise ValueError(f"Engram tensors invalid in {index_path}: {exc}") from exc
        self._weight_handle = safe_open(
            root / weight_map[weight_name], framework="pt", device="cpu"
        )
        self._scale_handle = safe_open(
            root / weight_map[scale_name], framework="pt", device="cpu"
        )
        weight = self._weight_handle.get_tensor(weight_name)
        scales = self._scale_handle.get_tensor(scale_name)
        if tuple(weight.shape[1:]) != (dim,):
            raise ValueError(f"unexpected Engram weight shape {tuple(weight.shape)}")
        if tuple(scales.shape[1:]) != (dim // block_size,):
            raise ValueError(f"unexpected Engram scale shape {tuple(scales.shape)}")
        if not 0 <= vocab_start < vocab_end <= weight.shape[0]:
            raise ValueError(
                f"invalid Engram row range [{vocab_start}, {vocab_end}) for "
                f"{weight.shape[0]} rows"
            )
        self.weight = weight[vocab_start:vocab_end]
        self.scales = scales[vocab_start:vocab_end]
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.dim = dim
        self.block_size = block_size

    @property
    def mapped_bytes(self) -> int:
        return self.weight.nbytes + self.scales.nbytes

    def lookup(
        self,
        indices: torch.Tensor,
        head_start: int,
        local_heads: int,
        out: torch.Tensor,
    ) -> None:
        """Synchronously stage requested rows; no whole-table pin or copy."""
        selected = indices[:, head_start : head_start + local_heads]
        local = selected.to(device="cpu", dtype=torch.long) - self.vocab_start
        valid = (local >= 0) & (local < self.weight.shape[0])
        safe = local.clamp(min=0, max=self.weight.shape[0] - 1)
        values = self.weight[safe]
        scales = self.scales[safe]
        if scales.dtype == torch.float8_e8m0fnu:
            scales = scales.view(torch.uint8)
        values = values.to(device=out.device, non_blocking=False).float()
        scales = scales.to(device=out.device, dtype=torch.int32, non_blocking=False)
        scale_values = (scales << 23).view(torch.float32)
        decoded = values * scale_values.repeat_interleave(self.block_size, dim=-1)
        decoded.masked_fill_(~valid.to(out.device)[..., None], 0)
        out.copy_(decoded.to(out.dtype))
