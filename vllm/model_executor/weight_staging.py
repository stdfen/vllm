# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded host staging for mmap-backed checkpoint tensors."""

from __future__ import annotations

import os
import threading

import torch

_STAGE_LOCK = threading.Lock()
_STAGE_BUFFER: torch.Tensor | None = None


def _size_from_env(name: str, default_mib: int) -> int:
    value = int(os.environ.get(name, default_mib))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value * 1024 * 1024


def _should_stage(destination: torch.Tensor, source: torch.Tensor) -> bool:
    if (
        source.device.type != "cpu"
        or destination.device.type != "cuda"
        or source.numel() == 0
        or not source.is_contiguous()
        or not destination.is_contiguous()
    ):
        return False
    mode = os.environ.get("VLLM_PINNED_WEIGHT_STAGING", "auto").lower()
    if mode in ("0", "false", "off"):
        return False
    if mode not in ("1", "true", "on", "auto"):
        raise ValueError(
            "VLLM_PINNED_WEIGHT_STAGING must be auto, on, or off, "
            f"got {mode!r}"
        )
    if mode == "auto" and torch.cuda.get_device_capability(destination.device) < (
        12,
        1,
    ):
        return False
    threshold = _size_from_env("VLLM_PINNED_WEIGHT_STAGING_MIN_MIB", 1)
    return source.numel() * source.element_size() >= threshold


def copy_loaded_weight(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy a loaded tensor without pinning its mmap-backed source pages."""
    if source.dtype != destination.dtype:
        source = source.to(dtype=destination.dtype)
    if not _should_stage(destination, source):
        destination.copy_(source)
        return

    stage_bytes = _size_from_env("VLLM_PINNED_WEIGHT_STAGING_MIB", 64)
    stage_elements = max(1, stage_bytes // source.element_size())
    source_flat = source.view(-1)
    destination_flat = destination.view(-1)
    global _STAGE_BUFFER
    with _STAGE_LOCK:
        if _STAGE_BUFFER is None or _STAGE_BUFFER.numel() != stage_bytes:
            _STAGE_BUFFER = torch.empty(
                stage_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
        typed_stage = _STAGE_BUFFER[
            : stage_elements * source.element_size()
        ].view(source.dtype)
        for offset in range(0, source_flat.numel(), stage_elements):
            count = min(stage_elements, source_flat.numel() - offset)
            stage = typed_stage[:count]
            stage.copy_(source_flat[offset : offset + count])
            destination_flat[offset : offset + count].copy_(stage)
