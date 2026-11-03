# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch


def create_exl3_linear_backend(
    *,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    marker: torch.Tensor,
    bits: int,
    codebook: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create the CUDA backend after weights have reached their final device."""
    if not trellis.is_cuda or not suh.is_cuda or not svh.is_cuda:
        raise RuntimeError(
            "EXL3 linear weights must be on CUDA before backend creation"
        )
    if trellis.dtype != torch.int16:
        raise TypeError(f"EXL3 trellis must be int16, got {trellis.dtype}")
    if suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise TypeError("EXL3 SUH and SVH must be float16")
    if trellis.ndim != 3:
        raise ValueError(f"EXL3 trellis must be rank 3, got {trellis.ndim}")
    inferred_bits = trellis.shape[-1] // 16
    if inferred_bits != bits:
        raise ValueError(
            f"EXL3 trellis/bit mismatch: shape={tuple(trellis.shape)}, bits={bits}"
        )
    expected_markers = {"mul1": -2082680531, "mcg": -877912083}
    if codebook not in expected_markers:
        raise ValueError(f"unsupported EXL3 codebook: {codebook!r}")
    actual_marker = int(marker.item())
    if actual_marker != expected_markers[codebook]:
        raise ValueError(
            f"invalid EXL3 {codebook} marker: {actual_marker}; "
            f"expected {expected_markers[codebook]}"
        )
    try:
        extension = importlib.import_module("exllamav3_ext")
    except ImportError as exc:
        raise RuntimeError(
            "the pinned ExLlamaV3 CUDA extension is required for EXL3"
        ) from exc
    transformed_input = torch.empty(
        (1, suh.numel()), dtype=torch.float16, device=trellis.device
    )
    inner = extension.BC_LinearEXL3(
        trellis.contiguous(),
        suh.contiguous(),
        svh.contiguous(),
        bits,
        None,
        codebook == "mcg",
        codebook == "mul1",
        transformed_input,
    )

    def apply(x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        output = inner.run_alloc(x.contiguous().to(torch.float16), svh.numel(), True)
        return output.to(source_dtype)

    # Keep graph workspace and extension handle alive with the callable.
    apply._exl3_inner = inner  # type: ignore[attr-defined]
    apply._exl3_xh = transformed_input  # type: ignore[attr-defined]
    return apply
