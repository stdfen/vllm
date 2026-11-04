# SPDX-License-Identifier: Apache-2.0
"""Short-context CUDA graph capture and dispatch bounds."""

import os


def _parse_bounds(value: str) -> list[int]:
    bounds = sorted({int(item) for item in value.replace(" ", "").split(",") if item})
    if any(bound <= 0 for bound in bounds):
        raise ValueError("VLLM_SHORT_CTX_BOUNDS values must be positive")
    return bounds


BOUNDS = _parse_bounds(os.environ.get("VLLM_SHORT_CTX_BOUNDS", ""))
MARGIN = int(os.environ.get("VLLM_SHORT_CTX_MARGIN", "16"))
if MARGIN < 0:
    raise ValueError("VLLM_SHORT_CTX_MARGIN must be non-negative")

_capture_bound: int | None = None
_dispatch_max_seq_len: int | None = None


def capture_bound() -> int | None:
    return _capture_bound


def set_capture_bound(bound: int | None) -> None:
    global _capture_bound
    _capture_bound = bound


def dispatch_max_seq_len() -> int | None:
    return _dispatch_max_seq_len


def set_dispatch_max_seq_len(length: int | None) -> None:
    global _dispatch_max_seq_len
    _dispatch_max_seq_len = length
