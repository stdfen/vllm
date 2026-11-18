# SPDX-License-Identifier: Apache-2.0
"""Decode graph bounds: only select a twin when the entire batch fits."""
import os
from contextvars import ContextVar

def _bounds():
    values = sorted({int(v.strip()) for v in os.environ.get(
        "VLLM_SHORT_CTX_BOUNDS", "").split(",") if v.strip()})
    if any(v <= 0 for v in values):
        raise ValueError("short context bounds must be positive")
    return values

BOUNDS = _bounds()
MARGIN = int(os.environ.get("VLLM_SHORT_CTX_MARGIN", "16"))
if MARGIN < 0:
    raise ValueError("short context margin cannot be negative")
_capture = ContextVar("short_ctx_capture", default=None)
_dispatch = ContextVar("short_ctx_dispatch", default=None)

def capture_bound():
    return _capture.get()

def set_capture_bound(bound):
    _capture.set(bound)

def dispatch_max_seq_len():
    return _dispatch.get()

def set_dispatch_max_seq_len(length):
    _dispatch.set(length)
