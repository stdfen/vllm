# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative process-wide fencing for disk-backed Engram staging."""

import functools
import threading

import torch

# Readers and their proxy are process-wide; runner-local locks are insufficient.
_lock = threading.RLock()
_state = threading.local()


def serialize_disk_engram(fn):
    """Drain device work around outer runner calls, including exceptional exits.

    This intentionally trades overlap for correctness. It is not a throughput
    equivalent of a device-side staging ownership protocol. Never use inside an
    active CUDA capture; decorate the entry that owns the entire capture instead.
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        config = getattr(getattr(self, "vllm_config", None), "engram_config", None)
        if not getattr(config, "disk_offload", False):
            return fn(self, *args, **kwargs)
        with _lock:
            depth = getattr(_state, "depth", 0)
            if depth == 0:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Disk Engram runner entry during external capture")
                torch.accelerator.synchronize()
            _state.depth = depth + 1
            try:
                return fn(self, *args, **kwargs)
            finally:
                try:
                    if depth == 0:
                        torch.accelerator.synchronize()
                finally:
                    _state.depth = depth
    return wrapped
