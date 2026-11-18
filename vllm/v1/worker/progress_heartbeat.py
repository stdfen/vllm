# SPDX-License-Identifier: Apache-2.0
"""Optional progress and separately recorded worker-thread liveness.

The consumer must compare monotonic timestamps on the SAME host, and bind the
output directory to a run-specific, empty host directory. This is not a GPU
completion heartbeat and does not replace driver/cgroup monitoring.
"""
import json
import os
import time
import threading
from pathlib import Path

_last = 0.0

def beat(phase: str, *, force: bool = False) -> None:
    global _last
    directory = os.environ.get("VLLM_PROGRESS_HEARTBEAT_DIR")
    if not directory:
        return
    now = time.monotonic()
    if not force and now - _last < 1.0:
        return
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("progress heartbeat requires initialized distributed rank")
    rank = dist.get_rank()
    target = Path(directory) / f"rank-{rank}.json"
    tmp = Path(directory) / f".rank-{rank}.{os.getpid()}.tmp"
    data = json.dumps({"rank": rank, "pid": os.getpid(), "phase": phase,
                       "monotonic": now}) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    _last = now

_liveness_thread = None
_liveness_stop = None


def start_liveness(rank: int) -> None:
    """Publish process-thread liveness independently of model progress.

    A live thread is not proof of GPU completion or main-thread progress.
    Consumers must separately enforce active-work progress deadlines.
    """
    global _liveness_thread, _liveness_stop
    directory = os.environ.get("VLLM_PROGRESS_HEARTBEAT_DIR")
    if not directory:
        return
    if type(rank) is not int or rank < 0 or _liveness_thread is not None:
        raise RuntimeError("invalid or duplicate liveness publisher")
    target = Path(directory) / f"liveness-{rank}.json"
    stop = threading.Event()

    def publish():
        tmp = Path(directory) / f".liveness-{rank}.{os.getpid()}.tmp"
        while not stop.is_set():
            try:
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump({"rank": rank, "pid": os.getpid(),
                               "monotonic": time.monotonic()}, f)
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            stop.wait(5)

    _liveness_stop = stop
    _liveness_thread = threading.Thread(target=publish, name="vllm-liveness", daemon=True)
    _liveness_thread.start()


def stop_liveness() -> None:
    global _liveness_thread, _liveness_stop
    if _liveness_thread is not None:
        _liveness_stop.set()
        _liveness_thread.join(timeout=6)
        if _liveness_thread.is_alive():
            raise RuntimeError("liveness publisher did not stop")
        _liveness_thread = _liveness_stop = None

def gpu_completed(phase: str) -> None:
    """Validation-mode device completion, never merely Python return.

    Deliberately synchronous when telemetry is enabled; not a performance-
    equivalent production monitoring implementation. A hung synchronize leaves
    progress stale even while the independent liveness publisher remains live.
    """
    if not os.environ.get("VLLM_PROGRESS_HEARTBEAT_DIR"):
        return
    import torch
    torch.accelerator.synchronize()
    beat(phase, force=True)