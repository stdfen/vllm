# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative process-wide fencing for disk-backed Engram staging."""

import functools
import os
import threading
import weakref

import torch

# Readers and their proxy are process-wide; runner-local locks are insufficient.
_lock = threading.RLock()
_state = threading.local()
_pending: dict[int, torch.cuda.Event] = {}
_native_pending: dict[int, tuple[weakref.ReferenceType, torch.cuda.Event, str]] = {}
_tf_pending: dict[int, tuple[weakref.ReferenceType, torch.cuda.Event, str]] = {}


def _tf_batch_without_engram(self, name, args, kwargs):
    """Audited TP2 TensorFold batches, including mixed prefill, up to C4."""
    config = self.vllm_config
    spec = getattr(config, "speculative_config", None)
    parallel = getattr(config, "parallel_config", None)
    if (
        getattr(spec, "dspark_implementation", None) != "tensorfold_full"
        or getattr(spec, "dspark_verification_policy", "native") != "native"
        or getattr(spec, "enable_adaptive_verification", False)
        or getattr(config, "lora_config", None) is not None
        or getattr(config, "kv_transfer_config", None) is not None
        or any(
            getattr(parallel, key, 1) != 1
            for key in (
                "data_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or any(
            getattr(parallel, key, False)
            for key in (
                "enable_eplb",
                "enable_expert_parallel",
                "enable_batch_sharded_sampling",
            )
        )
        or getattr(parallel, "tensor_parallel_size", 0) != 2
        or torch.cuda.current_stream() != getattr(self, "main_stream", None)
        or getattr(self, "aux_output_connector", None) is not None
    ):
        return False
    proposer = getattr(self, "speculator", None)
    if type(proposer).__name__ != "TensorFoldDSV41Speculator":
        return False
    layers = getattr(
        getattr(getattr(proposer, "model", None), "model", None), "layers", ()
    )
    if not len(layers) or any(
        getattr(layer, "engram", None) is not None for layer in layers
    ):
        return False
    if name == "execute_model":
        scheduler = args[0] if args else kwargs.get("scheduler_output")
        intermediate = args[1] if len(args) > 1 else kwargs.get("intermediate_tensors")
        dummy = args[2] if len(args) > 2 else kwargs.get("dummy_run", False)
        profile = args[4] if len(args) > 4 else kwargs.get("is_profile", False)
        return (
            intermediate is None
            and not dummy
            and not profile
            and 1 <= len(getattr(scheduler, "num_scheduled_tokens", {})) <= 4
        )
    if name == "sample_tokens":
        batch = getattr(getattr(self, "execute_model_state", None), "input_batch", None)
        return 1 <= getattr(batch, "num_reqs", 0) <= 4
    return False


def _tf_execute(self, fn, args, kwargs, device):
    """Do not reuse any batch staging until the previous whole round finishes."""
    previous = _tf_pending.get(device)
    try:
        if previous is not None and previous[0]() is self:
            previous[1].synchronize()
            event = previous[1]
        else:
            torch.accelerator.synchronize()
            event = torch.cuda.Event()
        _state.depth = 1
        result = fn(self, *args, **kwargs)
        event.record(torch.cuda.current_stream())
        _tf_pending[device] = (weakref.ref(self), event, "target")
        return result
    except BaseException:
        torch.accelerator.synchronize()
        _tf_pending.pop(device, None)
        raise
    finally:
        _state.depth = 0


def _tf_sample(self, fn, args, kwargs, device):
    """TF's existing feedback reads own their waits; extend ownership to round end."""
    previous = _tf_pending[device]
    _tf_pending[device] = (previous[0], previous[1], "sampling")
    try:
        _state.depth = 1
        result = fn(self, *args, **kwargs)
        # Includes postprocess, drafter and KV updates, whose auxiliary streams
        # are joined before returning. AsyncOutput retains its own copy event
        # and source tensor references; its output buffers are not staging here.
        previous[1].record(torch.cuda.current_stream())
        _tf_pending[device] = (previous[0], previous[1], "round")
        return result
    except BaseException:
        torch.accelerator.synchronize()
        _tf_pending.pop(device, None)
        raise
    finally:
        _state.depth = 0


def _native_c1_without_engram(self, name, args, kwargs):
    """Only the audited native C1 drafter may submit past the target fence."""
    if getattr(_state, "native_sampling", False):
        return False
    config = self.vllm_config
    spec = getattr(config, "speculative_config", None)
    parallel = getattr(config, "parallel_config", None)
    if (
        getattr(spec, "dspark_implementation", None) != "native"
        or getattr(spec, "dspark_verification_policy", "native") != "native"
        or getattr(spec, "enable_adaptive_verification", False)
        or any(
            getattr(parallel, key, 1) != 1
            for key in (
                "data_parallel_size",
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or getattr(parallel, "tensor_parallel_size", 0) != 2
        or torch.cuda.current_stream() != getattr(self, "main_stream", None)
    ):
        return False
    proposer = getattr(self, "speculator", None)
    if type(proposer).__name__ != "DSparkSpeculator":
        return False
    backbone = getattr(getattr(proposer, "model", None), "model", None)
    layers = getattr(backbone, "layers", ())
    if not len(layers) or any(
        getattr(layer, "engram", None) is not None for layer in layers
    ):
        return False
    if name == "execute_model":
        scheduler = args[0] if args else kwargs.get("scheduler_output")
        return (
            not kwargs.get("dummy_run", False)
            and not kwargs.get("is_profile", False)
            and len(getattr(scheduler, "num_scheduled_tokens", {})) == 1
        )
    if name == "sample_tokens":
        batch = getattr(getattr(self, "execute_model_state", None), "input_batch", None)
        return getattr(batch, "num_reqs", 0) == 1
    return False


def _native_execute(self, fn, args, kwargs, device):
    """Main-stream completion covers every joined Engram prefetch consumer."""
    previous = _native_pending.get(device)
    try:
        if previous is not None and previous[0]() is self:
            previous[1].synchronize()
            event = previous[1]
        else:
            torch.accelerator.synchronize()
            event = torch.cuda.Event()
        _state.depth = 1
        result = fn(self, *args, **kwargs)
        event.record(torch.cuda.current_stream())
        _native_pending[device] = (weakref.ref(self), event, "target")
        return result
    except BaseException:
        torch.accelerator.synchronize()
        _native_pending.pop(device, None)
        raise
    finally:
        _state.depth = 0


def serialize_disk_engram(fn):
    """Protect staging ownership around outer runner calls.

    Device fences are the default. Event mode transfers ownership only after
    the prior runner's joined stream completes, retaining a device drain on
    exceptional exits. Never enter during external CUDA capture; decorate the
    entry that owns the entire capture instead.
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        config = getattr(getattr(self, "vllm_config", None), "engram_config", None)
        if not getattr(config, "disk_offload", False):
            return fn(self, *args, **kwargs)
        mode = os.environ.get("VLLM_DISK_ENGRAM_FENCE", "device")
        if mode not in ("device", "event", "native_target"):
            raise ValueError(
                "VLLM_DISK_ENGRAM_FENCE must be device, event or native_target"
            )
        with _lock:
            depth = getattr(_state, "depth", 0)
            if depth == 0 and mode == "native_target":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Disk Engram runner entry during external capture"
                    )
                device = torch.accelerator.current_device_index()
                if _tf_batch_without_engram(self, fn.__name__, args, kwargs):
                    if device in _native_pending or device in _pending:
                        torch.accelerator.synchronize()
                        _native_pending.pop(device, None)
                        _pending.pop(device, None)
                    if fn.__name__ == "execute_model":
                        return _tf_execute(self, fn, args, kwargs, device)
                    previous = _tf_pending.get(device)
                    if (
                        previous is not None
                        and previous[0]() is self
                        and previous[2] == "target"
                    ):
                        return _tf_sample(self, fn, args, kwargs, device)
            if depth == 0 and _tf_pending:
                device = torch.accelerator.current_device_index()
                if device in _tf_pending:
                    # Administrative calls, unsupported batches/streams, mode
                    # switches and owner changes establish conservative ownership.
                    torch.accelerator.synchronize()
                    _tf_pending.pop(device, None)
                    _native_pending.pop(device, None)
                    _pending.pop(device, None)
            if depth == 0 and mode == "native_target":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Disk Engram runner entry during external capture"
                    )
                device = torch.accelerator.current_device_index()
                eligible = _native_c1_without_engram(self, fn.__name__, args, kwargs)
                if eligible and fn.__name__ == "execute_model":
                    return _native_execute(self, fn, args, kwargs, device)
                previous = _native_pending.get(device)
                if (
                    eligible
                    and previous is not None
                    and previous[0]() is self
                    and previous[2] == "target"
                ):
                    # Sampling uses no Engram table. Upstream stream ordering owns
                    # its GPU inputs; the next target waits before CPU staging reuse.
                    _native_pending[device] = (previous[0], previous[1], "sampled")
                    _state.native_sampling = True
                    try:
                        return fn(self, *args, **kwargs)
                    except BaseException:
                        torch.accelerator.synchronize()
                        _native_pending.pop(device, None)
                        raise
                    finally:
                        _state.native_sampling = False
                _native_pending.pop(device, None)
                mode = "device"
            elif depth == 0 and _native_pending:
                device = torch.accelerator.current_device_index()
                if device in _native_pending:
                    # Mode changes and other owners must also finish the drafter.
                    if mode == "event":
                        torch.accelerator.synchronize()
                        _pending.pop(device, None)
                    _native_pending.pop(device, None)
            event = None
            if depth == 0:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Disk Engram runner entry during external capture"
                    )
                if mode == "event":
                    device = torch.accelerator.current_device_index()
                    event = _pending.get(device)
                    if event is None:
                        # Establish ownership once, including previously submitted work.
                        torch.accelerator.synchronize()
                        event = torch.cuda.Event()
                        _pending[device] = event
                    else:
                        # Engram prefetch streams are joined before runner completion.
                        # Keep staging owned until the previous call finishes.
                        event.synchronize()
                else:
                    torch.accelerator.synchronize()
            _state.depth = depth + 1
            failed = True
            try:
                result = fn(self, *args, **kwargs)
                failed = False
                return result
            finally:
                try:
                    if depth == 0:
                        if event is None or failed:
                            torch.accelerator.synchronize()
                            if event is not None:
                                _pending.pop(device, None)
                        else:
                            try:
                                event.record(torch.cuda.current_stream())
                            except Exception:
                                torch.accelerator.synchronize()
                                _pending.pop(device, None)
                                raise
                finally:
                    _state.depth = depth

    return wrapped
