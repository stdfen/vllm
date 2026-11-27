# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os
import threading

from .config import AuthConfig
from .errors import LicenseConfigurationError
from .manager import LicenseManager
from .transport import ControlServer, StateClient

_lock = threading.RLock()
_contexts = {}


def validate_topology(config):
    p = config.parallel_config
    if (
        p.distributed_executor_backend not in ("uni", "mp")
        or p.data_parallel_size != 1
        or p.pipeline_parallel_size != 1
    ):
        raise LicenseConfigurationError()
    if config.device_config.device_type != "cuda" or p.enable_fault_tolerance:
        raise LicenseConfigurationError()
    if (
        getattr(config, "kv_transfer_config", None) is not None
        or getattr(config, "ec_transfer_config", None) is not None
        or getattr(p, "decode_context_parallel_size", 1) != 1
        or getattr(p, "prefill_context_parallel_size", 1) != 1
        or getattr(config.model_config, "enable_sleep_mode", False)
    ):
        raise LicenseConfigurationError()
    if (
        p.worker_cls not in ("auto", "vllm.v1.worker.gpu_worker.Worker")
        or p.worker_extension_cls
    ):
        raise LicenseConfigurationError()
    # Do not fork an initialized native SDK or a process with control threads.
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") != "spawn":
        raise LicenseConfigurationError()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    p.distributed_timeout_seconds = min(p.distributed_timeout_seconds or 60, 60)
    p.cpu_distributed_timeout_seconds = min(p.cpu_distributed_timeout_seconds or 60, 60)


class AuthContext:
    def __init__(self, config, owner=False, nodes=1, world_size=1):
        self.config = config
        self.manager = None
        self.server = None
        self.client = None
        try:
            if owner:
                self.manager = LicenseManager(config)
                self.server = ControlServer(
                    config, self.manager.store, nodes, world_size
                )
                self.manager.wait_initial()
            self.client = StateClient(config)
            self.client.wait_initial()
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def snapshot(self):
        if self.manager is not None:
            return self.manager.store.snapshot().check()
        return self.client.snapshot()

    def close(self):
        if self.client is not None:
            self.client.close()
        if self.server is not None:
            self.server.close()
        if self.manager is not None:
            self.manager.close()


def attach(owner=False, vllm_config=None):
    config = AuthConfig.load()
    identity = (os.getpid(), config.deployment_id)
    with _lock:
        existing = _contexts.get(identity)
        if existing is not None:
            if existing.config.policy_hash != config.policy_hash or (
                owner and existing.manager is None
            ):
                raise LicenseConfigurationError()
            return existing
        p = vllm_config.parallel_config if vllm_config is not None else None
        context = AuthContext(
            config, owner, p.nnodes if p else 1, p.world_size if p else 1
        )
        _contexts[identity] = context
        return context


def bootstrap_core(config):
    validate_topology(config)
    if config.parallel_config.node_rank != 0:
        raise LicenseConfigurationError()
    return attach(owner=True, vllm_config=config)


def bootstrap_executor(config):
    validate_topology(config)
    context = attach(vllm_config=config)
    p = config.parallel_config
    context.client.join(p.node_rank, p.nnodes, p.world_size)
    return context


def close_context(context):
    if context is None:
        return
    with _lock:
        _contexts.pop((os.getpid(), context.config.deployment_id), None)
        context.close()
