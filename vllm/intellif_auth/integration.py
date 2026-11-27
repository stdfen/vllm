# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Version-specific vLLM hooks; inference paths never call the SDK."""

import functools

from .context import bootstrap_core, close_context
from .errors import LicenseConfigurationError, LicenseError


def core_init(method):
    @functools.wraps(method)
    def wrapped(self, vllm_config, executor_class, *args, **kwargs):
        if (executor_class.__module__, executor_class.__name__) not in {
            ("vllm.v1.executor.uniproc_executor", "UniProcExecutor"),
            ("vllm.v1.executor.multiproc_executor", "MultiprocExecutor"),
        }:
            raise LicenseConfigurationError()
        self._ifauth = bootstrap_core(vllm_config)
        self._ifauth_rejected = []
        try:
            result = method(self, vllm_config, executor_class, *args, **kwargs)
            self._ifauth.snapshot()
            return result
        except BaseException:
            executor = getattr(self, "model_executor", None)
            if executor is not None:
                executor.shutdown()
            close_context(self._ifauth)
            raise

    return wrapped


def core_add(method):
    @functools.wraps(method)
    def wrapped(self, request, *args, **kwargs):
        existing = self.scheduler.requests.get(request.request_id)
        if existing is not None and existing.resumable:
            # Later chunks of an admitted streaming-input session.
            return method(self, request, *args, **kwargs)
        try:
            self._ifauth.snapshot()
        except LicenseError as error:
            # A request-scoped result, not an exception escaping the core loop.
            if hasattr(self, "output_queue"):
                for client, output in error_outputs([(request, error.code)]).items():
                    self.output_queue.put_nowait((client, output))
            else:
                self._ifauth_rejected.append((request, error.code))
            return
        return method(self, request, *args, **kwargs)

    return wrapped


def error_outputs(requests):
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

    outputs = {}
    for request, code in requests:
        group = outputs.setdefault(
            request.client_index, EngineCoreOutputs(outputs=[], finished_requests=set())
        )
        group.outputs.append(
            EngineCoreOutput(
                request.request_id,
                [],
                finish_reason=FinishReason.ERROR,
                stop_reason=code,
            )
        )
        group.finished_requests.add(request.request_id)
    return outputs


def checked_worker(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        self._ifauth.snapshot()
        result = method(self, *args, **kwargs)
        self._ifauth.snapshot()
        return result

    return wrapped


def core_shutdown(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            close_context(getattr(self, "_ifauth", None))

    return wrapped
