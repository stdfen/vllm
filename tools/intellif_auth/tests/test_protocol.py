# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression coverage for request-only IFAuth admission."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import auth_test_imports  # noqa: F401
from intellif_auth.errors import LicenseError
from intellif_auth.integration import core_add


class Engine:
    def __init__(self):
        self.scheduler = SimpleNamespace(requests={})
        self._ifauth_rejected = []
        self._ifauth = SimpleNamespace(snapshot=self.check)
        self.checks = 0
        self.denied = False
        self.steps = 0

    def check(self):
        self.checks += 1
        if self.denied:
            raise LicenseError("LICENSE_DENIED")

    @core_add
    def add(self, request):
        self.scheduler.requests[request.request_id] = request

    def step(self):
        self.steps += 1
        return list(self.scheduler.requests)


def request(name, resumable=False):
    return SimpleNamespace(request_id=name, resumable=resumable)


class ProtocolTests(unittest.TestCase):
    def test_running_request_continues_and_new_request_is_rejected(self):
        engine = Engine()
        engine.add(request("existing"))
        engine.denied = True
        self.assertEqual(engine.step(), ["existing"])
        engine.add(request("denied"))
        self.assertEqual(engine.step(), ["existing"])
        self.assertEqual(engine.checks, 2)
        denied, code = engine._ifauth_rejected[0]
        self.assertEqual((denied.request_id, code), ("denied", "LICENSE_DENIED"))
        engine.denied = False
        engine.add(request("recovered"))
        self.assertEqual(engine.step(), ["existing", "recovered"])

    def test_async_rejection_is_queued_without_admission(self):
        engine = Engine()
        engine.output_queue = queue.Queue()
        engine.denied = True
        with patch(
            "intellif_auth.integration.error_outputs", return_value={4: "result"}
        ) as render:
            candidate = request("rejected")
            engine.add(candidate)
        render.assert_called_once_with([(candidate, "LICENSE_DENIED")])
        self.assertEqual(engine.output_queue.get_nowait(), (4, "result"))
        self.assertFalse(engine.scheduler.requests)

    def test_streaming_continuation_keeps_original_admission(self):
        engine = Engine()
        engine.add(request("stream", resumable=True))
        engine.denied = True
        engine.add(request("stream", resumable=True))
        self.assertEqual(engine.checks, 1)
        self.assertFalse(engine._ifauth_rejected)


if __name__ == "__main__":
    unittest.main()
