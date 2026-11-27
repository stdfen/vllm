# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests of the actual licensed vLLM classes; no GPU or real SDK calls."""

import os
import time
import unittest
from unittest.mock import patch

from vllm import SamplingParams
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.intellif_auth.errors import LicenseError
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector


class FrameworkTests(unittest.TestCase):
    def test_core_admission_and_inproc_rejection_use_one_check(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from vllm.v1.engine.core import EngineCore
        from vllm.v1.engine.core_client import InprocClient

        core = object.__new__(EngineCore)
        core._ifauth = SimpleNamespace(snapshot=Mock())
        core._ifauth_rejected = []
        core.scheduler = SimpleNamespace(requests={}, add_request=Mock())
        client = object.__new__(InprocClient)
        client.engine_core = core
        req = SimpleNamespace(
            request_id="licensed",
            client_index=0,
            pooling_params=None,
            kv_transfer_params=None,
            ec_transfer_params=None,
            abort_immediately=False,
        )
        core.add_request(req)
        core._ifauth.snapshot.assert_called_once_with()
        core.scheduler.add_request.assert_called_once_with(req)
        core._ifauth.snapshot.side_effect = LicenseError("LICENSE_UNAVAILABLE")
        denied = SimpleNamespace(request_id="rejected", client_index=0)
        core.add_request(denied)
        result = client.get_output()
        self.assertEqual(result.outputs[0].stop_reason, "LICENSE_UNAVAILABLE")
        self.assertEqual(core._ifauth.snapshot.call_count, 2)
        core.scheduler.add_request.assert_called_once_with(req)

    def request(self):
        return EngineCoreRequest(
            request_id="license-test",
            external_req_id="license-test",
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=4, temperature=0, detokenize=False
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    def test_offline_output_rejection_releases_actual_request_state(self):
        processor = OutputProcessor(None, log_stats=False)
        request = self.request()
        processor.add_request(request, None)
        with self.assertRaisesRegex(LicenseError, "LICENSE_DENIED"):
            processor.process_outputs(
                [
                    EngineCoreOutput(
                        request.request_id,
                        [],
                        finish_reason=FinishReason.ERROR,
                        stop_reason="LICENSE_DENIED",
                    )
                ]
            )
        self.assertEqual(processor.get_num_unfinished_requests(), 0)
        self.assertEqual(dict(processor.external_req_ids), {})

    def test_online_collector_receives_domain_error_and_state_is_released(self):
        processor = OutputProcessor(None, log_stats=False)
        request = self.request()
        collector = RequestOutputCollector(
            RequestOutputKind.FINAL_ONLY, request.request_id
        )
        processor.add_request(request, None, queue=collector)
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request.request_id,
                    [],
                    finish_reason=FinishReason.ERROR,
                    stop_reason="LICENSE_UNAVAILABLE",
                )
            ]
        )
        with self.assertRaisesRegex(LicenseError, "LICENSE_UNAVAILABLE"):
            collector.get_nowait()
        self.assertEqual(processor.get_num_unfinished_requests(), 0)

    def test_stable_http_status_does_not_leak_internal_sdk_details(self):
        for code, status in [("LICENSE_DENIED", 403), ("LICENSE_UNAVAILABLE", 503)]:
            response = create_error_response(LicenseError(code))
            self.assertEqual(response.error.code, status)
            self.assertEqual(response.error.type, code)

    def test_real_engine_constructor_rejects_before_executor_or_gpu(self):
        from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        config = VllmConfig(
            device_config=DeviceConfig("cuda"),
            parallel_config=ParallelConfig(distributed_executor_backend="uni"),
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(LicenseError):
            EngineCore(config, UniProcExecutor, log_stats=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
