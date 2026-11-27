# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import auth_test_imports  # noqa: F401
from intellif_auth.config import PRODUCT, AuthConfig
from intellif_auth.errors import LicenseError
from intellif_auth.manager import LicenseManager, SDKProcess
from intellif_auth.state import RemoteCache, StateStore
from intellif_auth.transport import ControlServer, StateClient, pack, receive


def config(**kw):
    values = dict(
        addrs="test-service",
        deployment_id="unit-tests",
        host="127.0.0.1",
        port=29795,
        key=b"x" * 32,
        interval=0.02,
        call_timeout=0.1,
        max_age=0.2,
        startup_timeout=0.5,
        recovery_successes=2,
    )
    values.update(kw)
    return AuthConfig(**values)


class ConfigTests(unittest.TestCase):
    def test_verified_vllm_product(self):
        self.assertEqual(PRODUCT, "ifaas-vllm")

    def test_missing_fails(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(LicenseError):
            AuthConfig.load()

    def test_file_env_priority_and_no_disable(self):
        with tempfile.TemporaryDirectory() as folder:
            key = pathlib.Path(folder) / "key"
            key.write_bytes(b"x" * 32)
            path = pathlib.Path(folder) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "IFAUTH_ADDRS": "file-service",
                        "IFAUTH_DEPLOYMENT_ID": "test-config",
                        "IFAUTH_CONTROL_KEY_FILE": str(key),
                    }
                )
            )
            with patch.dict(
                os.environ,
                {"IFAUTH_CONFIG_FILE": str(path), "IFAUTH_ADDRS": "env-service"},
                clear=True,
            ):
                self.assertEqual(AuthConfig.load().addrs, "env-service")
            with (
                patch.dict(os.environ, {"IFAUTH_DISABLE": "1"}, clear=True),
                self.assertRaises(LicenseError),
            ):
                AuthConfig.load()

    def test_nonfinite_and_unbounded(self):
        with tempfile.TemporaryDirectory() as folder:
            key = pathlib.Path(folder) / "key"
            key.write_bytes(b"x" * 32)
            env = {
                "IFAUTH_ADDRS": "test",
                "IFAUTH_DEPLOYMENT_ID": "test-case",
                "IFAUTH_CONTROL_KEY_FILE": str(key),
            }
            for value in ("nan", "inf", "999999", "0", "-1"):
                with (
                    self.subTest(value=value),
                    patch.dict(
                        os.environ,
                        {**env, "IFAUTH_MAX_STATE_AGE_SECONDS": value},
                        clear=True,
                    ),
                    self.assertRaises(LicenseError),
                ):
                    AuthConfig.load()


class StateTests(unittest.TestCase):
    def test_initial_deny_recovery_and_expiry(self):
        now = [0.0]
        state = StateStore(config(), lambda: now[0])
        with self.assertRaises(LicenseError):
            state.snapshot().check(now[0])
        state.update("VALID", 0, initial=True)
        state.snapshot().check(now[0])
        state.update("DENIED", 12)
        with self.assertRaisesRegex(LicenseError, "LICENSE_DENIED"):
            state.snapshot().check(now[0])
        state.update("VALID", 0)
        self.assertEqual(state.snapshot().state, "RECOVERING")
        state.update("VALID", 0)
        state.snapshot().check(now[0])
        now[0] = 1
        with self.assertRaises(LicenseError):
            state.snapshot().check(now[0])
        state.update("VALID", 0)
        self.assertEqual(state.snapshot().state, "RECOVERING")
        state.update("VALID", 0)
        state.snapshot().check(now[0])

    def test_epoch_only_changes_validity(self):
        state = StateStore(config())
        state.update("VALID", 0, initial=True)
        epoch = state.snapshot().epoch
        for _ in range(5):
            state.update("VALID", 0)
        self.assertEqual(state.snapshot().epoch, epoch)
        state.update("ERROR")
        self.assertGreater(state.snapshot().epoch, epoch)

    def test_remote_replay_does_not_refresh(self):
        now = [10.0]
        cache = RemoteCache(lambda: now[0])
        message = dict(
            boot_id="boot",
            sequence=1,
            epoch=1,
            state="VALID",
            sdk_code=0,
            observation=1,
            remaining=0.2,
        )
        cache.accept(message, 10, 0.2)
        cache.snapshot()
        now[0] = 10.1
        cache.accept(message, 10.1, 0.2)
        now[0] = 10.21
        with self.assertRaises(LicenseError):
            cache.snapshot()

    def test_boot_swap_and_rollback_rejected(self):
        cache = RemoteCache(lambda: 0)
        data = dict(
            boot_id="a",
            sequence=2,
            epoch=1,
            state="VALID",
            sdk_code=0,
            observation=2,
            remaining=1,
        )
        cache.accept(data, 0, 2)
        with self.assertRaises(LicenseError):
            cache.accept({**data, "sequence": 1}, 0, 2)
        with self.assertRaises(LicenseError):
            cache.accept({**data, "boot_id": "b"}, 0, 2)
        with self.assertRaises(LicenseError):
            cache.snapshot()


class ManagerTests(unittest.TestCase):
    def test_init_failure_never_calls_status(self):
        calls = []

        class Fake:
            def __init__(self, c):
                pass

            def call(self, op):
                calls.append(op)
                return 7

            def close(self):
                pass

        manager = LicenseManager(config(), Fake)
        try:
            with self.assertRaises(LicenseError):
                manager.wait_initial()
            self.assertEqual(calls, ["init"])
        finally:
            manager.close()

    def test_first_status_failure_blocks_start(self):
        calls = []

        class Fake:
            def __init__(self, c):
                pass

            def call(self, op):
                calls.append(op)
                return 0 if op == "init" else 9

            def close(self):
                pass

        manager = LicenseManager(config(), Fake)
        try:
            with self.assertRaises(LicenseError):
                manager.wait_initial()
            self.assertEqual(calls, ["init", "status"])
        finally:
            manager.close()

    def test_requests_do_not_invoke_sdk(self):
        calls = []

        class Fake:
            def __init__(self, c):
                pass

            def call(self, op):
                calls.append(op)
                return 0

            def close(self):
                pass

        manager = LicenseManager(config(interval=1), Fake)
        try:
            manager.wait_initial()
            for _ in range(10000):
                manager.store.snapshot().check()
            self.assertEqual(calls, ["init", "status"])
        finally:
            manager.close()

    def test_actual_subprocess_timeout_and_reaping(self):
        child = SDKProcess.__new__(SDKProcess)
        child.config = config()
        child.conn, peer = socket.socketpair()
        child.conn.settimeout(0.1)
        child.process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            pass_fds=(peer.fileno(),),
        )
        peer.close()
        started = time.monotonic()
        try:
            with self.assertRaises((TimeoutError, LicenseError, OSError)):
                child.call("status")
        finally:
            child.close()
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(child.process.poll())


class TransportTests(unittest.TestCase):
    def test_bad_hmac(self):
        a, b = socket.socketpair()
        try:
            a.sendall(pack({"ok": True}, b"a" * 32))
            with self.assertRaises(LicenseError):
                receive(b, b"b" * 32)
        finally:
            a.close()
            b.close()

    def test_two_nodes_join_and_disconnect_fails_closed(self):
        port_socket = socket.socket()
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
        port_socket.close()
        cfg = config(port=port, max_age=5, startup_timeout=2)
        state = StateStore(cfg)
        state.update("VALID", 0, initial=True)
        server = ControlServer(cfg, state, nodes=2, world_size=2)
        clients = [StateClient(cfg), StateClient(cfg)]
        errors = []

        def join(i):
            try:
                clients[i].join(i, 2, 2)
            except Exception as e:
                errors.append(type(e).__name__)

        try:
            for client in clients:
                client.wait_initial()
            threads = [threading.Thread(target=join, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(3)
            self.assertFalse(errors)
            self.assertFalse(any(t.is_alive() for t in threads))
            server.close()
            deadline = time.monotonic() + 3
            for client in clients:
                while True:
                    try:
                        client.snapshot()
                    except LicenseError:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("Client kept a valid snapshot after disconnect")
                    time.sleep(0.05)
        finally:
            for client in clients:
                client.close()
            server.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
