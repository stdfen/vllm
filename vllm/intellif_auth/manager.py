# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time

from .config import PRODUCT
from .errors import LicenseError
from .state import StateStore


class SDKProcess:
    def __init__(self, config):
        self.config = config
        self.conn, child = socket.socketpair()
        self.conn.settimeout(config.call_timeout)
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(pathlib.Path(__file__).with_name("sdk_host.py")),
                    str(child.fileno()),
                ],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except BaseException:
            self.conn.close()
            raise
        finally:
            child.close()

    def call(self, op):
        request = {"op": op}
        if op == "init":
            request.update(addrs=self.config.addrs, product=PRODUCT)
        self.conn.sendall(json.dumps(request).encode() + b"\n")
        data = bytearray()
        deadline = time.monotonic() + self.config.call_timeout
        while not data.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LicenseError()
            self.conn.settimeout(remaining)
            piece = self.conn.recv(4096)
            if not piece or len(data) + len(piece) > 16384:
                raise LicenseError()
            data.extend(piece)
        reply = json.loads(data)
        if "error" in reply or type(reply.get("code")) is not int:
            raise LicenseError()
        return reply["code"]

    def close(self):
        self.conn.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


class LicenseManager:
    def __init__(self, config, process_factory=SDKProcess):
        self.config = config
        self.store = StateStore(config)
        self._factory = (
            process_factory  # Python-only unit-test injection, not runtime config.
        )
        self.stop = threading.Event()
        self.initialized = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="ifauth-sdk-watchdog", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def wait_initial(self):
        if not self.initialized.wait(self.config.startup_timeout):
            self.close()
            raise LicenseError()
        return self.store.snapshot().check()

    def _run(self):
        first, restarts = True, 0
        while not self.stop.is_set():
            child = None
            try:
                child = self._factory(self.config)
                code = child.call("init")
                if code != 0:
                    self.store.update("DENIED", code)
                    self.initialized.set()
                    return  # Initial denial never becomes an infinite Init loop.
                code = child.call("status")
                self.store.update(
                    "VALID" if code == 0 else "DENIED", code, initial=first
                )
                self.initialized.set()
                if first and code != 0:
                    return
                first = False
                while not self.stop.wait(self.config.interval):
                    code = child.call("status")
                    self.store.update("VALID" if code == 0 else "DENIED", code)
            except Exception:
                self.store.update("ERROR")
                self.initialized.set()
                if first:
                    return
            finally:
                if child is not None:
                    child.close()
            restarts += 1
            if restarts > 3 or self.stop.wait(self.config.interval):
                break

    def close(self):
        self.stop.set()
        self.store.update("STOPPING")
        if self.thread is not threading.current_thread():
            self.thread.join(self.config.call_timeout + 4)
