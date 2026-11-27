# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded authenticated JSON control channel; never pickle network input.

The deployment key is mounted at runtime. HMAC authenticates status, not the
license issuer, and is not intended to defeat an administrator modifying code.
"""

import atexit
import hashlib
import hmac
import json
import secrets
import socket
import threading
import time

from .errors import LicenseError
from .state import RemoteCache


def pack(payload, key):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return (
        json.dumps(
            {
                "body": body.decode(),
                "mac": hmac.new(key, body, hashlib.sha256).hexdigest(),
            }
        ).encode()
        + b"\n"
    )


def receive(conn, key):
    data = bytearray()
    deadline = time.monotonic() + 2
    while not data.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LicenseError()
        conn.settimeout(remaining)
        chunk = conn.recv(4096)
        if not chunk or len(data) + len(chunk) > 16384:
            raise LicenseError()
        data.extend(chunk)
    message = json.loads(data)
    body = message["body"].encode()
    if not hmac.compare_digest(
        message["mac"], hmac.new(key, body, hashlib.sha256).hexdigest()
    ):
        raise LicenseError()
    return json.loads(body)


class ControlServer:
    def __init__(self, config, store, nodes=1, world_size=1):
        self.config, self.store = config, store
        self.nodes, self.world_size = nodes, world_size
        self.members = {}
        self.member_lock = threading.Lock()
        self.stop = threading.Event()
        self.slots = threading.BoundedSemaphore(16)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((config.host, config.port))
            self.sock.listen(32)
            self.sock.settimeout(0.2)
        except OSError:
            self.sock.close()
            raise LicenseError() from None
        self.thread = threading.Thread(
            target=self._serve, name="ifauth-control", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def _serve(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self.slots.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            with conn:
                request = receive(conn, self.config.key)
                if (
                    request.get("deployment") != self.config.deployment_id
                    or request.get("policy") != self.config.policy_hash
                ):
                    return
                nonce = request.get("nonce")
                if (
                    not isinstance(nonce, str)
                    or len(nonce) != 32
                    or request.get("op") not in ("snapshot", "join")
                ):
                    return
                if request["op"] == "join":
                    rank, member = request.get("rank"), request.get("member")
                    if (
                        type(rank) is not int
                        or not 0 <= rank < self.nodes
                        or request.get("nodes") != self.nodes
                        or request.get("world_size") != self.world_size
                        or not isinstance(member, str)
                        or len(member) != 32
                    ):
                        return
                    with self.member_lock:
                        if rank in self.members and self.members[rank] != member:
                            return
                        self.members[rank] = member
                response = {
                    "deployment": self.config.deployment_id,
                    "policy": self.config.policy_hash,
                    "nonce": nonce,
                    "snapshot": self.store.wire(),
                    "joined": len(self.members) == self.nodes,
                }
                conn.sendall(pack(response, self.config.key))
        except (OSError, ValueError, KeyError, TypeError, LicenseError):
            pass
        finally:
            self.slots.release()

    def close(self):
        self.stop.set()
        self.sock.close()
        if self.thread is not threading.current_thread():
            self.thread.join(3)


class StateClient:
    def __init__(self, config):
        self.config = config
        self.cache = RemoteCache()
        self.stop = threading.Event()
        self.observed = threading.Event()
        self.thread = threading.Thread(
            target=self._poll, name="ifauth-status-cache", daemon=True
        )
        self.thread.start()
        atexit.register(self.close)

    def _poll(self):
        while not self.stop.is_set():
            nonce = secrets.token_hex(16)
            try:
                sent = time.monotonic()
                with socket.create_connection(
                    (self.config.host, self.config.port), timeout=2
                ) as conn:
                    conn.sendall(
                        pack(
                            {
                                "op": "snapshot",
                                "deployment": self.config.deployment_id,
                                "policy": self.config.policy_hash,
                                "nonce": nonce,
                            },
                            self.config.key,
                        )
                    )
                    response = receive(conn, self.config.key)
                if (
                    response.get("nonce") != nonce
                    or response.get("policy") != self.config.policy_hash
                    or response.get("deployment") != self.config.deployment_id
                ):
                    raise LicenseError()
                self.cache.accept(response["snapshot"], sent, self.config.max_age)
                self.observed.set()
            except (OSError, ValueError, KeyError, TypeError, LicenseError):
                self.cache.fail()
            self.stop.wait(0.25)

    def wait_initial(self):
        end = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < end and not self.stop.is_set():
            try:
                return self.cache.snapshot()
            except LicenseError as error:
                if error.code == "LICENSE_DENIED":
                    raise
            self.stop.wait(0.05)
        raise LicenseError()

    def snapshot(self):
        return self.cache.snapshot()

    def join(self, rank, nodes, world_size):
        member = secrets.token_hex(16)
        end = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < end:
            nonce = secrets.token_hex(16)
            try:
                sent = time.monotonic()
                with socket.create_connection(
                    (self.config.host, self.config.port), timeout=2
                ) as conn:
                    conn.sendall(
                        pack(
                            {
                                "op": "join",
                                "deployment": self.config.deployment_id,
                                "policy": self.config.policy_hash,
                                "nonce": nonce,
                                "member": member,
                                "rank": rank,
                                "nodes": nodes,
                                "world_size": world_size,
                            },
                            self.config.key,
                        )
                    )
                    response = receive(conn, self.config.key)
                if (
                    response.get("nonce") != nonce
                    or response.get("policy") != self.config.policy_hash
                    or response.get("deployment") != self.config.deployment_id
                ):
                    raise LicenseError()
                self.cache.accept(response["snapshot"], sent, self.config.max_age)
                self.cache.snapshot()
                if response.get("joined") is True:
                    return
            except (OSError, ValueError, KeyError, TypeError, LicenseError):
                pass
            self.stop.wait(0.1)
        raise LicenseError()

    def close(self):
        self.stop.set()
        if self.thread is not threading.current_thread():
            self.thread.join(5)
