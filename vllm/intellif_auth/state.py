# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import logging
import threading
import time
import uuid

from .errors import LicenseError


@dataclasses.dataclass(frozen=True)
class Snapshot:
    boot_id: str
    sequence: int = 0
    epoch: int = 0
    state: str = "INITIALIZING"
    sdk_code: int | None = None
    observation: int = 0
    deadline: float = 0.0

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        if self.state != "VALID" or now >= self.deadline:
            raise LicenseError(
                "LICENSE_DENIED" if self.state == "DENIED" else "LICENSE_UNAVAILABLE"
            )
        return self


class StateStore:
    def __init__(self, config, clock=time.monotonic):
        self.config, self.clock = config, clock
        self.lock = threading.Lock()
        self.current = Snapshot(uuid.uuid4().hex)
        self.successes = 0

    def update(self, state, code=None, initial=False):
        with self.lock:
            old = self.current
            observation, deadline = old.observation, old.deadline
            usable = old.state == "VALID" and self.clock() < old.deadline
            if old.state == "VALID" and not usable:
                self.successes = 0
            if state == "VALID":
                self.successes += 1
                observation += 1
                deadline = self.clock() + self.config.max_age
                if (
                    not initial
                    and not usable
                    and self.successes < self.config.recovery_successes
                ):
                    state = "RECOVERING"
            else:
                self.successes = 0
                deadline = 0
            changed = (old.state == "VALID") != (state == "VALID")
            self.current = Snapshot(
                old.boot_id,
                old.sequence + 1,
                old.epoch + int(changed),
                state,
                code,
                observation,
                deadline,
            )
            if old.state != state:
                logging.getLogger("vllm.intellif_auth").info(
                    "IFAUTH_STATE state=%s sdk_code=%s epoch=%s",
                    state,
                    code,
                    self.current.epoch,
                )

    def snapshot(self):
        with self.lock:
            return self.current

    def wire(self):
        s = self.snapshot()
        out = dataclasses.asdict(s)
        out["remaining"] = max(
            0.0, min(self.config.max_age, out.pop("deadline") - self.clock())
        )
        return out


class RemoteCache:
    """Bound observations to local send time; replay never extends deadlines."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.current = Snapshot("")
        self.seen_deadline = {}
        self.failed = True

    def accept(self, wire, sent_at, max_age):
        with self.lock:
            old = self.current
            # An owner restart requires a new engine deployment, not a live swap.
            if old.boot_id and wire["boot_id"] != old.boot_id:
                self.failed = True
                raise LicenseError()
            if wire["sequence"] < old.sequence:
                raise LicenseError()
            remaining = float(wire["remaining"])
            if not 0 <= remaining <= max_age:
                raise LicenseError()
            key = (wire["boot_id"], wire["observation"])
            deadline = sent_at + remaining
            deadline = min(deadline, self.seen_deadline.get(key, deadline))
            self.seen_deadline = {key: deadline}
            fields = {
                k: wire[k]
                for k in [
                    "boot_id",
                    "sequence",
                    "epoch",
                    "state",
                    "sdk_code",
                    "observation",
                ]
            }
            self.current = Snapshot(**fields, deadline=deadline)
            self.failed = False

    def fail(self):
        with self.lock:
            self.failed = True

    def snapshot(self):
        with self.lock:
            if self.failed:
                raise LicenseError()
            return self.current.check(self.clock())
