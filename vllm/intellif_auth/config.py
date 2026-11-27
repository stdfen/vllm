# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re

from .errors import LicenseConfigurationError

PRODUCT = "ifaas-vllm"


@dataclasses.dataclass(frozen=True, repr=False)
class AuthConfig:
    addrs: str
    deployment_id: str
    host: str
    port: int
    key: bytes
    interval: float = 10.0
    call_timeout: float = 5.0
    max_age: float = 30.0
    startup_timeout: float = 60.0
    recovery_successes: int = 2

    @classmethod
    def load(cls):
        try:
            values = {}
            if path := os.environ.get("IFAUTH_CONFIG_FILE"):
                values = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
                if not isinstance(values, dict):
                    raise ValueError()
            values.update(
                {k: v for k, v in os.environ.items() if k.startswith("IFAUTH_")}
            )

            def value(name, default=None):
                return values.get("IFAUTH_" + name, default)

            addrs = value("ADDRS")
            deployment = value("DEPLOYMENT_ID")
            if not isinstance(addrs, str) or not addrs.strip() or len(addrs) > 4096:
                raise ValueError()
            if not isinstance(deployment, str) or not re.fullmatch(
                r"[a-zA-Z0-9_-]{8,128}", deployment
            ):
                raise ValueError()
            # Host and port are separate to avoid ambiguous IPv6 parsing.
            host = value("COORDINATOR_HOST", "127.0.0.1")
            port = int(value("COORDINATOR_PORT", 29795))
            if not isinstance(host, str) or not host or not 1024 <= port <= 65535:
                raise ValueError()
            key = pathlib.Path(value("CONTROL_KEY_FILE")).read_bytes()
            if not 32 <= len(key) <= 4096:
                raise ValueError()

            def number(name, default, low, high):
                x = float(value(name, default))
                if not math.isfinite(x) or not low <= x <= high:
                    raise ValueError()
                return x

            interval = number("CHECK_INTERVAL_SECONDS", 10, 1, 30)
            timeout = number("CALL_TIMEOUT_SECONDS", 5, 0.5, 15)
            age = number("MAX_STATE_AGE_SECONDS", 30, 2, 60)
            startup = number("STARTUP_TIMEOUT_SECONDS", 60, 5, 120)
            recovery = number("RECOVERY_SUCCESSES", 2, 2, 5)
            if not recovery.is_integer() or age < interval + timeout:
                raise ValueError()
            return cls(
                addrs,
                deployment,
                host,
                port,
                key,
                interval,
                timeout,
                age,
                startup,
                int(recovery),
            )
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            raise LicenseConfigurationError() from None

    @property
    def policy_hash(self):
        # Key and addresses never leave the process in plaintext.
        data = dataclasses.asdict(self)
        data["key"] = hashlib.sha256(self.key).hexdigest()
        data["product"] = PRODUCT
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
