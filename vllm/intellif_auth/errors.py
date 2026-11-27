# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class LicenseError(RuntimeError):
    """Safe public error; never include SDK messages or service endpoints."""

    def __init__(self, code="LICENSE_UNAVAILABLE"):
        self.code = (
            "LICENSE_DENIED" if code == "LICENSE_DENIED" else "LICENSE_UNAVAILABLE"
        )
        self.http_status = 403 if self.code == "LICENSE_DENIED" else 503
        super().__init__(self.code + ": model license validation failed")


class LicenseConfigurationError(LicenseError):
    pass
