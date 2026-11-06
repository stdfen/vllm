"""Emit package/tool versions without container environment or credentials."""

import importlib.metadata
import json
import platform
import subprocess
import sys

packages = {
    distribution.metadata["Name"]: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata["Name"]
}
system_packages = subprocess.check_output(
    ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
)
print(
    json.dumps(
        {
            "architecture": platform.machine(),
            "python": sys.version,
            "pythonPackages": dict(sorted(packages.items())),
            "systemPackages": system_packages.splitlines(),
            "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        },
        indent=2,
    )
)
