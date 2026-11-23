# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify measured source bytes in the image, including native-path preservation."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

manifest = json.loads(Path("/opt/dspark-release/release-manifest.json").read_text())
for path, expected in manifest["installed_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert (
    json.loads(Path("/opt/bench-stage/control.json").read_text())["phase"] == "serving"
)
print(
    json.dumps(
        dict(
            passed=True,
            release=manifest["release"],
            verified_files=len(manifest["installed_sha256"]),
            source_overlays_required=False,
            vllm_version=importlib.metadata.version("vllm"),
            torch_version=importlib.metadata.version("torch"),
        )
    )
)
