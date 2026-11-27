# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the A1 256K profile changes on the archived GB10 auto release."""

import json
import sys
from pathlib import Path

A1_FLAGS = (
    "tree_dflash2_tensorfold_batch_verify_costs",
    "tree_dflash2_tensorfold_batch_device_commit",
    "tree_dflash2_tensorfold_c8_graph_buckets",
    "tree_dflash2_tensorfold_batch_graph_buckets",
    "tree_dflash2_tensorfold_long_context_graph",
)
HISTORICAL_PARENT_IMAGE_ID = (
    "sha256:ab1d34c0dfb6f733f8f7ba467d39e8f58b4ae8d6a1eda4db9fe43d324ab282d0"
)


def configure(root: Path) -> None:
    profile_path = root / "profiles/gb10-auto.yaml"
    lock_path = root / "release-lock.json"
    verifier = root / "verify_image.py"
    profile = json.loads(profile_path.read_bytes())
    lock = json.loads(lock_path.read_bytes())
    config = profile["speculative-config"]
    assert config == lock["speculative_configs"]["gb10-auto"]
    assert profile["max-model-len"] == 131072 and profile["kv-cache-dtype"] == "fp8"
    assert config["num_speculative_tokens"] == 15
    assert config["tree_dflash2_tensorfold_c1_auto_isolate"]
    assert all(flag not in config for flag in A1_FLAGS)

    config.update(dict.fromkeys(A1_FLAGS, True))
    profile["max-model-len"] = 262144
    lock["speculative_configs"]["gb10-auto"] = config
    lock["gb10_auto_max_model_len"] = 262144
    lock["gb10_auto_historical_a1_flags"] = A1_FLAGS
    lock["gb10_auto_parent_image_id"] = HISTORICAL_PARENT_IMAGE_ID

    # Keep the installed verifier aligned with the archived A1 release.
    source = verifier.read_text(encoding="utf-8")
    isolate = '        "tree_dflash2_tensorfold_c1_auto_isolate": True,\n'
    spec = "        spec = args.speculative_config\n"
    assert source.count(isolate) == source.count(spec) == 1
    source = source.replace(
        isolate,
        isolate
        + '        **dict.fromkeys(lock["gb10_auto_historical_a1_flags"], True),\n',
    )
    source = source.replace(
        spec,
        spec + '        assert args.max_model_len == (lock["gb10_auto_max_model_len"]'
        ' if profile == "gb10-auto" else 131072)\n',
    )
    for path, content in ((profile_path, profile), (lock_path, lock)):
        path.write_bytes((json.dumps(content, indent=2) + "\n").encode("utf-8"))
    verifier.write_bytes(source.encode("utf-8"))


if __name__ == "__main__":
    configure(Path(sys.argv[1]))
