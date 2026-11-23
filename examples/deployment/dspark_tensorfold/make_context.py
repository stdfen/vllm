# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build an inspectable image context from this checkout and release helpers."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    package = "/usr/local/lib/python3.12/dist-packages/vllm/"
    manifest = json.loads((here / "release-reference.json").read_text())
    # These three source fingerprints describe the required frozen base image.
    installed = {
        path: digest
        for path, digest in manifest["installed_sha256"].items()
        if path.endswith(
            (
                "/config/speculative.py",
                "/spec_decode/__init__.py",
                "/dspark/speculator.py",
            )
        )
    }
    sources = []
    for name in (
        "__init__.py",
        "core.py",
        "head.py",
        "sampling.py",
        "speculator.py",
        "verify.py",
    ):
        relative = "v1/worker/gpu/spec_decode/tensorfold_dsv41/" + name
        sources.append(
            (repo / "vllm" / relative, "tensorfold_dsv41/" + name, package + relative)
        )
    sources.extend(
        [
            (
                repo / "vllm/v1/worker/gpu/model_runner.py",
                "model_runner.py",
                package + "v1/worker/gpu/model_runner.py",
            ),
            (here / "jit_guard.py", "jit_guard.py", package + "utils/jit_monitor.py"),
            (
                here / "warm_kernels.py",
                "warm_kernels.py",
                "/opt/bench-helpers/warm_kernels.py",
            ),
            (
                here / "markov_top_tokens.json",
                "markov_top_tokens.json",
                "/opt/ab/assets/markov_top_tokens.json",
            ),
            (
                here / "check_autotune_guard.py",
                "check_autotune_guard.py",
                "/opt/check_autotune_guard.py",
            ),
        ]
    )
    for source, name, target in sources:
        destination = output / "payload" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        installed[target] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest["installed_sha256"] = installed
    manifest["candidate_sha256"] = {
        name: installed[
            package
            + (
                "v1/worker/gpu/model_runner.py"
                if name == "model_runner.py"
                else "v1/worker/gpu/spec_decode/tensorfold_dsv41/" + name
            )
        ]
        for name in ("core.py", "head.py", "speculator.py", "model_runner.py")
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    manifest.update(
        git_head=revision,
        release="dspark-graph24-source-build",
        image_tag="local/vllm-dsv41:graph24-source-build",
    )
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("control.json", "verify_image.py"):
        shutil.copy2(here / name, output / name)
    dockerfile = (here / "Dockerfile").read_text()
    dockerfile = dockerfile.replace(
        'image.version="dspark-tensorfold-graph24-v1-20261123"',
        'image.version="dspark-graph24-source-build"',
    ).replace("574524ce02affc442738cb4c83aa765d45489950", revision)
    (output / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    print(f"Image context ready: {output}")


if __name__ == "__main__":
    main()
