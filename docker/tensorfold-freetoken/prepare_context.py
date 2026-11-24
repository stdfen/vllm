# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble the vLLM source delta, FreeToken patch and locked build inputs."""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

VLLM_BASE = "4ba787ec9959e566d3b587a796dff07673324da5"
FREETOKEN_REV = "af71ba43206e124f5ff6419b47ee36c6e9981078"
FREETOKEN_PATCH = "vllm-0.33.0-qwen36-freetoken.patch"
FLASHLIB_WHEEL = "flashlib-0.3.0-py3-none-any.whl"
FLASHLIB_SHA256 = "9037911f315fef3c9f4453261597363abcb83868813ad1b258ab1308b34fd11a"
FLASHLIB_URL = (
    "https://files.pythonhosted.org/packages/e5/b8/"
    f"4c085892462e521bb9f2d943ff34fa219e3a5a206798f3c96a983538039f/{FLASHLIB_WHEEL}"
)
TENSORFOLD_WHEEL = "tensorfold-0.6.0-py3-none-any.whl"
TENSORFOLD_SHA256 = "68f8c24002d3efbb1345462e7b7a3df2a462a2d0e06f5ca54ad0ff330ca3cbef"
TENSORFOLD_ENTRY = "vllm/v1/worker/gpu/spec_decode/dflash2/tensorfold_round.py"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("freetoken", type=Path, help="FreeToken Git checkout")
    parser.add_argument("output", type=Path, help="New build context directory")
    parser.add_argument(
        "--flashlib-wheel", type=Path, help="Use a locally cached wheel"
    )
    parser.add_argument(
        "--tensorfold-wheel", type=Path, help="Pinned wheel for the combined image"
    )
    args = parser.parse_args()

    recipe = Path(__file__).resolve().parent
    repo = recipe.parents[1]
    git(repo, "merge-base", "--is-ancestor", VLLM_BASE, "HEAD")
    changes = git(repo, "diff", "--name-only", "-z", VLLM_BASE, "--", "vllm")
    new_files = git(
        repo, "ls-files", "--others", "--exclude-standard", "-z", "--", "vllm"
    )
    paths = sorted(set((changes + new_files).decode().strip("\0").split("\0")))
    if not paths or any(
        not p.endswith(".py") or not (repo / p).is_file() for p in paths
    ):
        raise ValueError(
            "The release image overlay requires a Python-only source delta"
        )
    if args.tensorfold_wheel and TENSORFOLD_ENTRY not in paths:
        raise ValueError("The combined image requires the vLLM TensorFold source delta")

    patch = recipe / "patches" / FREETOKEN_PATCH
    patch_stats = git(repo, "apply", "--numstat", "-z", str(patch)).decode()
    patch_paths = [
        entry.split("\t", 2)[2] for entry in patch_stats.rstrip("\0").split("\0")
    ]
    if not set(patch_paths).issubset(paths):
        raise ValueError(
            "Apply the FreeToken source patch before preparing the context"
        )

    output = args.output.resolve()
    output.mkdir(parents=True)
    shutil.copyfile(patch, output / FREETOKEN_PATCH)
    manifest = {
        "vllm_base": VLLM_BASE,
        "vllm_head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "freetoken_revision": FREETOKEN_REV,
        "freetoken_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "freetoken_patch_paths": patch_paths,
        "overlay": {},
    }
    for path in paths:
        dest = output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        source = (repo / path).read_bytes()
        if path in patch_paths:
            source = source.replace(b"\r\n", b"\n")
        dest.write_bytes(source)
        manifest["overlay"][path] = hashlib.sha256(source).hexdigest()

    archive = git(
        args.freetoken,
        "archive",
        FREETOKEN_REV,
        "python/freetoken",
        "setup.py",
        "pyproject.toml",
        "README.md",
        "LICENSE",
    )
    manifest["freetoken_source_sha256"] = hashlib.sha256(archive).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(output / "freetoken", filter="data")

    if args.flashlib_wheel:
        wheel = args.flashlib_wheel.read_bytes()
    else:
        with urlopen(FLASHLIB_URL, timeout=30) as response:
            wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != FLASHLIB_SHA256:
        raise ValueError("Flashlib wheel SHA-256 mismatch")
    (output / FLASHLIB_WHEEL).write_bytes(wheel)
    manifest["flashlib_wheel_sha256"] = FLASHLIB_SHA256

    if args.tensorfold_wheel:
        wheel = args.tensorfold_wheel.read_bytes()
        if hashlib.sha256(wheel).hexdigest() != TENSORFOLD_SHA256:
            raise ValueError("TensorFold wheel SHA-256 mismatch")
        (output / TENSORFOLD_WHEEL).write_bytes(wheel)
        manifest["tensorfold_wheel_sha256"] = TENSORFOLD_SHA256

    (output / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename in ("Dockerfile", "verify_runtime.py", "verify_decode_routes.py"):
        shutil.copyfile(recipe / filename, output / filename)


if __name__ == "__main__":
    main()
