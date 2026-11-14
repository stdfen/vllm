# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rebuild pinned EXL3 sources against the official 0.33 image's PyTorch ABI."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ARCHIVE_SHA = "73cdc89f493d3c28f66a8cc05ec50e8678817c4615314446ac12"
ARCHIVE_SHA += "95c9f0908ba4"
REVISION = "151539c77abc7ab7425d30da7a4e8e3c5c154e7b"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    archive = Path(sys.argv[1])
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("EXL3 source archive SHA256 differs from the pinned source")
    root = Path("/opt/vllm-033-compat/build")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        source.extractall(root, filter="data")
    source = root / ("exllamav3-" + REVISION)
    extension = source / "exllamav3/exllamav3_ext"
    shutil.copyfile(
        Path(__file__).with_name("aarch64_cuda_only.cpp"),
        extension / "aarch64_cuda_only.cpp",
    )
    setup = source / "setup.py"
    text = setup.read_text()
    anchor = "\nif hip:\n"
    if text.count(anchor) != 1:
        raise ValueError("Pinned EXL3 setup source contract changed")
    injected = """
# Common GB10 compatibility: GPU-only EXL3, with explicit unsupported CPU stubs.
excluded = ("avx2_target.cpp", "avx512_target.cpp", "cpu/moe_mul1.cpp",
            "parallel/all_reduce_cpu_avx2.cpp", "parallel/all_reduce_cpu_avx512.cpp")
sources = [s for s in sources if not s.replace("\\\\", "/").endswith(excluded)]
"""
    setup.write_text(text.replace(anchor, "\n" + injected + anchor))
    for relative in ("parallel/all_reduce_cpu.cu", "cpu/moe_handoff.cu"):
        path = extension / relative
        text = path.read_text()
        path.write_text(
            text.replace("__builtin_ia32_pause();", 'asm volatile("yield");')
        )
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="12.1a", MAX_JOBS="4")
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=source,
        env=env,
        check=True,
    )
    binaries = list(source.glob("exllamav3_ext*.so"))
    if len(binaries) != 1:
        raise ValueError("Expected exactly one rebuilt EXL3 extension")
    package_dir = Path(importlib.util.find_spec("vllm").origin).parent
    target = package_dir.parent / binaries[0].name
    shutil.copyfile(binaries[0], target)
    shutil.copytree(
        source / "exllamav3", package_dir.parent / "exllamav3", dirs_exist_ok=True
    )
    shutil.copyfile(source / "LICENSE", root.parent / "EXLLAMAV3-LICENSE")
    # Build the common Engram callback against the same headers and ABI.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
    os.environ["MAX_JOBS"] = "4"
    engram_source = package_dir / "models/deepseek_v41/common/engram_disk.cpp"
    reader = load(
        name="vllm_engram_disk_v033",
        sources=[str(engram_source)],
        extra_cflags=["-O2"],
        with_cuda=True,
        verbose=True,
    )
    reader_target = package_dir.parent / Path(reader.__file__).name
    shutil.copyfile(reader.__file__, reader_target)
    receipt = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        exllamav3_revision=REVISION,
        source_archive_sha256=ARCHIVE_SHA,
        architecture="12.1a",
        exl3_binary_sha256=sha(target),
        engram_source_sha256=sha(engram_source),
        engram_binary_sha256=sha(reader_target),
    )
    Path("/opt/vllm-033-compat/native-build.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
