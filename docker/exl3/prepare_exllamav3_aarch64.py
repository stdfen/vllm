# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare pinned ExLlamaV3 sources for a CUDA-only aarch64 build."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_EXCLUDED = (
    "avx2_target.cpp",
    "avx512_target.cpp",
    "cpu/moe_mul1.cpp",
    "parallel/all_reduce_cpu_avx2.cpp",
    "parallel/all_reduce_cpu_avx512.cpp",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--stub", required=True, type=Path)
    args = parser.parse_args()
    extension = args.source / "exllamav3" / "exllamav3_ext"
    setup = args.source / "setup.py"
    if not setup.is_file() or not extension.is_dir():
        raise SystemExit(f"not an ExLlamaV3 source tree: {args.source}")

    stub_target = extension / "aarch64_cuda_only.cpp"
    shutil.copyfile(args.stub, stub_target)
    text = setup.read_text(encoding="utf-8")
    anchor = "print (sources)\n"
    if anchor not in text:
        raise SystemExit("unsupported ExLlamaV3 setup.py: source anchor not found")
    excluded = repr(_EXCLUDED)
    injected = f"""# vLLM EXL3: aarch64 uses CUDA collectives and GPU MoE only.
if os.uname().machine in (\"aarch64\", \"arm64\"):
    excluded = {excluded}
    sources = [
        source for source in sources
        if not source.replace(\"\\\\\", \"/\").endswith(excluded)
    ]

"""
    if "# vLLM EXL3: aarch64 uses CUDA collectives" not in text:
        setup.write_text(text.replace(anchor, injected + anchor), encoding="utf-8")

    for relative in ("parallel/all_reduce_cpu.cu", "cpu/moe_handoff.cu"):
        path = extension / relative
        source = path.read_text(encoding="utf-8")
        if "__builtin_ia32_pause();" in source:
            path.write_text(
                source.replace("__builtin_ia32_pause();", 'asm volatile("yield");'),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
