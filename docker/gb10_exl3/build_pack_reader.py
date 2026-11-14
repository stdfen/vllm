# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the P0 reader, retaining A's exact EXL3 native binary."""

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
os.environ["MAX_JOBS"] = "4"
package = Path(importlib.util.find_spec("vllm").origin).parent
source = package / "models/deepseek_v41/common/engram_nvme.cpp"
module = load(
    name="vllm_engram_pack_v033",
    sources=[str(source)],
    extra_cflags=["-O2"],
    with_cuda=True,
    verbose=True,
)
target = package.parent / Path(module.__file__).name
shutil.copyfile(module.__file__, target)
receipt = dict(
    torch=torch.__version__,
    cuda=torch.version.cuda,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
)
Path("/opt/vllm-033-compat/p0-native-build.json").write_text(
    json.dumps(receipt, indent=2)
)
print(json.dumps(receipt), flush=True)
