# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise a real Triton autotune cache miss, outside all formal timings."""

import json
import os
import pathlib
import runpy

import torch

from vllm.triton_utils import tl, triton

control = pathlib.Path("/tmp/autotune-guard-contract.json")
os.environ["VLLM_BENCH_STAGE"] = str(control)
control.write_text(json.dumps(dict(phase="warmup")))
monitor = runpy.run_path(
    "/usr/local/lib/python3.12/dist-packages/vllm/utils/jit_monitor.py"
)
monitor["activate"](mode="warn", verbose=True)


@triton.autotune(
    configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})], key=["n"]
)
@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + offsets, tl.load(src + offsets, offsets < n, 0), offsets < n)


source = torch.arange(128, device="cuda", dtype=torch.float32)
output = torch.zeros_like(source)
copy_kernel[lambda meta: (triton.cdiv(32, meta["BLOCK"]),)](source, output, 32)
torch.accelerator.synchronize()
assert torch.equal(output[:32], source[:32])
for phase, n in (("canary", 64), ("formal", 96)):
    control.write_text(json.dumps(dict(phase=phase)))
    runpy.run_path("/opt/bench-helpers/warm_kernels.py")["warm_dflash_inputs"]()
    try:
        copy_kernel[lambda meta, size=n: (triton.cdiv(size, meta["BLOCK"]),)](
            source, output, n
        )
    except RuntimeError as exc:
        assert "FORMAL_JIT_FORBIDDEN: Triton/autotuning benchmark/" in str(exc), str(
            exc
        )
    else:
        raise AssertionError("A real autotuning cache miss was not rejected")
print("REAL_TRITON_AUTOTUNE_GUARD_PASS")
