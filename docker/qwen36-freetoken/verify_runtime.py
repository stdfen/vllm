# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the audited source overlay and verify the CPU-side image ABI."""

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def install_overlay() -> None:
    root = Path(__file__).resolve().parent
    spec = importlib.util.find_spec("vllm")
    assert spec and spec.origin
    packages = Path(spec.origin).parent.parent
    manifest = json.loads(
        Path("/usr/share/qwen36-freetoken/source-manifest.json").read_text()
    )
    for path, digest in manifest["overlay"].items():
        source = root / path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, path
        dest = packages / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copytree(root / "freetoken", packages / "freetoken")


def main() -> None:
    install_overlay()

    import torch
    import vllm._C_stable_libtorch  # noqa: F401
    import vllm._moe_C_stable_libtorch  # noqa: F401
    from freetoken.kernel import _pinned_tensor  # noqa: F401
    from freetoken.kernel.batch_memcpy import _jit_batch_memcpy_module
    from freetoken.kernel.fast_index_copy import _jit_fast_index_copy_multi_module
    from freetoken.layers.moe import OffloadMoELayer  # noqa: F401
    from freetoken.moe.offload_cache import OffloadMoeCache  # noqa: F401
    from tvm_ffi.cpp import load_inline

    from vllm.model_executor.layers.fused_moe.qwen36_freetoken import (
        Qwen36FreeTokenRuntime,  # noqa: F401
    )

    assert importlib.metadata.version("vllm").split("+")[0] == "0.33.0"
    assert torch.__version__.split("+")[0] == "2.14.0"
    assert torch.version.cuda and torch.version.cuda.startswith("13.0")
    assert importlib.metadata.version("apache-tvm-ffi") == "0.1.11"
    module = load_inline(
        "qwen36_freetoken_ffi_smoke",
        cpp_sources=[
            "#include <tvm/ffi/function.h>",
            "int answer() { return 42; }",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(answer, answer);",
        ],
    )
    assert module.answer() == 42
    assert callable(_jit_batch_memcpy_module().batch_memcpy)
    assert callable(
        _jit_fast_index_copy_multi_module(num_threads=1024, blocks_per_bank=8).launch
    )
    for package in ("vllm", "torch", "triton", "apache-tvm-ffi", "flashlib"):
        print(f"{package}=={importlib.metadata.version(package)}")
    print("FreeToken runtime imports, PyTorch extension ABI and TVM FFI JIT passed")


def verify_tensorfold() -> None:
    from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2  # noqa: F401

    from vllm.v1.worker.gpu.spec_decode.dflash2 import (  # noqa: F401
        tensorfold_attention,
        tensorfold_commit,
        tensorfold_round,
    )

    assert importlib.metadata.version("tensorfold") == "0.6.0"
    print("TensorFold 0.6.0 Tree-DFlash2 and vLLM integration imports passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--tensorfold"]:
        verify_tensorfold()
    else:
        main()
