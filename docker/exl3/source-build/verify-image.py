"""Check the pinned EXL3 reference stack without requiring a loaded model."""

import argparse
import importlib.metadata
import importlib.util
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    expected = {
        "torch": "2.13.0+cu130",
        "exllamav3": "1.6.0",
        "vllm-exl3-moe": "0.1.0.dev0",
        "flashinfer-python": "0.6.18",
        "flashinfer-cubin": "0.6.18",
        "flashinfer-jit-cache": "0.6.18+cu130",
    }
    installed = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        if installed[name] != version:
            raise RuntimeError(f"{name}: expected {version}, got {installed[name]}")
    for name in (
        "lmcache",
        "nixl",
        "nixl-cu12",
        "nixl-cu13",
        "mooncake-transfer-engine",
        "mooncake-transfer-engine-cuda13",
    ):
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"unexpected KV connector: {name}={value}")
    for name in ("lmcache", "nixl", "mooncake"):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"unexpected module: {name}")

    import torch

    assert torch.version.cuda == "13.0"

    import exllamav3_ext
    import flashinfer
    from vllm.model_executor.layers.quantization.exl3.config import Exl3Config

    assert Exl3Config.get_name() == "exl3"
    assert hasattr(exllamav3_ext, "BC_LinearEXL3")
    assert flashinfer.__version__ == "0.6.18"
    if args.gpu:
        assert torch.cuda.is_available()
        assert torch.cuda.get_device_capability() == (12, 1)
        installed["gpu"] = torch.cuda.get_device_name()
    print(json.dumps(installed, indent=2))


if __name__ == "__main__":
    main()
