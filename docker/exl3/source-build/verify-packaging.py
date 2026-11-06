"""Verify final filesystem changes and retained runtime JIT prerequisites."""

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubin-policy", type=pathlib.Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    reports = pathlib.Path("/usr/local/share/vllm-build")
    static = json.loads((reports / "cuda-static-prune.json").read_text())
    dedupe = json.loads((reports / "cuda-runtime-dedupe.json").read_text())
    cubins = json.loads((reports / "flashinfer-cubin-prune.json").read_text())
    if not static["applied"] or not dedupe["applied"]:
        raise RuntimeError("Missing applied CUDA policy")
    for row in static["files"]:
        path = pathlib.Path(row["path"])
        if path.exists() or path.is_symlink():
            raise RuntimeError("Static candidate was restored by a later layer")
    for row in dedupe["files"]:
        path = pathlib.Path(row["path"])
        if not path.is_symlink() or path.resolve() != pathlib.Path(row["target"]):
            raise RuntimeError("Dynamic duplicate is not the approved symlink")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != row["sha256"]:
                raise RuntimeError("Symlink target content changed")
    packages = pathlib.Path("/usr/local/lib/python3.12/dist-packages")
    policy = json.loads(args.cubin_policy.read_text())
    removed = {"flashinfer_cubin/cubins/" + row["path"] for row in policy["remove"]}
    for relative in removed:
        if (packages / relative).exists():
            raise RuntimeError("Removed cubin restored by a later layer")
    record = packages / "flashinfer_cubin-0.6.18.dist-info/RECORD"
    with record.open(newline="") as source:
        if any(row[0] in removed for row in csv.reader(source)):
            raise RuntimeError("Wheel RECORD still references a removed cubin")
    for tool in ("nvcc", "ptxas"):
        if not shutil.which(tool):
            raise RuntimeError(f"Missing JIT tool {tool}")
    cuda = pathlib.Path("/usr/local/cuda-13.0/targets/sbsa-linux")
    for relative in (
        "include/cuda.h",
        "include/nvrtc.h",
        "include/cuda_runtime.h",
        "lib/libcudart_static.a",
        "lib/libcudadevrt.a",
    ):
        if not (cuda / relative).is_file():
            raise RuntimeError(f"Missing JIT prerequisite {relative}")
    for module in ("lmcache", "nixl", "mooncake"):
        if importlib.util.find_spec(module) is not None:
            raise RuntimeError(f"Unexpected optional connector {module}")
    if list(pathlib.Path("/var/cache/apt/archives").rglob("*.deb")):
        raise RuntimeError("APT input archives persisted in final image")
    print(
        json.dumps(
            {
                "status": "passed",
                "imageId": args.image_id,
                "staticBytesRemoved": static["candidateBytes"],
                "dynamicBytesDeduplicated": dedupe["candidateBytes"],
                "cubinBytesRemoved": cubins["removedBytes"],
                "cubinCountRemoved": len(removed),
                "jitToolsAndHeadersRetained": True,
                "optionalConnectorsAbsent": True,
                "aptInputArchivesAbsent": True,
            }
        )
    )


if __name__ == "__main__":
    main()
