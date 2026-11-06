"""Inventory, then optionally remove verified CUDA duplicates or static libraries."""

import argparse
import hashlib
import json
import os
import pathlib

STATIC_CANDIDATES = {"libcublas_static.a", "libcublasLt_static.a", "libcurand_static.a"}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", type=pathlib.Path, default=pathlib.Path("/usr/local/cuda-13.0")
    )
    parser.add_argument(
        "--nvidia",
        type=pathlib.Path,
        default=pathlib.Path("/usr/local/lib/python3.12/dist-packages/nvidia"),
    )
    parser.add_argument("--mode", choices=("static", "dedupe"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-manifest", type=pathlib.Path)
    parser.add_argument("--approved-sha256")
    args = parser.parse_args()
    cuda = args.cuda.resolve(strict=True)
    if cuda != pathlib.Path("/usr/local/cuda-13.0"):
        raise RuntimeError("Unexpected CUDA runtime root")
    candidates = []
    if args.mode == "static":
        for path in sorted(cuda.rglob("*.a")):
            if path.name in STATIC_CANDIDATES and not path.is_symlink():
                candidates.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                        "action": "unlink",
                    }
                )
    else:
        if args.nvidia.resolve(strict=True) != pathlib.Path(
            "/usr/local/lib/python3.12/dist-packages/nvidia"
        ):
            raise RuntimeError("Unexpected NVIDIA package root")
        primary = {}
        for path in sorted(cuda.rglob("*.so*")):
            if path.is_file() and not path.is_symlink():
                primary.setdefault(path.stat().st_size, []).append(path)
        for path in sorted(args.nvidia.rglob("*.so*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(args.nvidia.resolve())
            ):
                continue
            for original in primary.get(path.stat().st_size, []):
                checksum = digest(path)
                if checksum == digest(original):
                    candidates.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": checksum,
                            "action": "symlink",
                            "target": str(original),
                        }
                    )
                    break
    if args.apply:
        if (
            not args.approved_manifest
            or digest(args.approved_manifest) != args.approved_sha256
        ):
            raise RuntimeError("A hash-pinned dependency-reviewed manifest is required")
        approved = json.loads(args.approved_manifest.read_text())
        evidence_hashes = (
            approved.get("referenceValidationSha256"),
            approved.get("kernelSmokeSha256"),
            approved.get("traceSha256"),
        )
        if (
            approved.get("referenceValidation") != "passed"
            or approved.get("mode") != args.mode
            or not str(approved.get("referenceImageId", "")).startswith("sha256:")
            or not all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in evidence_hashes
            )
            or approved.get("files") != candidates
        ):
            raise RuntimeError(
                "Candidate list differs from the reviewed dependency manifest"
            )
        for item in candidates:
            path = pathlib.Path(item["path"])
            path.unlink()
            if item["action"] == "symlink":
                path.symlink_to(os.path.relpath(item["target"], path.parent))
    result = {
        "mode": args.mode,
        "applied": args.apply,
        "candidateBytes": sum(item["bytes"] for item in candidates),
        "files": candidates,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
