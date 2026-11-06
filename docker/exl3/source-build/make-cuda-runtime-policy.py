"""Approve exact CUDA candidates against a successful source reference and traces."""

import argparse
import hashlib
import json
import pathlib


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--reference-validation", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-smoke", type=pathlib.Path, required=True)
    parser.add_argument("--reference-image-id", required=True)
    parser.add_argument("--rank0-traces", type=pathlib.Path, required=True)
    parser.add_argument("--rank1-traces", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.reference_validation.read_text())
    smoke = json.loads(args.kernel_smoke.read_text())
    if (
        receipt.get("status") != "passed"
        or receipt.get("imageId") != args.reference_image_id
        or receipt.get("usage", {}).get("completion_tokens", 0) <= 0
        or smoke.get("status") != "passed"
    ):
        raise RuntimeError(
            "Successful reference generation and cold kernel checks required"
        )
    inventories = {
        mode: json.loads((args.inventory / f"{mode}.json").read_text())
        for mode in ("static", "dedupe")
    }
    static_names = {
        pathlib.PurePosixPath(row["path"]).name.encode()
        for row in inventories["static"]["files"]
    }
    traces = []
    if args.rank0_traces.resolve() == args.rank1_traces.resolve():
        raise RuntimeError("Separate traces for each rank required")
    for rank, directory in ((0, args.rank0_traces), (1, args.rank1_traces)):
        files = sorted(directory.glob("strace*"))
        if not files:
            raise RuntimeError(f"Missing rank {rank} traces")
        for path in files:
            checksum = hashlib.sha256()
            with path.open("rb") as source:
                for line in source:
                    checksum.update(line)
                    if any(name in line for name in static_names):
                        raise RuntimeError("Static candidate appeared in runtime trace")
            traces.append(
                {"rank": rank, "file": path.name, "sha256": checksum.hexdigest()}
            )
    # The applier independently reconstructs every file and verifies exact bytes
    # before unlinking or replacing duplicates with relative symlinks.
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f"{mode}.json").exists() for mode in inventories):
        raise RuntimeError("Refusing to overwrite an existing reviewed manifest")
    for mode, inventory in inventories.items():
        policy = {
            **inventory,
            "referenceValidation": "passed",
            "referenceImageId": args.reference_image_id,
            "referenceValidationSha256": digest(args.reference_validation),
            "kernelSmokeSha256": digest(args.kernel_smoke),
            "inventorySha256": digest(args.inventory / f"{mode}.json"),
            "traceSha256": hashlib.sha256(
                json.dumps(traces, sort_keys=True).encode()
            ).hexdigest(),
            "traceFiles": traces,
        }
        output = args.output / f"{mode}.json"
        output.write_text(json.dumps(policy, indent=2) + "\n")
        print(json.dumps({"mode": mode, "sha256": digest(output)}))


if __name__ == "__main__":
    main()
