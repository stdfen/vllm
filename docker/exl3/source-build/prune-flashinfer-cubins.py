"""Apply a hash-pinned, trace-reviewed cubin allowlist during installation."""

import argparse
import csv
import hashlib
import json
import pathlib
import struct
from collections import Counter


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--allowlist", type=pathlib.Path, required=True)
    parser.add_argument("--allowlist-sha256", required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if digest(args.allowlist) != args.allowlist_sha256:
        raise RuntimeError("Allowlist digest mismatch")
    policy = json.loads(args.allowlist.read_text())
    if policy.get("referenceValidation") != "passed" or not policy.get("traceSha256"):
        raise RuntimeError(
            "Reference validation and runtime trace evidence are required"
        )
    root = args.root.resolve(strict=True)
    if root.name != "cubins" or root.parent.name != "flashinfer_cubin":
        raise RuntimeError("Unexpected cubin root")
    observed = set(policy["keepObserved"])
    candidates = []
    for item in policy["remove"]:
        path = root / item["path"]
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_relative_to(root)
            or path.is_symlink()
            or path.suffix != ".cubin"
        ):
            raise RuntimeError(f"Invalid candidate: {item['path']}")
        if item["path"] in observed:
            raise RuntimeError("An observed runtime file must be retained")
        with path.open("rb") as source:
            header = source.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            raise RuntimeError("Unknown cubin format")
        machine = struct.unpack_from("<H", header, 18)[0]
        flags = struct.unpack_from("<I", header, 48)[0]
        architecture = (flags >> 8) & 255
        if machine != 190 or architecture not in {90, 100, 103, 107}:
            raise RuntimeError("Only proven non-GB10 CUDA ELF files may be removed")
        if item["sm"] != architecture or digest(path) != item["sha256"]:
            raise RuntimeError("Candidate architecture or content changed")
        candidates.append((path, path.stat().st_size))
    report = {
        "removedCount": len(candidates),
        "removedBytes": sum(size for _, size in candidates),
    }
    # Keep wheel metadata consistent with the files retained in this image.
    site_packages = root.parent.parent
    records = list(site_packages.glob("flashinfer_cubin-*.dist-info/RECORD"))
    if len(records) != 1:
        raise RuntimeError("Exactly one FlashInfer cubin RECORD is required")
    record = records[0]
    removed_names = {
        path.relative_to(site_packages).as_posix() for path, _ in candidates
    }
    if len(removed_names) != len(candidates):
        raise RuntimeError("Duplicate removal candidate")
    with record.open(newline="") as source:
        rows = list(csv.reader(source))
    if any(len(row) != 3 for row in rows):
        raise RuntimeError("Invalid wheel RECORD")
    recorded_names = Counter(row[0] for row in rows)
    if any(recorded_names[name] != 1 for name in removed_names):
        raise RuntimeError("Every candidate must appear exactly once in RECORD")
    retained_rows = [row for row in rows if row[0] not in removed_names]
    temporary_record = record.with_name("RECORD.slimming.tmp")
    if temporary_record.exists():
        raise RuntimeError("Unexpected temporary RECORD already exists")
    for path, _ in candidates:
        path.unlink()
    with temporary_record.open("x", newline="") as target:
        csv.writer(target, lineterminator="\n").writerows(retained_rows)
    temporary_record.replace(record)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
