"""Create a conservative removal policy from a validated two-node reference."""

import argparse
import hashlib
import json
import pathlib
import re
import struct
import subprocess

PACKAGE_MARKER = "/flashinfer_cubin/cubins/"
PATH_PATTERN = re.compile(r'/flashinfer_cubin/cubins/([^"<>\s]+\.cubin)')
REMOVABLE_ARCHITECTURES = {90, 100, 103, 107}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--rank0-traces", type=pathlib.Path, required=True)
    parser.add_argument("--rank1-traces", type=pathlib.Path, required=True)
    parser.add_argument("--reference-validation", type=pathlib.Path, required=True)
    parser.add_argument("--reference-image-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.reference_validation.read_text())
    if (
        validation.get("status") != "passed"
        or validation.get("imageId") != args.reference_image_id
        or validation.get("usage", {}).get("completion_tokens", 0) <= 0
    ):
        raise RuntimeError(
            "A successful generation receipt for this reference image is required"
        )
    root = args.root.resolve(strict=True)
    if root.name != "cubins" or root.parent.name != "flashinfer_cubin":
        raise RuntimeError("Unexpected cubin root")
    if args.rank0_traces.resolve() == args.rank1_traces.resolve():
        raise RuntimeError("Separate trace directories for both ranks are required")
    observed = set()
    traces = []
    for role, directory in (("rank0", args.rank0_traces), ("rank1", args.rank1_traces)):
        files = sorted(path for path in directory.rglob("strace*") if path.is_file())
        if not files:
            raise RuntimeError(f"No runtime traces found for {role}")
        rank_hits = 0
        for path in files:
            checksum = hashlib.sha256()
            with path.open("rb") as source:
                for raw_line in source:
                    checksum.update(raw_line)
                    line = raw_line.decode(errors="replace")
                    if any(
                        marker in line
                        for marker in (
                            "/flashinfer/",
                            "/flashinfer_cubin/",
                            "/flashinfer_jit_cache/",
                        )
                    ):
                        rank_hits += 1
                    if PACKAGE_MARKER not in line:
                        continue
                    # Even metadata-only accesses are retained conservatively.
                    for match in PATH_PATTERN.finditer(line):
                        relative = pathlib.PurePosixPath(match.group(1))
                        if relative.is_absolute() or ".." in relative.parts:
                            raise RuntimeError("Unexpected cubin path in runtime trace")
                        observed.add(relative.as_posix())
            traces.append(
                {
                    "rank": role,
                    "file": path.relative_to(directory).as_posix(),
                    "sha256": checksum.hexdigest(),
                }
            )
        if not rank_hits:
            raise RuntimeError(f"No FlashInfer package accesses observed for {role}")

    architecture_checks = {}
    remove = []
    retained_unknown = []
    for path in sorted(root.rglob("*.cubin")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError("Unexpected cubin symlink")
        with path.open("rb") as source:
            header = source.read(64)
        if (
            len(header) != 64
            or header[:6] != b"\x7fELF\x02\x01"
            or struct.unpack_from("<H", header, 18)[0] != 190
        ):
            retained_unknown.append(relative)
            continue
        flags = struct.unpack_from("<I", header, 48)[0]
        architecture = (flags >> 8) & 255
        if flags not in architecture_checks:
            listing = subprocess.check_output(
                ["cuobjdump", "--list-elf", str(path)], text=True
            )
            found = sorted(set(re.findall(r"sm_([0-9]+)", listing)))
            if found != [str(architecture)]:
                raise RuntimeError("CUDA ELF flags disagree with cuobjdump")
            architecture_checks[flags] = architecture
        if architecture in REMOVABLE_ARCHITECTURES and relative not in observed:
            remove.append(
                {
                    "path": relative,
                    "sm": architecture,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
            )

    trace_sha = hashlib.sha256(json.dumps(traces, sort_keys=True).encode()).hexdigest()
    policy = {
        "schemaVersion": 1,
        "referenceValidation": "passed",
        "referenceImageId": args.reference_image_id,
        "referenceValidationSha256": digest(args.reference_validation),
        "traceSha256": trace_sha,
        "traceFiles": traces,
        "architectureChecks": {
            hex(flags): sm for flags, sm in architecture_checks.items()
        },
        "keepObserved": sorted(observed),
        "retainedUnknown": retained_unknown,
        "remove": remove,
        "candidateBytes": sum(row["bytes"] for row in remove),
    }
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite an existing reviewed policy")
    args.output.write_text(json.dumps(policy, indent=2) + "\n")
    print(
        json.dumps(
            {
                "policySha256": digest(args.output),
                "candidateBytes": policy["candidateBytes"],
                "candidateCount": len(remove),
                "observedCount": len(observed),
            }
        )
    )


if __name__ == "__main__":
    main()
