"""Inventory CUDA ELF architectures without changing any installed files."""

import argparse
import collections
import json
import pathlib
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    unknown = []
    for path in sorted(args.root.rglob("*.cubin")):
        with path.open("rb") as source:
            header = source.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            unknown.append(str(path.relative_to(args.root)))
            continue
        machine = struct.unpack_from("<H", header, 18)[0]
        flags = struct.unpack_from("<I", header, 48)[0]
        if machine != 190:
            unknown.append(str(path.relative_to(args.root)))
            continue
        groups[flags].append((path, path.stat().st_size))

    result = {"root": str(args.root), "groups": [], "unknown": unknown}
    for flags, files in sorted(groups.items()):
        sample = files[0][0]
        listing = subprocess.check_output(
            ["cuobjdump", "--list-elf", str(sample)], text=True
        )
        architectures = sorted(set(re.findall(r"sm_([0-9]+)", listing)))
        architecture = (flags >> 8) & 255
        if architectures != [str(architecture)]:
            raise RuntimeError(f"CUDA ELF flags disagree with cuobjdump: {sample}")
        result["groups"].append(
            {
                "flags": hex(flags),
                "sm": architecture,
                "count": len(files),
                "bytes": sum(size for _, size in files),
                "sample": str(sample.relative_to(args.root)),
            }
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
