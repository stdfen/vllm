"""Fetch canonical, hash-checked ARM64 wheels over parallel HTTP/1.1 ranges."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

PACKAGES = {
    "nvidia-cublas": "13.1.1.3",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "nvidia-cusolver": "12.0.4.66",
    "nvidia-cusparselt-cu13": "0.8.1",
    "nvidia-nccl-cu13": "2.30.7",
    "nvidia-cufft": "12.0.0.61",
    "triton": "3.7.1",
    "nvidia-cusparse": "12.6.3.3",
    "nvidia-curand": "10.4.0.35",
    "nvidia-nvshmem-cu13": "3.4.5",
    "tilelang": "0.1.12",
    "z3-solver": "4.15.4.0",
    "flashinfer-python": "0.6.18",
}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "wheelhouse"
    chunks = root / "wheel-download-parts"
    output.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(exist_ok=True)
    metadata_path = root / "wheelhouse.lock.json"
    if metadata_path.exists():
        rows = json.loads(metadata_path.read_text())["wheels"]
    else:
        rows = []
        for package, version in PACKAGES.items():
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
            ) as response:
                metadata = json.load(response)
            compatible = [
                item
                for item in metadata["urls"]
                if item["filename"].endswith(".whl")
                and (
                    "aarch64" in item["filename"]
                    or item["filename"].endswith("none-any.whl")
                )
                and (
                    "-py3-" in item["filename"]
                    or "-cp312-" in item["filename"]
                    or "-cp38-abi3-" in item["filename"]
                )
            ]
            if len(compatible) != 1:
                raise RuntimeError(
                    f"Ambiguous ARM64 wheel: {package}: {len(compatible)}"
                )
            row = compatible[0]
            assert pathlib.PurePath(row["filename"]).name == row["filename"]
            rows.append(
                {
                    "package": package,
                    "version": version,
                    "filename": row["filename"],
                    "size": row["size"],
                    "sha256": row["digests"]["sha256"],
                    "canonicalUrl": row["url"],
                    "mirrorUrl": row["url"].replace(
                        "https://files.pythonhosted.org/",
                        "https://pypi.tuna.tsinghua.edu.cn/",
                    ),
                }
            )
        metadata_path.write_text(
            json.dumps({"schemaVersion": 1, "wheels": rows}, indent=2)
        )
    jobs = []
    block = 4 * 1024 * 1024
    for row in rows:
        destination = output / row["filename"]
        if (
            destination.exists()
            and destination.stat().st_size == row["size"]
            and sha(destination) == row["sha256"]
        ):
            continue
        for start in range(0, row["size"], block):
            jobs.append((row, start, min(start + block, row["size"]) - 1))
    print(
        json.dumps(
            {"totalBytes": sum(row["size"] for row in rows), "rangeJobs": len(jobs)}
        ),
        flush=True,
    )

    def fetch(job):
        row, start, end = job
        path = chunks / (row["sha256"] + "." + str(start))
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        partial = path.with_suffix(path.suffix + ".part")
        for attempt in range(6):
            try:
                request = urllib.request.Request(
                    row["mirrorUrl"], headers={"Range": f"bytes={start}-{end}"}
                )
                with (
                    urllib.request.urlopen(request, timeout=30) as response,
                    partial.open("wb") as target,
                ):
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{row['size']}"
                    ):
                        raise RuntimeError("Incorrect range response")
                    while data := response.read(1024 * 1024):
                        target.write(data)
                if partial.stat().st_size != end - start + 1:
                    raise RuntimeError("Truncated range")
                partial.replace(path)
                return
            except (OSError, ValueError, RuntimeError):
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, _ in enumerate(pool.map(fetch, jobs), 1):
            if number % 16 == 0:
                print(
                    json.dumps({"rangesCompleted": number, "rangesTotal": len(jobs)}),
                    flush=True,
                )
    for row in rows:
        destination = output / row["filename"]
        if destination.exists() and sha(destination) == row["sha256"]:
            continue
        partial = destination.with_suffix(".whl.part")
        with partial.open("wb") as target:
            for start in range(0, row["size"], block):
                with (chunks / (row["sha256"] + "." + str(start))).open("rb") as source:
                    while data := source.read(1024 * 1024):
                        target.write(data)
        if partial.stat().st_size != row["size"] or sha(partial) != row["sha256"]:
            raise RuntimeError("Wheel checksum mismatch: " + row["filename"])
        partial.replace(destination)
        print("VERIFIED " + row["filename"], flush=True)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{row['sha256']}  {row['filename']}\n"
            for row in sorted(rows, key=lambda r: r["filename"])
        )
    )
    (output / "wheelhouse.lock.json").write_text(metadata_path.read_text())
    print("WHEELHOUSE_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
