"""Validate an OCI layout against Docker's config, repairing exporter index mixups."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import pathlib

MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=pathlib.Path, required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument("--repair-index", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    layout = args.layout.resolve(strict=True)
    blobs = layout / "blobs/sha256"

    def checked_blob(descriptor):
        algorithm, checksum = descriptor["digest"].split(":", 1)
        if (
            algorithm != "sha256"
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise RuntimeError("Invalid blob digest")
        path = blobs / checksum
        if (
            path.is_symlink()
            or path.stat().st_size != descriptor["size"]
            or digest(path) != checksum
        ):
            raise RuntimeError("OCI blob size/hash mismatch")
        return path

    index_path = layout / "index.json"
    original = index_path.read_bytes()
    index = json.loads(original)
    if len(index["manifests"]) != 1:
        raise RuntimeError("Expected a single-platform source image layout")
    descriptor = index["manifests"][0]
    if (blobs / descriptor["digest"].removeprefix("sha256:")).is_file():
        manifest = json.loads(checked_blob(descriptor).read_text())
    else:
        if not args.repair_index:
            raise RuntimeError("OCI index references a missing manifest")
        candidates = []
        for path in blobs.iterdir():
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                continue
            try:
                candidate = json.loads(path.read_bytes())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(candidate, dict):
                continue
            if (
                candidate.get("mediaType") in MANIFEST_TYPES
                and candidate.get("config", {}).get("digest") == args.config_digest
            ):
                if digest(path) != path.name:
                    raise RuntimeError("Manifest blob digest mismatch")
                candidates.append((path, candidate))
        if len(candidates) != 1:
            raise RuntimeError(
                "Cannot uniquely recover this config's exported manifest"
            )
        path, manifest = candidates[0]
        descriptor = {
            **descriptor,
            "mediaType": manifest["mediaType"],
            "digest": "sha256:" + path.name,
            "size": path.stat().st_size,
        }
    if manifest.get("config", {}).get("digest") != args.config_digest:
        raise RuntimeError("OCI image does not match the loaded Docker image config")
    config = json.loads(checked_blob(manifest["config"]).read_text())
    if config.get("architecture") != "arm64" or config.get("os") != "linux":
        raise RuntimeError("Expected linux/arm64 image")
    layers = manifest["layers"]
    diff_ids = config["rootfs"]["diff_ids"]
    if len(layers) != len(diff_ids):
        raise RuntimeError("Layer/diff-ID count mismatch")

    def verify_layer(item):
        layer, expected_diff = item
        path = checked_blob(layer)
        media = layer["mediaType"]
        if media.endswith(("+gzip", ".gzip")):
            open_layer = gzip.open
        elif media.endswith(".tar"):
            open_layer = pathlib.Path.open
        else:
            raise RuntimeError("Unsupported layer compression")
        checksum = hashlib.sha256()
        size = 0
        with open_layer(path, "rb") as source:
            while block := source.read(1024 * 1024):
                checksum.update(block)
                size += len(block)
        if "sha256:" + checksum.hexdigest() != expected_diff:
            raise RuntimeError(
                "Uncompressed layer does not match Docker RootFS diff-ID"
            )
        return size

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        layer_sizes = list(executor.map(verify_layer, zip(layers, diff_ids)))
    index["manifests"] = [descriptor]
    repaired = descriptor != json.loads(original)["manifests"][0]
    if repaired:
        backup = layout / "index.before-repair.json"
        if not backup.exists():
            backup.write_bytes(original)
        temporary = layout / "index.repair.tmp"
        temporary.write_text(json.dumps(index, separators=(",", ":")) + "\n")
        temporary.replace(index_path)
    report = {
        "status": "passed",
        "repairApplied": repaired,
        "configDigest": args.config_digest,
        "manifestDigest": descriptor["digest"],
        "indexSha256": digest(index_path),
        "layerCount": len(layers),
        "compressedLayerBytes": sum(layer["size"] for layer in layers),
        "uncompressedLayerTarBytes": sum(layer_sizes),
        "allDiffIdsVerified": True,
        "platform": "linux/arm64",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
