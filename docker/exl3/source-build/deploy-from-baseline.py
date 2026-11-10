"""Create a candidate from the retained deployment without replacing it."""

import argparse
import copy
import hashlib
import http.client
import json
import pathlib
import socket
import urllib.parse


class DockerConnection(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=120)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def request(method, path, body=None):
    connection = DockerConnection()
    try:
        connection.request(
            method,
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = response.read()
        result = json.loads(data) if data else {}
        if response.status >= 400:
            raise RuntimeError(
                f"Docker HTTP {response.status}: {result.get('message', '')}"
            )
        return result
    finally:
        connection.close()


def environment(values):
    return dict(value.split("=", 1) for value in values or [])


def split_bind(value):
    """Split a Docker bind without confusing a Windows drive letter for a mode."""
    source, _, mode = value.rpartition(":")
    if not source:
        return None
    root, _, target = source.rpartition(":")
    if not root:
        return source, "", ""
    return root, target, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", default="dsv41-serve")
    parser.add_argument("--cache-root", type=pathlib.Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--trace-bundle", type=pathlib.Path)
    parser.add_argument("--trace-output", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    quote = urllib.parse.quote
    baseline = request("GET", f"/containers/{quote(args.baseline, safe='')}/json")
    if baseline["State"]["Running"]:
        raise RuntimeError(
            "Baseline is running; candidate would conflict with its ports and GPU"
        )
    old_image = request("GET", f"/images/{quote(baseline['Image'], safe='')}/json")
    new_image = request("GET", f"/images/{quote(args.image, safe='')}/json")
    old_defaults = environment(old_image["Config"].get("Env"))
    deployed = environment(baseline["Config"].get("Env"))
    merged = environment(new_image["Config"].get("Env"))
    merged.update(
        {
            key: value
            for key, value in deployed.items()
            if old_defaults.get(key) != value
        }
    )
    if args.offline:
        merged.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            FLASHINFER_NO_DOWNLOAD="1",
        )
    cache = args.cache_root / new_image["Id"].removeprefix("sha256:")
    if not cache.is_absolute():
        raise RuntimeError("Cache root must be absolute")
    for key, relative in {
        "CUDA_CACHE_PATH": "cuda",
        "TRITON_CACHE_DIR": "triton",
        "VLLM_CACHE_ROOT": "vllm",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "flashinfer-autotune",
        "TORCHINDUCTOR_CACHE_DIR": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "XDG_CACHE_HOME": "xdg",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "TILELANG_CACHE_DIR": "tilelang",
    }.items():
        merged[key] = f"/var/cache/vllm-runtime/{relative}"

    host = copy.deepcopy(baseline["HostConfig"])
    baseline_binds = host.get("Binds") or []
    binds = list(baseline_binds)
    host["Binds"] = [
        value
        for value in binds
        if (split_bind(value) or (None, None))[1]
        not in {"/root/.cache", "/var/cache/vllm-runtime"}
    ]
    host["Binds"].append(f"{cache}:/var/cache/vllm-runtime:rw")
    # The old cache bind also contains a required DSpark token-list input.
    # Preserve that individual file without bringing historical caches along.
    external_inputs = {}
    markov = merged.get("VLLM_DSPARK_MARKOV_CACHE")
    if markov and markov.startswith("/root/.cache/"):
        relative = pathlib.PurePosixPath(markov).relative_to("/root/.cache")
        if ".." in relative.parts:
            raise RuntimeError("Invalid Markov input path")
        parsed_binds = [split_bind(value) for value in binds]
        markov_binds = [
            pathlib.Path(parts[0])
            for parts in parsed_binds
            if parts and parts[1] == markov and parts[2] == "ro"
        ]
        baseline_cache_binds = [
            pathlib.Path(parts[0])
            for parts in (split_bind(value) for value in baseline_binds)
            if parts and parts[1] == "/root/.cache"
        ]
        if len(markov_binds) > 1:
            raise RuntimeError("Multiple Markov file binds are not allowed")
        if len(markov_binds) == 1:
            source = markov_binds[0].resolve(strict=True)
        elif len(baseline_cache_binds) == 1:
            old_cache_root = baseline_cache_binds[0].resolve(strict=True)
            if not old_cache_root.is_dir():
                raise RuntimeError("Baseline cache bind is not a directory")
            source = (old_cache_root / relative).resolve(strict=True)
        else:
            raise RuntimeError(
                "A unique baseline cache bind or Markov file bind is required"
            )
        if not markov_binds:
            allowed_root = baseline_cache_binds[0].resolve(strict=True)
            if not source.is_relative_to(allowed_root):
                raise RuntimeError("Markov input is outside its verified source root")
        if not source.is_file():
            raise RuntimeError("Markov input is not a regular file")
        with source.open("rb") as token_input:
            checksum = hashlib.file_digest(token_input, "sha256").hexdigest()
        host["Binds"] = [
            value
            for value in host["Binds"]
            if (split_bind(value) or (None, None))[1] != markov
        ]
        host["Binds"].append(f"{source}:{markov}:ro")
        external_inputs["markovTokenListSha256"] = checksum
    host["ContainerIDFile"] = ""
    host["RestartPolicy"] = {"Name": "no", "MaximumRetryCount": 0}
    config = baseline["Config"]
    body = {
        key: copy.deepcopy(config[key])
        for key in (
            "Cmd",
            "Entrypoint",
            "WorkingDir",
            "User",
            "StopSignal",
            "StopTimeout",
            "Tty",
            "OpenStdin",
            "Healthcheck",
            "ExposedPorts",
        )
        if key in config
    }
    body.update(
        Image=args.image,
        Env=[f"{key}={value}" for key, value in sorted(merged.items())],
        HostConfig=host,
        Labels={
            **(config.get("Labels") or {}),
            "ai.vllm.exl3.candidate-stage": args.stage,
            "ai.vllm.exl3.candidate-image-id": new_image["Id"],
        },
    )
    if bool(args.trace_bundle) != bool(args.trace_output):
        raise RuntimeError("Trace bundle and output must be provided together")
    if args.trace_bundle:
        bundle = args.trace_bundle.resolve(strict=True)
        if not args.trace_output.is_absolute():
            raise RuntimeError("Trace output must be absolute")
        for relative in ("strace", "ld-linux-aarch64.so.1", "lib/libc.so.6"):
            if not (bundle / relative).is_file():
                raise RuntimeError(f"Missing trace tool: {relative}")
        host["Binds"].extend(
            [
                f"{bundle}:/opt/vllm-trace:ro",
                f"{args.trace_output}:/var/log/vllm-trace:rw",
            ]
        )
        host["CapAdd"] = list(
            dict.fromkeys([*(host.get("CapAdd") or []), "SYS_PTRACE"])
        )
        host["SecurityOpt"] = [
            item
            for item in host.get("SecurityOpt") or []
            if not item.startswith("seccomp=")
        ] + ["seccomp=unconfined"]
        body["Entrypoint"] = [
            "/opt/vllm-trace/ld-linux-aarch64.so.1",
            "--library-path",
            "/opt/vllm-trace/lib",
            "/opt/vllm-trace/strace",
            "-ff",
            "-ttt",
            "-s",
            "4096",
            "-yy",
            "-e",
            "trace=openat,openat2,statx,newfstatat,access,readlink,readlinkat,mmap",
            "-o",
            "/var/log/vllm-trace/strace",
            "--",
            *(body.get("Entrypoint") or []),
        ]
    print(
        json.dumps(
            {
                "name": args.name,
                "imageId": new_image["Id"],
                "stage": args.stage,
                "baselineRunning": False,
                "cacheDirectory": str(cache),
                "environmentKeys": sorted(merged),
                "bindCount": len(host["Binds"]),
                "networkMode": host["NetworkMode"],
                "dryRun": args.dry_run,
                "offline": args.offline,
                "trace": bool(args.trace_bundle),
                "externalInputs": external_inputs,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    cache.mkdir(parents=True, exist_ok=True, mode=0o750)
    if args.trace_output:
        args.trace_output.mkdir(parents=True, exist_ok=True, mode=0o750)
    created = request(
        "POST",
        "/containers/create?" + urllib.parse.urlencode({"name": args.name}),
        body,
    )
    request("POST", f"/containers/{created['Id']}/start")
    print(json.dumps({"startedContainerId": created["Id"]}))


if __name__ == "__main__":
    main()
