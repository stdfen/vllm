"""Wait for a candidate and verify a real, deterministic generation request."""

import argparse
import http.client
import json
import pathlib
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


class DockerConnection(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=10)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def container_state(name):
    connection = DockerConnection()
    try:
        connection.request(
            "GET", "/containers/" + urllib.parse.quote(name, safe="") + "/json"
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Container inspection returned HTTP {response.status}")
        return json.loads(response.read())
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dsv41")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    address = urllib.parse.urlsplit(args.base_url)
    if address.scheme != "http" or address.hostname not in {"localhost", "127.0.0.1"}:
        raise RuntimeError("Only the local candidate HTTP endpoint is permitted")
    state = container_state(args.container)
    environment = dict(item.split("=", 1) for item in state["Config"].get("Env", []))
    headers = {"Content-Type": "application/json"}
    # Keep any existing deployment credential in memory, never in the receipt.
    if environment.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + environment["VLLM_API_KEY"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, payload=None):
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        with opener.open(request, timeout=180 if payload else 10) as response:
            return json.load(response)

    started = time.monotonic()
    deadline = started + args.wait_seconds
    while True:
        state = container_state(args.container)
        if not state["State"]["Running"]:
            raise RuntimeError(
                f"Candidate exited with code {state['State']['ExitCode']}"
            )
        try:
            models = call("/v1/models")
            if args.model not in {item["id"] for item in models.get("data", [])}:
                raise RuntimeError("Requested model is absent from /v1/models")
            break
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise RuntimeError(
                    "Local candidate rejected deployment authentication"
                ) from None
            if time.monotonic() >= deadline:
                raise RuntimeError("Candidate readiness timed out") from None
            time.sleep(5)
        except (urllib.error.URLError, TimeoutError, OSError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Candidate readiness timed out") from None
            time.sleep(5)
    ready_seconds = time.monotonic() - started
    request_started = time.monotonic()
    result = call(
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "请直接回答：1+1等于多少？"}],
            "temperature": 0,
            "seed": 7,
            "max_tokens": 64,
            "stream": False,
        },
    )
    message = result["choices"][0]["message"]
    if not (
        message.get("content")
        or message.get("reasoning_content")
        or message.get("reasoning")
    ):
        raise RuntimeError("Generation returned no text")
    if result.get("usage", {}).get("completion_tokens", 0) <= 0:
        raise RuntimeError("Generation returned no completion tokens")
    receipt = {
        "status": "passed",
        "container": args.container,
        "imageId": state["Image"],
        "model": args.model,
        "readinessWaitSeconds": ready_seconds,
        "requestSeconds": time.monotonic() - request_started,
        "usage": result["usage"],
        "finishReason": result["choices"][0]["finish_reason"],
        "response": message,
        "scope": "Local model listing and real generation; not a throughput benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps({key: value for key, value in receipt.items() if key != "response"})
    )


if __name__ == "__main__":
    main()
