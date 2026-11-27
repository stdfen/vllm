# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Private subprocess protocol. All SDK calls happen only in this interpreter."""

import ctypes
import json
import os
import resource
import signal
import socket
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0)
    # PID 1 can legitimately be the owner inside a container.
    if os.getppid() != parent:
        return
    conn = socket.socket(fileno=int(sys.argv[1]))
    stream = conn.makefile("rwb", buffering=0)
    initialized = False
    while line := stream.readline(16385):
        if len(line) > 16384:
            return
        try:
            request = json.loads(line)
            import IFAuthSDKpy as sdk

            if request["op"] == "init" and not initialized:
                code = int(sdk.IFAuthSDKInitPy(request["addrs"], request["product"]))
                initialized = code == 0
            elif request["op"] == "status" and initialized:
                code = int(sdk.IFAuthSDKStatusPy())
            else:
                raise ValueError()
            reply = {"code": code}
        except Exception:
            reply = {"error": "SDK_CALL_FAILED"}
        stream.write(json.dumps(reply).encode() + b"\n")


if __name__ == "__main__":
    main()
