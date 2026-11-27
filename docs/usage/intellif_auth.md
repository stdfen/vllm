# Intellif product licensing in this fork

This licensed build requires authorization from the existing IFAuth service.
It is separate from the OpenAI HTTP API key. Removing the entrypoint or using
the Python engine does not disable the framework checks. Administrators who
modify the Python code or native SDK are outside this client-side threat model.

## Runtime configuration

Mount a deployment-specific directory read-only at `/run/ifauth` and set
`IFAUTH_CONFIG_FILE=/run/ifauth/config.json`. Both TP nodes must use the same
deployment ID, policy, and binary control key. Keep configuration and keys out
of the image, source repository, build context, and logs.

```json
{
  "IFAUTH_ADDRS": "<existing-license-service>:8460",
  "IFAUTH_DEPLOYMENT_ID": "<unique-deployment-id>",
  "IFAUTH_COORDINATOR_HOST": "<head-node-reachable-ipv4>",
  "IFAUTH_COORDINATOR_PORT": 29795,
  "IFAUTH_CONTROL_KEY_FILE": "/run/ifauth/control.key",
  "IFAUTH_CHECK_INTERVAL_SECONDS": 10,
  "IFAUTH_CALL_TIMEOUT_SECONDS": 5,
  "IFAUTH_MAX_STATE_AGE_SECONDS": 30,
  "IFAUTH_STARTUP_TIMEOUT_SECONDS": 120,
  "IFAUTH_RECOVERY_SUCCESSES": 2
}
```

The fixed SDK product is `ifaas-vllm`. The control key authenticates internal
state exchange; it is not a license. Generate one random 32-byte binary key
per deployment, distribute it securely to its nodes, and reuse it on restart.
The SDK ABI must match ARM64, CPython 3.12, and the runtime dynamic libraries.

The head EngineCore owns the only SDK subprocess. Startup authorization precedes
executor creation and model loading. Other processes maintain local state using
the bounded HMAC channel. Explicit denial fails immediately; TTL is not a grace
period after a known failure. Runtime recovery needs consecutive successes.

## Request admission in vLLM 0.33

EngineCore checks its current cached license once when admitting a new scheduler
request. The normal HTTP generation path with one sequence has a single
decision. Input chunks for an admitted resumable session reuse that admission.
An API call that fans out into multiple scheduler requests is checked for each
admission.

After admission, a request may finish after revocation or expiry. New requests
receive 403 on explicit denial or 503 when authorization is unavailable.
Health reports unavailable while the front end's cached state is invalid.
The separate SDK process refreshes the owner's state in the background.
In the deployed release, a chat request rejected at scheduler admission after
the SSE stream has opened may receive a generic streaming error. The scheduler
still rejects the request.

Prefill, decode, sampling, speculative verification, and KV updates use the
normal vLLM execution path without authorization RPCs. Startup still requires
SDK authorization, the signed executor join, and worker checks around model
loading and warmup. Deployment-level IFAuth is separate from HTTP API keys.

Supported topology remains CUDA UniProc/MP, DP=1, PP=1, spawn, standard
workers, no fault tolerance, no split KV/encoder transfer, no context
parallelism, and no sleep mode.

## Release images and validation

The request-admission release uses the following ARM64 images with the TensorFold
DSpark path and `native_target` enabled in the deployment.

| TP rank | Image tag | Docker image ID prefix |
| --- | --- | --- |
| 0 | `local/vllm-dsv41:ifauth-request-v1-20261127-rank0` | `b89157bb16f9` |
| 1 | `local/vllm-dsv41:ifauth-request-v1-20261127-rank1` | `c06daab6ebb9` |

The release layers the request-admission changes onto the licensed TF v2 runtime
for each rank. For a rebuild from source, package the fork's v0.33 framework and
the matching ARM64 CPython 3.12 SDK, and mount deployment configuration and the
control key at runtime.

The CPU suite in `tools/intellif_auth/tests/` covers license state, transport,
scheduler admission, and framework error propagation. The deployed TP2 release
was also verified with GPU inference, eight image inputs, and SSE streaming.
With the SDK subprocess unavailable, an admitted stream completed, a new
request was rejected, and admission recovered after successful SDK checks.
