# Independent TensorFold DSpark image

This opt-in GB10 DSV4.1 deployment uses `dspark_implementation=tensorfold_full`.
It retains the native DSpark implementation and requires a checkpoint with
the matching Markov and confidence weights. See the
[deployment guide](../../../docs/deployment/sudoingx_clean_port.md) for scope,
configuration, correctness checks and the native A1/B/native A2 results.

## Retained release

The validated release is
`local/vllm-dsv41:dspark-tensorfold-graph24-v1-20261123`, with image ID
`sha256:e8ec98062462d93222877c36ca21852d13bdfb87a1dc36384712712a288ac68e`.
The complete retained image archive has SHA256
`671d3a0addac7881f7f9ae7313c0b1ce9ca9dc851ed85cd089d69193455ead82`.
`release-reference.json` records the measured source and asset fingerprints.
An image restored from that archive preserves its identity; rebuilding from
this checkout produces a separate source-build image.

## Build from this checkout

The frozen Arm64 base image
`local/sudoingx-clean-port:tensorfold-full-runtime-v7-20261121` must already
be available locally. It contains the required model operators and transport.
Model weights are mounted separately and are not included in either image.

```bash
.venv/bin/python examples/deployment/dspark_tensorfold/make_context.py \
  --output /tmp/dspark-image-context
docker build --network=none -t local/vllm-dsv41:graph24-source-build \
  /tmp/dspark-image-context
```

The generated context copies this checkout's independent modules and runner,
plus the prewarm/monitor helpers and frozen frequent-token asset. It computes
fresh file fingerprints rather than claiming byte identity with the retained
release. The three native-source checks continue to describe the frozen base.

Serve with five reserved drafts, greedy draft sampling, the independent
implementation, no generic adaptive verification, and `FULL_DECODE_ONLY`
capture buckets 1 through 24. The tested deployment uses TP2, maximum eight
requests, `max_model_len=524288`, `max_num_batched_tokens=2048`, 8 GiB KV,
block size 64 and disabled prefix caching/FlashInfer autotuning.

The monitor performs scratch prewarm before activation. Its default `serving`
control permits warnings for new runtime compilation keys. Separate strict
canary/formal control files reject monitored compilation and autotuning;
these benchmark gates are not the serving restart policy.

The measured workload is 1K/8K input plus 1024 output at sustained C2/C4/C8.
Maximum configured context is not evidence that eight simultaneous maximum
contexts have been validated. Native A/A and cross-path streamed texts differ;
the reported performance does not establish token-identical outputs.
