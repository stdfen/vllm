# sudoingX clean-port development status

The DeepSeek-V4.1 clean port is experimental and not ready for serving.
`VLLM_DSPARK_OWN_HEAD=1` requests an independent quantized draft head;
missing head pieces must fail loading rather than alias the target head.
This path still requires real model loading and multi-rank acceptance tests.

The experimental Engram fields `disk_offload`, `disk_threads`, and
`disk_cache_rows` select checkpoint-backed rows. Disk mode requires CPU offload,
no shared DP storage, no THP, and no cross-DP embedding sharding. Use the default
lazy safetensors loader: eager/prefetch, fastsafetensors, instanttensor, and
multithread loading are not supported for selective tables. Reader failures
terminate the worker rather than returning substituted rows. This path has
single-rank layer tests; full-model and TP2 acceptance remain outstanding.
Do not treat enabling this option as deployment approval. Default Engram
behavior remains unchanged.
Production images using disk offload must package the ABI-matched native reader
as `vllm.models.deepseek_v41.common.vllm_engram_disk` and set
`VLLM_ENGRAM_DISK_REQUIRE_PREBUILT=1`. With that setting a missing reader
fails closed instead of invoking the development JIT compiler. Validate the
reader against the image's exact PyTorch and CUDA versions before deployment.

`disk_direct_io=true` selects O_DIRECT for the disk reader and requires
`disk_offload=true`. It uses 4096-byte aligned bounce reads and never falls back
to buffered reads. Verify alignment support on the target filesystem. This is a
synchronous threaded reader, not the author's spin/native-AIO implementation;
its production performance and end-to-end model acceptance remain unverified.

### V36 fixed-window depth policy (port in progress)

`VLLM_DSPARK_DEPTH_POLICY=conf` enables the trained DSpark confidence head and
selects depth by prefix survival / measured round cost. `fixed:d` and `off` are
also supported. Costs require exactly k+1 finite positive entries; draft cost
may be zero. The current port rejects adaptive verification, batch-sharded
sampling, PP/PCP and disabled MoE padding instead of silently ignoring policy.
Rank-zero policy-file reload is inherited from the author. Runner/TP/graph and
probabilistic acceptance integration remain unverified; this is not V36 approval.

Setting `VLLM_ENGRAM_DISK_SPIN_MS` to a nonzero value with disk offload is
currently rejected: the spin/native-AIO path is not implemented. It cannot be
claimed as enabled by copying the historical environment variable.
