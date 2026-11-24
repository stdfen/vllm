# FreeToken expert offload for Qwen3.6-35B-A3B

This experimental path ports the BF16 expert offload used in the
`vllm-qwen36-sm89-ft:20260909-decode-v4` image to vLLM 0.33.0
(`4ba787ec9959e566d3b587a796dff07673324da5`). Enable it with
`VLLM_QWEN36_FREETOKEN=1`. The default is `0`, which uses native vLLM
expert execution.

[Patch and image build guide](qwen36_freetoken/patch_and_image.md) ·
[Single RTX 4090 validation and A–B–A report](qwen36_freetoken/performance.md)

## Integration

1. `Qwen3_5Model` creates a FreeToken runtime for the supported MoE
   geometry when the feature flag is enabled. The Qwen sparse block
   passes a routed-expert subclass to `FusedMoEFactory`. vLLM continues
   to compute the router and shared expert.
2. Native UVA offloading retains the pinned CPU storage and its CUDA
   alias for each routed expert bank. The runtime checks the original
   pointer, expert shapes, BF16 storage and contiguity before registering
   all 40 expert layers. The same weights back UVA and FreeToken. In-place
   weight reload requires restarting the engine because FreeToken caches
   the bank addresses and expert data.
3. Prefill streams full expert layers through two alternating buffers.
   The vLLM Triton expert kernel processes each layer while the next
   layer is being copied. The buffers draw from the same GPU allocation
   as the FreeToken expert pool.
4. Single-token Decode uses the on-device LRU slot map, copies missing
   experts and executes the FreeToken GPU expert kernel. The same
   device-side operations run during `FULL_DECODE_ONLY` capture and
   replay.

During graph warmup, vLLM marks padding routes with expert ID `-1`.
FlashLib's LRU lookup expects an in-range expert ID. The decode adapter
uses valid scratch IDs and gives invalid routes zero weight. It zeros the
routed result for a fully padded token. This handling runs on the device
and is captured for replay. The uncorrected image failed startup on GPU
with an illegal CUDA memory access. The corrected image completed the
same startup and served the validation workload.

## Image dependencies

The recipe uses the official vLLM 0.33.0 amd64 CUDA base and keeps its
PyTorch 2.14.0+cu130, CUDA, Triton 3.8.0, and TVM FFI 0.1.11. It rebuilds
the three FreeToken extensions with C++20 and installs the pinned
`flashlib==0.3.0` wheel without resolving transitive dependencies.
The build checks the vLLM CUDA stable ABI and FreeToken imports, calls a
C++ function through TVM FFI, and compiles and loads the SM89 BatchMemcpy
and multi-bank copy bindings. These checks run with the CUDA Toolkit
driver stub during the build. GPU inference and CUDA Graph replay require
a separate check on a GPU. The resulting image includes a manifest with
the base and fork commits, FreeToken revision, and hashes of the vLLM
source files layered onto the release.

## Runtime requirements

- Use the BF16 Qwen3.6-35B-A3B checkpoint on one CUDA GPU with at most one
  sequence in flight. Select the modular Triton MoE backend.
- The image includes the FreeToken CUDA runtime from
  [`af71ba43206e124f5ff6419b47ee36c6e9981078`](https://github.com/FlashML-org/FreeToken/tree/af71ba43206e124f5ff6419b47ee36c6e9981078).
  When building outside the image, use the target PyTorch and CUDA
  toolchain and run the same import and TVM FFI checks before serving.
- Budget at least 60 GiB of pinned CPU memory for the 40 routed-expert
  layers, in addition to the memory needed for model loading and serving.
  Allow GPU memory for the expert pool, the resident model weights,
  activations, and KV cache. `VLLM_QWEN36_CACHE_GB` defaults to 13 GiB and
  includes the two prefill buffers. The minimum is 3 GiB.
- Use `mode=0` with CUDA Graph mode `NONE` or `FULL_DECODE_ONLY`.
  Speculative decoding, fused shared experts, LoRA, sleep mode,
  microbatching, and in-place weight reload are unsupported.

## Example

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_QWEN36_FREETOKEN=1 VLLM_QWEN36_CACHE_GB=13 \
  vllm serve /path/to/Qwen3.6-35B-A3B \
    --dtype bfloat16 --language-model-only --tensor-parallel-size 1 \
    --max-model-len 12288 --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --moe-backend triton --offload-backend uva --cpu-offload-gb 62 \
    --cpu-offload-params w13_weight w2_weight \
    --attention-backend TRITON_ATTN \
    --kv-cache-dtype fp8 --kv-cache-memory-bytes 1073741824 \
    --enable-chunked-prefill --enable-prefix-caching \
    --block-size 4096 --mamba-block-size 4096 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

`VLLM_QWEN36_FREETOKEN` defaults to `0`. An A–B–A comparison can use a
single vLLM 0.33.0 build and identical serve arguments, setting the
flag to `0`, `1`, and `0` respectively. Capture the effective cache
block sizes, actual prefill chunk lengths and graph replay evidence.
Validate token IDs before timing.
