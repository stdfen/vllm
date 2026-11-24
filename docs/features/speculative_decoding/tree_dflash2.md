# Tree-DFlash2 verification

Tree-DFlash2 is an opt-in verifier in V2 Model Runner for Qwen3.5/3.8
dense text targets on one CUDA GPU. It reuses a `DFlash2DraftModel` and the
predictor's conditional candidate scores.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
  --language-model-only --enforce-eager --no-async-scheduling \
  --no-enable-prefix-caching --kv-cache-dtype fp8 \
  --speculative-config '{
    "method": "dflash",
    "model": "/path/to/drafter",
    "num_speculative_tokens": 7,
    "enable_tree_dflash2": true,
    "tree_dflash2_max_replays": 2
  }'
```

The worker starts with the normal draft chain. On a greedy decode-only batch,
if the target's rejection token matches a valid sibling in the draft tree, the
worker restores the convolution and recurrent inputs and verifies that sibling
path. Every replay reuses the batch's allocated KV positions. The existing
runner commits the selected path and sends its hidden-state taps to the next
draft step. The scheduler discards unaccepted positions using its existing
draft rejection counts.

`tree_dflash2_max_replays` bounds candidate replay attempts and defaults to 2.
A divergent replay requires one extra forward to recover the previous path.
Mixed prefill batches, stochastic sampling, structured outputs, requests with
logprobs, and thinking budget controls stay on the normal DFlash2 path.
Adaptive verification, dynamic speculation, LoRA, watermarking, multimodal
inputs, prefix caching, and asynchronous scheduling are unsupported.

The default `replay` mode is the reference implementation. Set
`"tree_dflash2_mode": "packed"` to verify the bounded tree in one eager target
forward, or `"tree_dflash2_mode": "graph"` to capture and replay that forward
with fixed buffers per eligible request count. The modes use eight scheduled
rows per request: one root, a four-token primary branch, and a three-token
sibling branch. The verifier uses ancestor-only Triton Attention and runs GDN
level by level within the target forward. It compacts the winner into normal
KV and GDN slots and reorders the hidden-state taps before the next proposal.

For either packed mode, add `--attention-backend TRITON_ATTN`,
`--mamba-ssm-cache-dtype float32` and `--max-num-seqs 8` to the command above.
The target requires BF16 activations, FP8 E4M3 KV cache, CUDA GDN, and
`num_speculative_tokens=7`. Keep `--enforce-eager` for the ordinary V2 runner;
only the dedicated graph mode captures packed forwards. Use plain greedy
requests with `min_tokens=0` and default logits transformations. Mixed
prefill/decode, partial draft windows, and active context above 4096 tokens
use standard DFlash2. Dedicated graphs capture on the first qualifying batch
of each size; exclude that capture from steady-state timing.

The worker logs successful sibling selection, packed graph capture, and
packed graph replay. Ordinary speculative counters report scheduled slots
and committed tokens. They do not distinguish tree rows, and the ordinary
V2 CUDA graph counters do not cover the dedicated graph.

In TensorFold mode, `tree_dflash2_tensorfold_compile_prefill` enables the
vLLM-compiled target forward for batches containing prefill, including mixed
prefill and decode. Omit `--enforce-eager` and pass
`--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}'`.
Decode-only batches keep the TensorFold verifier and its dedicated Graph
settings.

For service qualification, compare `return_token_ids` against an eager
target-only run with the same target, KV dtype, and greedy requests. Cover
single-request and batched generation, partial acceptance, a winning sibling,
different output lengths and requests arriving during decode. Check the
sibling-selection log so an ordinary-chain fallback cannot satisfy the gate.
Qualify packed eager against replay and dedicated graph against packed eager
under identical backend, dtype, and engine settings. Check actual graph
replay at the tested request counts, not only a capture log.
## TensorFold round mode

Select `"tree_dflash2_mode": "tensorfold"` with
`"enable_tree_dflash2": true`, `"method": "dflash"` and
`"num_speculative_tokens": 15`. The 15 tokens are a lookahead ceiling.
The synchronous scheduler gets the actual variable number of draft rows per
request. The TensorFold mode requires the TensorFold 0.6.0 package, GB10,
local Qwen3.8 ModelOpt NVFP4 target and DFlash2 checkpoints, eager V2,
Triton Attention, FP8 E4M3 or BF16 KV, and FP32 GDN state. The package's CUDA extensions
are compiled with the running image's PyTorch and CUDA toolchain.

The TensorFold drafter reuses the target embedding and reads the saved NVFP4
head. It uses TensorFold's 4-bit projection packing, per-request tap windows,
context copies, and batched best-first tree and row allocator. The allocator
starts with the GB10 reference cost curve and updates it from target forwards.
vLLM owns requests, page allocation, model weights, and target projections.
The TensorFold tree attention maps committed keys through vLLM's page table
and dequantizes FP8 keys with the layer's scales. BF16 pages are read directly.
New tree nodes are read in
BF16 from the current forward. The kernel uses TensorFold's ancestor paths,
chunked softmax update, and ordered merge.

Parent-ordered draft rows are verified in one target forward. The TensorFold
GDN kernels read the committed recurrent state and return provisional row
outputs. Acceptance uses the target prediction at each node. The accepted GDN
paths are replayed with one cross-layer launch. The convolution history keeps
the tail of the old history and the accepted path at vLLM's next state offset.
Only accepted attention rows are quantized into the destination page slots,
and only the accepted taps feed the next proposal. The runner publishes the
normal accepted and rejected counts. Mixed prefill/decode batches still
verify any scheduled drafts as a linear prefix. Non-greedy and constrained
requests receive no TensorFold drafts.

Set `"tree_dflash2_tensorfold_cuda_graph": true` in the speculative config to
capture and replay the TensorFold verification forward and target argmax.
Keep `--enforce-eager` for the outer V2 runner. Before replay, the tree verifier
refreshes token IDs, positions, parent paths, page tables, and GDN source
slots. Sampling the accepted path and committing state run after replay.
By default, graphs are captured for recurring shapes up to 16 attention chunks.
Each runner retains at most eight graphs, with at most two per request count.
Other shapes use the same eager verification. With
`"tree_dflash2_tensorfold_long_context_graph": true`, eight-request batches
with 128 verification rows can also replay exact shapes with 17–19 chunks.
Each of those chunk counts has an independent two-shape cap.

Set `"tree_dflash2_tensorfold_c8_graph_buckets": true` to reuse verifier graphs
for eight short-context requests with 16 rows each. Attention capacities 5/6
and 7/8 share buckets 6 and 8. GDN scratch slots share the existing 0, 2, and
4 launch sizes. Exact tree paths, per-request positions, and the original
graph-count limits are retained.

Set `"tree_dflash2_tensorfold_batch_graph_buckets": true` to apply the same
short-prefix capacities to real batches of two through eight requests. Trees
may have different widths up to 16 rows. Captures retain the actual batch
size and replay only when the tensor shapes and launch parameters match.
An independent bank keeps at most one additional graph per batch size.
If the original C8 bucket is also enabled, full-width C8 batches continue
using it. With `TF_VERIFY_GRAPH_METRICS=1`, the
`vllm:tensorfold_batch_graph_bucket_total` counter reports execution by
batch size and route for the additional buckets.

Set `"tree_dflash2_tensorfold_c1_graph_bucket": true` with CUDA Graph to reuse
one launch shape across committed prefix lengths up to 4096 tokens for a
single request. The page table, tree paths, and actual lengths are refreshed
on each replay. Multi-request batches use the corresponding settings above.

Set `"tree_dflash2_tensorfold_c1_expand_graph_bank": true` with the C1 Graph
bucket to retain up to six recurring verification shapes in a separate bank.
Each captured shape uses its own graph buffers.

Set `"tree_dflash2_tensorfold_c1_fixed_width": true` with the native target
and C1 Graph bucket to validate up to 16 real rows in a 16-row graph. Short
trees use provisional padding leaves. Only real tree nodes can be accepted.

Set `"tree_dflash2_tensorfold_batch_device_commit": true` to select accepted
paths on CUDA for two to eight requests in a pure TensorFold tree batch.
Each request selects its own path and commits only its accepted KV rows,
recurrent state, and convolution history to the vLLM-owned cache. The worker
copies a compact receipt for the drafter and uses device row mappings for
runner compaction. Compatible captured target shapes also replay the batched
state commit. The option leaves single-request execution and the existing
draft budget unchanged.

Set `"tree_dflash2_tensorfold_c1_commit_graph": true` with the C1 Graph bucket
to replay KV, recurrent state, and convolution history updates after path
selection. It batches the recurrent slot copies and uses the standard
convolution history path.

Set `"tree_dflash2_tensorfold_c1_overlap_commit": true` with the C1 commit
Graph to dispatch the accepted cache writes on a side stream. The model
runner joins that stream before the next round can access or recycle the
cache. The draft model consumes the verification taps on the main stream.

Set `"tree_dflash2_tensorfold_c1_device_accept": true` with the C1
commit Graph to resolve greedy trees of at most 16 verification rows on CUDA.
The commit consumes device row indices directly while a compact receipt
returns the same accepted path and emitted IDs to the runner.
It can dispatch the commit on the side stream when overlap is enabled.

Set `"tree_dflash2_tensorfold_c1_direct_recurrent_commit": true` with device
acceptance and the TensorFold destination-slot replay extension to read the
previous recurrent state and write the accepted slot in one replay kernel.
Set `"tree_dflash2_tensorfold_c1_recurrent_commit_coverage": true` to apply
the same replay to both captured and ordinary C1 commits when verification
can run without a matching graph.

Set `"tree_dflash2_tensorfold_c1_resident_direct_stage": true` with device
acceptance to refresh captured C1 verification buffers from vLLM device slots.
The path keeps exact verification widths and uses the expanded C1 graph bank.
Set `"tree_dflash2_tensorfold_c1_grouped_resident_stage": true` to use the
same path with a persistent GDN layer layout. Each refresh then receives the
live slot and accepted-count tensors once per cache group.

Set `"tree_dflash2_tensorfold_c1_device_metadata": true` with device acceptance
to prepare recurrent state pointers directly from vLLM's device slot tables.
The synchronous scheduler supplies the committed length for C1 verification.

Set `"tree_dflash2_tensorfold_c1_direct_stage": true` to reuse the captured
verification buffers for matching single-request tree shapes with host cache
metadata. Changes in cache bindings fall back to the regular plan.

Set `"tree_dflash2_tensorfold_c1_reference_curve": true` to use the GB10
TensorFold reference costs for single-request tree allocation. This avoids
per-round calibration for those batches. Larger batches retain online costs.

Set `"tree_dflash2_tensorfold_c1_draft_graph": true` to capture the native
single-request DFlash2 draft block. This option needs the TensorFold device
draft entry point. The drafter still updates its rolling context and builds
the candidate tree with the existing policy. Set
`"tree_dflash2_tensorfold_c1_pinned_readback": true` to include the candidate
ID and score copies to pinned host memory in the draft Graph.

Set `"tree_dflash2_tensorfold_c1_async_draft_upload": true` to use a pinned
asynchronous host-to-device upload when writing the draft IDs back to a
single-request vLLM runner batch after proposal.

Set `"tree_dflash2_tensorfold_native_target": true` to use the native
TensorFold Qwen3.8-27B embedding, projections, normalizations, MLP, and
LM head for pure tree verification. It keeps the vLLM page tables and
accepted-state commit, and works with eager verification or the dedicated
CUDA Graph. This option retains a second layout of the target weights on the
device. Prefill and mixed batches use the regular vLLM model.

Set `"tree_dflash2_tensorfold_c1_qmm_split": true` with the native target to
select GB10 split counts for the qualified NVFP4 checkpoint's five projection
shapes. It applies to single-request verification with 16 rows. The changed
accumulation order can change low BF16 bits.

Set `"tree_dflash2_tensorfold_c1_fp8_tree_tail": true` to use the
configured FP8 E4M3 KV cache precision for provisional keys and values
inside single-request tree attention. The attention kernel applies the
cache's key and value scales before conversion and restores the scale
before the dot product. The paged cache writer commits accepted KV.

This mode uses the TensorFold concurrent eager round. The fixed-layout
`packed` and `graph` modes use their own verification path.

With the native TensorFold target, `"num_speculative_tokens": 31` lets the
scheduler reserve room to verify up to 31 draft candidates in one tree. The
draft forward still uses its original block width. The allocator chooses the
actual number of verified candidates for each round.

With the native target, `"tree_dflash2_tensorfold_fused_conv": true` batches
GDN convolution cache reads and accepted-path writes across layers within each
compatible BF16 cache slot group for one request. Independent groups keep their
own active cache slot and accepted offset during target verification and commit.
