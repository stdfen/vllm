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

### Experimental independent DSpark prefix policy

`dspark_verification_policy="native"` is the default and preserves the original
DSpark and `VLLM_DSPARK_DEPTH_POLICY` routes. To select the independent policy:

```json
{
  "method": "dspark",
  "num_speculative_tokens": 5,
  "draft_sample_method": "probabilistic",
  "enable_adaptive_verification": true,
  "dspark_verification_policy": "tensorfold"
}
```

Unset `VLLM_DSPARK_DEPTH_POLICY` and `VLLM_DSPARK_DEPTH_FILE` for this route.
It requires a trained confidence head, standard rejection sampling, full CUDA
graphs, PP1/PCP1/DP1, no LoRA, and no batch-sharded sampling. Host compaction
keeps CPU and GPU query boundaries identical and captures uniform decode
graphs for each depth. Nonuniform batches fall back to eager execution.
Native adaptive verification retains its device-only trimming checks.
A successful CPU policy test does not qualify SM121 or TP2.

The policy uses each position's conditional confidence `q[j]` to evaluate
`E(d) = 1 + sum(product(q[:j]))` for prefixes `d=0..k`. It chooses the first
maximum of `E(d) / (verify_ms[other_rows + d + 1] + draft_ms)`. Other requests'
scheduled windows are held fixed during this independent selection; a binding
batch logit budget removes the lowest-survival tails. This is an approximation
for C>1, not an exhaustive joint optimizer. Target verification actually runs
the chosen prefixes through vLLM's compact batch, with graph bucket padding;
dense, attention and MoE work need not shrink inside the same capture bucket.

Costs default to vLLM's startup-profiled draft and target curves, including
capture-bucket padding. Optional `dspark_tensorfold_verify_ms` gives positive,
non-decreasing verify costs for TOTAL live rows 1..N; it must cover the entire
scheduled batch. `dspark_tensorfold_draft_ms` is the nonnegative cost of one
parallel drafter pass. These overrides do not use the legacy depth-file format
(which already includes drafter time). Never reuse TensorFold's timings for
vLLM. Set `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` to the frozen profile
context before starting either arm of a comparison.

Only the admission policy is new: the DSpark backbone, sequential Markov head,
draft sampling method, target rejection sampler and KV commit machinery remain
vLLM's. `draft_sample_method="greedy"` is an independent experimental variable;
do not change it while attributing a speedup to compact verification. The port
does not import TensorFold kernels, change quantization or enable online cost
exploration. It waits for the latest confidence D2H copy and broadcasts host
depths once per verification round. Measure that synchronization cost too.

Standard scheduler metrics count all proposed drafts on both routes; local
policy logs separately count admitted drafts. Acceptance percentages cannot
serve as the sole performance criterion. Use accepted tokens per round, output
tokens per round, TTFT, TPOT, output throughput and decode tokens/s.

#### Frozen A-B-A validation

No performance acceptance is claimed yet. Before deployment, review both ranks'
image IDs, patched source hashes, weight manifests, commands and environment.
Build the same patched image for A and B: A selects `native` with the original
depth-file configuration; B selects `tensorfold` and unsets the legacy policy.
Keep target/drafter weights, sampling, Engram settings, graph capture sizes,
profiling context, TP topology and request order fixed. Repeat A after B.

Use the frozen ShareGPT prompts from the baseline: exact 1024/8192 chat input
tokens, 1024 actual output tokens, C1, temperature 0, and measured zero cached
tokens. Reuse the SAME prompts in all three arms (not regenerated per arm).
Keep warmup separate, gather per-request SSE/usage and metrics deltas, and check
for unrelated traffic or restarts. Verify final KV positions, zero-depth,
near-EOS and truncation behavior before measuring speed. For deterministic
token equality require batch-invariant execution, otherwise compare output
logprobs under the repository's model-evaluation protocol. Include sampled
requests in correctness checks; changing draft depth does not by itself promise
bit-identical stochastic draws.

First compare native versus compact verification with probabilistic drafting.
Only then run a separate A-B-A for greedy drafting. Report all three arms,
request counts and the A-to-A drift. Do not claim the earlier two-framework
baseline as the control arm: it uses different model weights and binaries.

### Independent pinned TensorFold DSV4.1 lane implementation

`dspark_implementation="tensorfold_full"` selects an independent experimental
C1 chain implementation. `native` remains the default. This differs from the
earlier `dspark_verification_policy="tensorfold"` prefix-only experiment.

```json
{
  "method": "dspark",
  "num_speculative_tokens": 5,
  "draft_sample_method": "greedy",
  "enable_adaptive_verification": false,
  "dspark_verification_policy": "native",
  "dspark_implementation": "tensorfold_full"
}
```

Unset the legacy depth policy and depth file. The pinned deployment source is
TensorFold `fbbc97132cd004237d9887b8d3185896efaedc5c` with the image's ordered
patch set `d2248683348c`. Its native lane loop uses `core/lanes/windows.zig`
allocation, live overhead EWMA, suffix-copy proposals, greedy Markov drafts and
keyed target-row sampling followed by exact prefix matching. `draft.chooseK`
alone is not this deployed algorithm. Its exact window is six rows (five
drafts), and suffix copies share that limit.

The port implements those lane decisions independently of `DSparkSpeculator`,
the adaptive manager, legacy depth policy and `RejectionSampler`. vLLM supplies
checkpoint loading, model operators, generic DFlash device/KV staging and graph
transport, and the serving/scheduler bookkeeping. This is an algorithm port;
it does not import the Zig engine or reproduce all of TensorFold's native
matrix/attention kernels. CPU fixtures execute the pinned original Zig code.

The first integration validates C1, PP1/DP1/PCP1, no LoRA or batch-sharded
sampling, full-vocabulary DSV4.1 draft/confidence weights, and no structured
outputs. Keyed sampling follows TensorFold's top-k compatibility rules, including
default 20 when no top-k is supplied and a maximum of 56 candidates. That first
integration rejected multi-request execution. The experimental multi-request
extension and its narrower validation scope are described below.

`VLLM_TENSORFOLD_TRACE` optionally writes leader-side per-request round counters:
proposed, actually verified, accepted, output, copy/head kind and measured round
time. Leave this unset for serving-speed benchmarks; collect stage diagnostics
in separate runs, and keep the trace setting identical across comparison arms.
Source-lock evidence and experiment scripts live under
`benchmarks/dspark-tensorfold-full-path` in the surrounding workspace.

The pinned C1 port passed 325 source-oracle tests (including native Rule cost
feedback), 21 real CUDA acceptance/KV cases, request-lifecycle checks, and all
4,226 formal trace rounds replayed against the pinned Zig allocator/acceptance
and overhead feedback. The 104 actual suffix-copy windows matched the original
proposer on their prompt and committed context. Finished/preempted requests drop
their independent lane and held-token state through the runner removal hook.

A same-image, same-weight, same-request A1-C-A2 on TP2 GB10 used exact 1024/8192
chat input tokens, 1024 output tokens, C1/H0, five measured requests plus one
warmup per cell. C decode medians were 57.58/59.61 tokens/s versus native
A1 62.54/63.71 and A2 58.23/61.75. Acceptance rose to 62.07%/61.59%, but output
per round was 2.392/2.460, below both native controls. This run does not establish
a speedup; the option remains experimental and native remains the default.

The performance image is `tensorfold-full-runtime-v7-20261121`, digest
`ebb12adf6217882a615af3420c62f855c5a290d42448cd13f8a4cbc3437778c3`.
Raw asynchronous tail rounds are retained, while compared counters use the same
metrics-reported round window in all arms. The C trace prefix must exactly
reproduce the standard accepted counter. See the workspace experiment report
and `frozen-v7.json` for inputs, scripts, hashes and A-to-A variation.

The independent FP16 Markov head now has its own frequent-token bias cache.
`VLLM_TENSORFOLD_BIAS_CACHE_ROWS` defaults to 256; set 0 to use the reference
FP16 matrix operation. `VLLM_TENSORFOLD_CACHE_IDS` supplies the JSON token list
(default `/opt/ab/assets/markov_top_tokens.json`). Cached and uncached rows use
the same FP32 reduction and FP16 output. This reduction may differ by FP16 ulps
from the reference GEMM; target verification remains authoritative. Confidence
is evaluated in one batch, feedback is read back together, and suffix-copy
hits maintain draft context KV without running the unused neural query head.
These changes apply only to `tensorfold_full`; native remains unchanged.
TensorFold's native matrix/attention kernels are not copied.
Do not describe this as a production-equivalent TensorFold execution engine.

A 2026-11-22 follow-up compared the earlier `tensorfold_full` implementation
with the optimized full path on both TP2 clusters. Formal runs disabled custom
diagnostics, profiler and trace; each 1024/8192-input cell used the same five
ShareGPT requests twice, 1024 output tokens, C1/H0/T0, and fresh A-B-A controls.
Aggregate decode throughput increased by 4.91%/4.54% on intellif-96/164 and
2.22%/5.26% on intellif-74/15 relative to the paired control durations. The
74/15 1K prompt-group interval includes zero; the other three are positive.
Stage costs were sampled separately. Native stayed unchanged and remains the
default. Source locks, raw data and the exact scope are in the surrounding
workspace's `benchmarks/dspark-decode-opt-20261122/report-v2.md`.

The next experimental independent-lane candidate fuses FP16 Markov bias and
greedy argmax, preserving the materialized head's token IDs and first-index
ties. It also corrects conditional confidence using request-local residuals,
shrunk by 32 neutral observations and capped at 0.2 probability. Only reached
head positions are observed; copied and rejected descendants are excluded.
This calibrated allocator extends the pinned TensorFold policy; the original
`LaneRule` remains the source-oracle reference. Existing copy lanes keep their
KV-only shortcut. Ordinary neural rounds enqueue their draft pass before host
feedback and read committed IDs, drafts and confidence together. Native DSpark
remains unchanged. The candidate benchmark explicitly uses 1024 bias-cache
rows (about 199 MB/rank above 256), with formal trace/diagnostics disabled and
separate correctness and stage runs. See `benchmarks/dspark-throughput-opt-20261122`
in the surrounding workspace for source locks and measured acceptance results.

The candidate completed fresh native-A1 / optimized-B / native-A2 on both
TP2 clusters with the same checkpoint and frozen ShareGPT prompts, 1024/8192
input, 1024 output, C1/H0/T0. Formal services had no custom diagnostics,
profiler or trace. Before formal measurements, 8K was prespecified at N20
(five prompts repeated four times); 1K stayed at N10 and separate stages N5.
Paired aggregate decode gains were 2.71%/4.08% on 96/164 and 4.09%/7.50%
on 74/15, with all prompt-group intervals positive. The 96/164 8K interval
was [2.95%, 5.10%] and B exceeded both native controls. Only 14/20 individual
8K pairs were faster, so this is a bounded aggregate result, not a guarantee
of per-request superiority. Output per round still trails native; reduced
per-round cost offsets the extra rounds. Originals were restored, exact
container configuration audits and real completions passed, and native source
hashes stayed unchanged. Reports, source locks and raw evidence are under
`benchmarks/dspark-throughput-opt-20261122` in the surrounding workspace.

### Experimental independent multi-request lanes

The independent path now maintains request-keyed histories, candidates and
confidence calibration for concurrent requests. A shared allocator reserves
prompt/copy rows before admitting head prefixes. Its pinned cost table ends at
24 rows; larger windows use the last marginal cost as an extrapolated prior,
not a measurement on the deployed device. The C1 fast path is retained.
Multi-request proposals use batched host feedback and leader broadcasts of
request-keyed grants and candidates. Prefill slices use each request's query
offset; unfinished partial prefills carry no valid draft candidates.

Real TP2 GB10 checks exercised C2/C4/C8 with mixed 1K/8K inputs, different
termination lengths, cancellation and slot reuse. Target-prefix acceptance,
KV increments, calibration replay and both TP replicas matched exactly across
2,462 request windows on one cluster and 2,411 on the other. Joint allocation
replay covered 636/588 batches; prompt/committed-history replay reproduced
164/171 suffix-copy windows. This validates the tested greedy serving path,
not arbitrary sampling settings, model families or deployment topologies.

`VLLM_TENSORFOLD_TRACE_ALL_RANKS=1` enables replica traces when
`VLLM_TENSORFOLD_TRACE` is set. Use a separate output directory on each rank.
Unset the trace path and diagnostics for speed measurements. Native DSpark
remains the default and its implementation is unchanged by this extension.
Evidence and the frozen dual-cluster C2/C4/C8 native A1 / independent B /
native A2 protocol are under `benchmarks/dspark-multilane-20261122` in the
surrounding workspace. The tested deployment permits eight sequences and has
a configured 524,288-token context limit; testing at 1K/8K input does not
establish eight simultaneous full-length contexts.

The extension completed same-image, same-weight native A1 / independent B /
native A2 on both clusters: 840 measured requests, five synchronous cohorts per
cell, C2/C4/C8, 1024/8192 input and 1024 output tokens, temperature 0 and zero
prefix hits. Formal diagnostics, trace and profiler were disabled. All B point
estimates exceeded both native controls. Decode gains on 96/164 were
10.06%/3.96% (C2), 11.60%/11.36% (C4), and 57.05%/52.24% (C8), for 1K/8K
respectively; 74/15 gains were 11.96%/3.83%, 16.13%/12.20%, and 68.43%/48.75%.
Eleven paired cohort intervals were positive. The 96/164 C2/8K interval
[-0.03%, 7.87%] includes zero and does not establish a stable gain.

These results retain the common maximum eight-token graph-capture limit and
include cohort tails at lower concurrency. They are not fully tuned,
continuously saturated multi-request results. Standard acceptance and tokens
per request draft-window decreased in B despite its speed gains. Visible JIT
warnings affected some C4/1K cohorts; primary results retain all samples and
paired sensitivity results are reported separately. The initial short-warmup
overlap assertion was corrected before any B results; all initial A1 samples
were excluded and the complete comparison restarted. Original deployments
were restored and real API completions passed on both clusters.

The validated graph-coverage candidate searches a common verification depth for
C2--C4 pure-decode batches, using the sum of request-local calibrated survival
probabilities divided by the pinned forward-cost prior plus observed overhead.
This constrains allocation to a captured uniform query shape. It can admit
fewer useful drafts than independent per-request allocation; graph coverage
alone is not evidence of higher decode speed. C5--C8 retain the original joint
per-request allocator. Native DSpark's strategy and deployed eight-token graph
configuration are retained; only the independent candidate expands capture
buckets to 1--24 tokens. Larger or nonuniform target windows fall back to eager.

The independent draft uses captured fixed-five-row query shapes for
up to four requests; larger draft batches currently use eager
execution. This bound does not restrict target graphs or serving concurrency.
Mixed-prefill batches use an eager draft manager so their entire
context KV insertion and query pass stay together. An expanded-graph pilot on
74/15 stalled after target completion in a mixed-prefill draft call; the exact
kernel cause is not established. A weight-free two-rank RDMA/NCCL graph probe
passed 24--48-row reductions at both 1 MB and 2 MB routing thresholds, so that
threshold alone has not reproduced the model stall. Candidate source,
rejected-attempt evidence and the frozen continuous protocol are under
`benchmarks/dspark-graph-steady-20261122` in the surrounding workspace.

The final candidate passed 339 CPU checks and dual-cluster request-state/KV/TP
replay checks. The continuous-load A-B-A completed 1,848 formal requests with
1,176 predefined steady decode requests. All twelve C2/C4/C8 and 1K/8K cells
exceeded both native controls under the prespecified paired-interval criterion.
On 96/164 C2/8K, native A1 / independent B / native A2 decode was
29.74 / 35.01 / 31.19 tokens/s; the paired 95% improvement intervals versus
A1 and A2 were [15.27,20.31]% and [10.44,14.14]%. Pooled-native-to-B output
throughput was 52.91 to 59.77 tokens/s. Budget acceptance and per-request-window
output decreased from 31.61% to 30.85% and 2.58 to 2.54 respectively; these
results do not attribute the whole speedup to CUDA graph alone.

A native A1 attempt triggered a previously unwarmed DFlash input tile JIT and
was excluded in full. Startup now numerically checks scratch input preparation
for tiles 8/16/32/64/128/256, query widths 1--5 and blocks 32/64. The whole A-B-A
was restarted after resealing helpers. All four ranks passed strict monitored
JIT/autotune audits; formal diagnostics and traces were disabled, and original
deployments were restored with real API completions.

This evidence is bounded to the frozen five prompts and deployed checkpoint.
None of the 616 complete streamed-text triples matched across arms, including
native A1 versus native A2. The benchmark therefore compares identical requests
and output token counts, not token-identical generations. State-contract replay
does not establish cross-path token identity. Wider contexts, other checkpoints
and long-term repeatability remain outside this validation.

The retained standalone image is
`local/vllm-dsv41:dspark-tensorfold-graph24-v1-20261123`, with image ID
`sha256:e8ec98062462d93222877c36ca21852d13bdfb87a1dc36384712712a288ac68e`.
It embeds the independent source, startup prewarm and frequent-token asset;
only model weights and compiler cache are mounted by the serving containers.
It is running on 96/164 with the original native containers stopped and
preserved for rollback. The 74/15 original deployment was not switched.
Startup acceptance passed 58 long warmup requests and 56 strict-canary
requests across all six C2/C4/C8 and 1K/8K cells, with zero monitored JIT or
autotuning events during canary on both ranks. Serving returns to warning
mode for new runtime compilation keys after the strict canary.

The [image example](../../examples/deployment/dspark_tensorfold/README.md)
retains the release fingerprints, prewarm helpers and a build-context generator.
Source rebuilds use a separate tag and fresh fingerprints; the complete
retained archive, rather than a rebuilt image, restores the validated image ID.
