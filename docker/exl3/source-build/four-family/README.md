# Four-family implementation ledger

Status: draft, NOT an acceptance certificate.

- Baseline: 4116682cb6833d0996bc9f74fc707b3540438197.
- A1: 50b2109fd78e524b61a4656e62db739f453b5957.
- retained-models.json records 101 mapping rows (including shared backends and
  helpers), not 101 unique checkpoint architectures. Null fields are unresolved,
  not unrestricted approval. Product-review entries remain retained.
- removed-entries.json records 298 removed mappings. Runtime rejection is pending
  A2; registration removal alone is not an admission boundary.
- removal-decisions.csv seeds dependency review. No file is approved for removal
  solely because it was a static candidate.
- The inspected model/cache roots on both serving nodes contained DeepSeek
  configurations only. Other mounts and external model stores are not covered.
- Current build host available memory is below the 96 GiB gate. The peer Docker
  filesystem has about 25 GiB free, below the prior S3 import requirement.

## Build identity

The source builder now requires ALLOWED_SOURCE_BRANCH and EXPECTED_SOURCE_SHA
(full 40-character commit), rejects tracked changes and untracked source files,
and preserves the fixed base-ref check. Use an isolated clean build clone;
worklogs and audit artifacts belong outside it. The build writes
manifests/source-identity.txt. Do not reuse old checkpoint metadata for a new SHA.

Validation status is recorded below; none of it is GPU or image acceptance.

K3DSparkModel was mistakenly removed in A1 and restored during A2 review.
Its Kimi pairing still requires runtime/weight acceptance.

## B1 isolated removal verification (2026-11-09)

The GPT-J, Bloom and Falcon model implementation files are physically removed
in the local working tree. An offline, memory-limited disposable container also
removed all three installed files before running the full registry test file:
261 passed, 2 skipped, 14 warnings. The skips require CUDA/ROCm. The four A2
source overrides were mounted together; the repository root conftest was not
loaded. This verifies retained registry dependency closure only, not weights,
workers, native extensions, wheel contents or model output.

## Current implementation checkpoint (2026-11-09)

The earlier B1 paragraph is historical evidence, not the current removal inventory.
Falcon config/converter aliases and OPT implementation have since been removed;
OPT plugin fixtures now use approved Qwen implementations. Shared falcon_h1,
ALiBi/RoPE, Llama/EAGLE and audio/vision dependencies remain retained.

Dots3Note, HY V4, MiniMax M3 and Inkling model packages, plus Cohere2 MoE,
Llama4 variants, Laguna DFlash and MiMo V2 MTP implementations are removed in
this working tree. Shared routing and SwiGLU code was moved before deletion.
Python/Rust parser and renderer changes have targeted tests; this is not an
assertion that every excluded model implementation has been removed.

Latest unified regression applies the current source overrides and 126 production
file deletions, with the real root conftest. Seven complete files cover model
registry, speculative/config, model architecture configuration, attention backend,
parser replay, tokenizer registry and multimodal registry. The isolated offline
2 GiB container reports 4026 passed, 2 skipped, 27 warnings in 200.09 seconds;
no deselection was used. The skipped cases require CUDA/ROCm.

A separate VLLM_TARGET_DEVICE=empty packaging check produced an ARM64 wheel:
all 2527 Python inputs are present and none of the 126 deleted production paths
are included. It contains no native extensions; optional Rust compilation could
not run without a compiler. This is Python packaging closure, NOT a deployable
CUDA/Rust wheel or candidate OCI acceptance.

Runtime registry tables exactly match the 101 retained manifest mapping rows
(99 unique runtime architecture keys): no missing rows or implementation mismatch.
Checkpoint identity fields remain missing in 97 rows; this equality does not
freeze the support manifest or verify weights, workers, GPU output or deployment.
Evidence is in the workspace audit directory vllm-four-family-slim-20261109.

The support manifest remains draft. Clean-source identity, wheel contents,
ARM64/CUDA S1-S3 builds, throughput gates and four-family/two-node acceptance
remain required. The last resource observation was below the 96 GiB available
RAM build gate; the peer also lacked image-import disk capacity. Baseline S3
services and rollback assets have not been replaced.


## Final-seven evidence correction and remaining gates (2026-11-09)

The newer final-seven JUnit contains 4036 total cases: **4034 passed, 2 skipped**,
0 failures and 0 errors (929.76 seconds). Do not report 4036 as the passed count.
This supersedes the earlier seven-file run above, not its historical evidence.

Static review of `llama_eagle3.py` shows that the shared Qwen-VL EAGLE3
implementation ignores external multimodal embeddings in `embed_input_ids`.
`llm_base_proposer.py` checks `supports_multimodal_embeddings` and falls back to
text-only draft inputs when absent. Configuration construction therefore does
not establish visual-input, weight-loading, acceptance-rate or GPU correctness.
Keep both Qwen-VL EAGLE3 entries pending runtime acceptance; do not add a
multimodal-support flag without implementation and GPU evidence.

A current completeness audit finds no retained/removed mapping-key overlap.
However, unresolved rows remain: tokenizer/processor and parser 101 each;
tasks and allowed backends 96 each; devices, parallelism and quantization 100
each; top-level checkpoint identity 95. Empty alias lists are not proof of
missing aliases and require review, not fabricated values. Captured config
hashes must not be mistaken for tested deployment combinations. The manifest
remains `draft-not-frozen`.

The latest read-only host check still fails the 96 GiB available-memory gate
(about 3.4 GiB available). No candidate S1/S2/S3 build was started and no
production service was stopped. Source identity remains the A1 commit plus
uncommitted implementation changes; this is not a clean build identity.

## Verified candidate identity backfill

79 previously empty top-level checkpoint identity fields were populated from
captured candidate configs after recomputing each local file SHA-256 and matching
repository, revision and digest against the capture index. Six existing identity
fields were left unchanged. Sixteen rows still lack a top-level identity (including
ten generic Transformers backends). These are candidate configuration identities,
not tested weights, approved aliases or deployment acceptance. The earlier count
of 95 empty identity fields is historical; the current count is 16. Evidence:
`manifest-identity-backfill.json` in the workspace audit directory. The manifest
remains draft; no missing runtime capability was inferred from config capture.

## Processor declaration evidence backfill

Three entries now record tokenizer/processor declarations from hash-verified
config.json files. Four entries also carry processor asset metadata, retrieved
from the isolated test directory and SHA-256 checked against the original
repository/revision capture index. This leaves 97 null tokenizer/processor rows.
`auto_map` remains a declaration, not remote-code permission; no remote module
was executed. Class declarations and asset capture do not prove processor
construction, image/video handling or GPU acceptance. Evidence:
`manifest-processor-declarations.json` and
`manifest-processor-assets-backfill.json` in the workspace audit directory.

## Additional leaf implementation pruning (2026-11-10)

75 excluded model implementation modules and 26 model-exclusive test files were
removed after production-import and symbol-reference review. A mixed speculator
test was edited to remove only its deleted-model cases. No test covering a
retained model was removed. The registered four-family implementations and
shared infrastructure remain.

The first 41-module batch passed 4043 tests with 2 skips; the 49-module batch
passed 4084 tests with 2 skips in isolated offline containers. The final
75-module and 26-exclusive-test state passed 4131 tests with 2 skips from a
clean source snapshot (JUnit `leaf-prune-75-final-20261110-junit.xml`), and the
static production-import closure check reported no missing model modules. The
152 held candidates still require reference classification or migration; this is
an incremental change, not completion of all excluded implementations. Runtime
image acceptance for the final source commit is tracked separately, and earlier
S1/49-module images do not represent this state.
