#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?NODE_RANK must be 0 or 1}"
: "${MASTER_ADDR:?MASTER_ADDR must identify the rank-0 node}"

MODEL_DIR=${MODEL_DIR:-/model}
ENGRAM_DIR=${ENGRAM_DIR:-/engram-src}
MASTER_PORT=${MASTER_PORT:-29501}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MIN_CUDA_FREE_GIB=${MIN_CUDA_FREE_GIB:-110}
API_PORT=${API_PORT:-8000}
KV_BLOCK_SIZE=${KV_BLOCK_SIZE:-64}
SPEC_METHOD=${SPEC_METHOD:-none}
DSPARK_TOKENS=${DSPARK_TOKENS:-3}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-"1 2 3 4 6 8 12 18 24"}

export VLLM_PINNED_WEIGHT_STAGING=${VLLM_PINNED_WEIGHT_STAGING:-auto}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP7s7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enP7s7}

python3 /opt/vllm-exl3/tp2_preflight.py \
  --model "${MODEL_DIR}" \
  --engram-model "${ENGRAM_DIR}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --min-cuda-free-gib "${MIN_CUDA_FREE_GIB}"

extra_args=()
if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then
  extra_args+=(--enforce-eager)
else
  # Keep capture bounded to the author-validated small decode batches.  The
  # default inferred list can reserve materially more unified memory on GB10.
  read -r -a capture_sizes <<< "${CUDAGRAPH_CAPTURE_SIZES}"
  extra_args+=(--cudagraph-capture-sizes "${capture_sizes[@]}")
fi

if [[ "${SPEC_METHOD}" == "dspark" ]]; then
  # DeepSeek V4.1 embeds its DSpark drafter in the checkpoint.  k=3 is the
  # author's measured throughput sweet spot; dspark_block_size=5 is only the
  # checkpoint ceiling and should not be copied as the runtime setting.
  extra_args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_TOKENS}}")
elif [[ "${SPEC_METHOD}" != "none" ]]; then
  echo "unsupported SPEC_METHOD=${SPEC_METHOD}; expected none or dspark" >&2
  exit 2
fi

if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  extra_args+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

# Only rank 0 owns the API server and EngineCore client.  Secondary nodes in
# vLLM's multi-node MP executor must run the follower executor in headless
# mode; otherwise they incorrectly initialize a second EngineCore and attempt
# driver-only KV-cache RPCs after loading the model.
if [[ "${NODE_RANK}" != "0" ]]; then
  extra_args+=(--headless)
fi

exec vllm serve "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --trust-remote-code \
  --quantization exl3 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --distributed-executor-backend mp \
  --engram-config "{\"file_backed\":true,\"table_dir\":\"${ENGRAM_DIR}\",\"cpu_offload\":true}" \
  --block-size "${KV_BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --no-enable-flashinfer-autotune \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${extra_args[@]}"
