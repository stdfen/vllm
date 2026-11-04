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
ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-1}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-deepseek_v41}
REASONING_PARSER=${REASONING_PARSER:-deepseek_v41}
TP_PEER_WATCHDOG=${TP_PEER_WATCHDOG:-1}
TP_PEER_WATCHDOG_INTERVAL=${TP_PEER_WATCHDOG_INTERVAL:-2}
TP_PEER_WATCHDOG_FAILURES=${TP_PEER_WATCHDOG_FAILURES:-5}
AUTHOR_OPTIMIZATIONS=${AUTHOR_OPTIMIZATIONS:-1}

export VLLM_PINNED_WEIGHT_STAGING=${VLLM_PINNED_WEIGHT_STAGING:-auto}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP7s7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enP7s7}

if [[ "${AUTHOR_OPTIMIZATIONS}" == "1" ]]; then
  export VLLM_SHORT_CTX_BOUNDS=${VLLM_SHORT_CTX_BOUNDS:-512,1025}
  export VLLM_SHORT_CTX_MARGIN=${VLLM_SHORT_CTX_MARGIN:-16}
  export VLLM_ENGRAM_WKV_ROWPAR=${VLLM_ENGRAM_WKV_ROWPAR:-1}
  export EXL3_L2PF=${EXL3_L2PF:-attn,moe,next}
  export EXL3_L2PF_ORDER=${EXL3_L2PF_ORDER:-1}
  export EXL3_L2PF_DRAFTER=${EXL3_L2PF_DRAFTER:-1}
  export EXL3_L2PF_HC=${EXL3_L2PF_HC:-1}
elif [[ "${AUTHOR_OPTIMIZATIONS}" != "0" ]]; then
  echo "unsupported AUTHOR_OPTIMIZATIONS=${AUTHOR_OPTIMIZATIONS}; expected 0 or 1" >&2
  exit 2
fi

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

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  extra_args+=(
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --reasoning-parser "${REASONING_PARSER}"
  )
elif [[ "${ENABLE_AUTO_TOOL_CHOICE}" != "0" ]]; then
  echo "unsupported ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}; expected 0 or 1" >&2
  exit 2
fi

# Only rank 0 owns the API server and EngineCore client.  Secondary nodes in
# vLLM's multi-node MP executor must run the follower executor in headless
# mode; otherwise they incorrectly initialize a second EngineCore and attempt
# driver-only KV-cache RPCs after loading the model.
if [[ "${NODE_RANK}" != "0" ]]; then
  extra_args+=(--headless)
fi

vllm serve "${MODEL_DIR}" \
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
  "${extra_args[@]}" &

serve_pid=$!
watchdog_pid=

watch_master() {
  local seen=0 failures=0
  while kill -0 "${serve_pid}" 2>/dev/null; do
    if timeout 1 bash -c "</dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
      seen=1
      failures=0
    elif [[ "${seen}" == "1" ]]; then
      failures=$((failures + 1))
      if (( failures >= TP_PEER_WATCHDOG_FAILURES )); then
        echo "rank 0 rendezvous disappeared; terminating rank ${NODE_RANK}" >&2
        kill -TERM "${serve_pid}" 2>/dev/null || true
        return
      fi
    fi
    sleep "${TP_PEER_WATCHDOG_INTERVAL}"
  done
}

if [[ "${NODE_RANK}" != "0" && "${TP_PEER_WATCHDOG}" == "1" ]]; then
  watch_master &
  watchdog_pid=$!
fi

terminate_serve() {
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
  fi
  if kill -0 "${serve_pid}" 2>/dev/null; then
    kill -TERM "${serve_pid}" 2>/dev/null || true
    wait "${serve_pid}" || true
  fi
}
trap terminate_serve TERM INT EXIT
if wait "${serve_pid}"; then
  serve_status=0
else
  serve_status=$?
fi
if [[ -n "${watchdog_pid}" ]]; then
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
fi
trap - TERM INT EXIT
exit "${serve_status}"
