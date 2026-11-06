#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${RUN_ROOT:?absolute stage run root required}
CONTAINER=${CONTAINER:?candidate container name required}
HOST_PYTHON=${HOST_PYTHON:-$(dirname "$RUN_ROOT")/.venv/bin/python}
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DONE_MARKER=${DONE_MARKER:-model-validation.exit}
MODEL_RECEIPT_NAME=${MODEL_RECEIPT_NAME:-model-api.json}
[[ $RUN_ROOT == /* && $DONE_MARKER =~ ^[a-zA-Z0-9._-]+$ ]]
[[ $MODEL_RECEIPT_NAME =~ ^[a-zA-Z0-9._-]+$ ]]
mkdir -p "$RUN_ROOT/manifests"
trap 'result=$?; printf "%s\n" "$result" > "$RUN_ROOT/$DONE_MARKER.tmp"; mv "$RUN_ROOT/$DONE_MARKER.tmp" "$RUN_ROOT/$DONE_MARKER"' EXIT
"$HOST_PYTHON" "$SCRIPT_ROOT/verify-model-api.py" --container "$CONTAINER" \
    --output "$RUN_ROOT/manifests/$MODEL_RECEIPT_NAME"
comparison=()
if [[ -n ${REFERENCE_BENCHMARK:-} ]]; then
    comparison+=(--reference "$REFERENCE_BENCHMARK")
fi
"$HOST_PYTHON" "$SCRIPT_ROOT/benchmark-model-api.py" --container "$CONTAINER" \
    --repeats "${MODEL_REPEATS:-5}" "${comparison[@]}" \
    --output "$RUN_ROOT/manifests/model-benchmark.json"
if [[ -n ${REFERENCE_BENCHMARK:-} ]]; then
    jq -e '.threePercentGatePassed == true' \
        "$RUN_ROOT/manifests/model-benchmark.json" >/dev/null
fi
