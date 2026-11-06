#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$(git rev-parse --show-toplevel)
RUN_ROOT=${RUN_ROOT:?new stage official-build run required}
REFERENCE_ROOT=${REFERENCE_ROOT:?validated S1 reference run required}
BUILD_STAGE=${BUILD_STAGE:?s2 or s3 required}
BUILDER=${BUILDER:-vllm-source-20261105}
[[ $BUILD_STAGE == s2 || $BUILD_STAGE == s3 ]]
[[ $RUN_ROOT == /* && $REFERENCE_ROOT == /* && $RUN_ROOT != "$REFERENCE_ROOT" ]]
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
trap 'result=$?; printf "%s\n" "$result" > "$RUN_ROOT/final-reuse.exit"' EXIT
[[ $(cat "$RUN_ROOT/s1.exit") == 0 ]]
[[ $(cat "$REFERENCE_ROOT/s1-reference.exit") == 0 ]]
[[ -f $RUN_ROOT/official.oci/index.json ]]
jq -e --arg config "$(jq -er '.[0].Id' "$RUN_ROOT/manifests/s1-image.json")" \
    --arg index "$(sha256sum "$RUN_ROOT/official.oci/index.json" | cut -d' ' -f1)" \
    '.status == "passed" and .configDigest == $config and .indexSha256 == $index
    and .allDiffIdsVerified == true' "$RUN_ROOT/manifests/official-oci-validation.json" >/dev/null
reference_image_id=$(jq -er '.[0].Id' "$REFERENCE_ROOT/manifests/s1-full-image.json")
jq -e --arg image "$reference_image_id" \
    '.status == "passed" and .usage.completion_tokens > 0 and .imageId == $image' \
    "$REFERENCE_ROOT/manifests/model-api.json" >/dev/null
sha256sum -c "$REFERENCE_ROOT/manifests/exllama-wheels.sha256"
CONTEXT="$RUN_ROOT/reuse-context"
mkdir -p "$CONTEXT/wheels"
cp "$REFERENCE_ROOT/exllama-wheels/"*.whl "$CONTEXT/wheels/"
(cd "$REFERENCE_ROOT/final-context/runtime-wheels" && sha256sum -c SHA256SUMS)
jq -er '.exllamav3.runtimeWheels[] | "\(.sha256)  \(.filename)"' \
    docker/exl3/source-build/sources.lock.json \
    | (cd "$REFERENCE_ROOT/final-context/runtime-wheels" && sha256sum -c -)
cmp <(jq -S '.exllamav3' "$REFERENCE_ROOT/final-context/sources.lock.json") \
    <(jq -S '.exllamav3' docker/exl3/source-build/sources.lock.json)
cp "$REFERENCE_ROOT/final-context/runtime-wheels/"*.whl "$CONTEXT/wheels/"
cp "$REFERENCE_ROOT/final-context/vllm_exl3_moe-0.1.0.dev0-py3-none-any.whl" "$CONTEXT/wheels/"
(cd "$CONTEXT/wheels" && sha256sum -c "$SOURCE_ROOT/docker/exl3/source-build/SHA256SUMS.plugin")
(cd "$CONTEXT/wheels" && sha256sum ./*.whl > SHA256SUMS)
cp docker/exl3/source-build/Dockerfile.final-reuse "$CONTEXT/Dockerfile"
cp docker/exl3/source-build/verify-image.py docker/exl3/source-build/smoke-kernels.py "$CONTEXT/"
cp docker/exl3/source-build/sources.lock.json "$CONTEXT/"
cp docker/exl3/run_tp2.sh tools/exl3/tp2_preflight.py tools/exl3/manifest.py "$CONTEXT/"
find "$CONTEXT" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_ROOT/manifests/reuse-context.sha256"
sha256sum "$RUN_ROOT/manifests/reuse-context.sha256" "$RUN_ROOT/official.oci/index.json" \
    > "$RUN_ROOT/manifests/reuse-inputs.sha256"
INPUTS_SHA=$(sha256sum "$RUN_ROOT/manifests/reuse-inputs.sha256" | cut -d' ' -f1)
TAG="vllm/vllm-openai:v0.32.0-dsv41-${BUILD_STAGE}-$(git rev-parse --short=10 HEAD)-${INPUTS_SHA:0:10}"
printf '%s\n' "$TAG" > "$RUN_ROOT/manifests/final-tag.txt"
docker buildx build --builder "$BUILDER" --progress=plain --platform linux/arm64 \
    --network none --load --target final-reuse \
    --build-context "official=oci-layout://$RUN_ROOT/official.oci" \
    --build-arg OFFICIAL_IMAGE=official --build-arg "SOURCE_SHA=$(git rev-parse HEAD)" \
    --build-arg "INPUTS_SHA=$INPUTS_SHA" --build-arg "BUILD_STAGE=$BUILD_STAGE" \
    --tag "$TAG" "$CONTEXT" 2>&1 | tee "$RUN_ROOT/logs/final-reuse.log"
docker run --rm --gpus all --network none \
    --entrypoint /opt/vllm-exl3/.venv/bin/python "$TAG" \
    /opt/vllm-exl3/verify-image.py --gpu | tee "$RUN_ROOT/manifests/image-check.json"
kernel_cache=$(mktemp -d "$RUN_ROOT/kernel-smoke-cache.XXXXXX")
docker run --rm --gpus all --network none \
    --mount "type=bind,src=$kernel_cache,dst=/var/cache/kernel-smoke" \
    -e FLASHINFER_NO_DOWNLOAD=1 -e CUDA_CACHE_PATH=/var/cache/kernel-smoke/cuda \
    -e TRITON_CACHE_DIR=/var/cache/kernel-smoke/triton \
    -e FLASHINFER_WORKSPACE_BASE=/var/cache/kernel-smoke/flashinfer \
    -e XDG_CACHE_HOME=/var/cache/kernel-smoke/xdg -e MAX_JOBS=2 \
    --entrypoint /opt/vllm-exl3/.venv/bin/python "$TAG" \
    /opt/vllm-exl3/smoke-kernels.py | tee "$RUN_ROOT/manifests/kernel-smoke.json"
docker image inspect "$TAG" > "$RUN_ROOT/manifests/final-image.json"
docker history --no-trunc "$TAG" > "$RUN_ROOT/manifests/final-history.txt"
