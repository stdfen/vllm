#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$(git rev-parse --show-toplevel)
RUN_ROOT=${RUN_ROOT:?absolute run directory required}
BUILDER=${BUILDER:-vllm-source-20261105}
SOURCE_ARCHIVE=${SOURCE_ARCHIVE:?pinned ExLlamaV3 source archive required}
LOCK_FILE="$SOURCE_ROOT/docker/exl3/source-build/sources.lock.json"
ARCHIVE_SHA=$(jq -er .exllamav3.sourceArchiveSha256 "$LOCK_FILE")
EXLLAMAV3_REVISION=$(jq -er .exllamav3.revision "$LOCK_FILE")
[[ $(cat "$RUN_ROOT/s1.exit") == 0 ]]
OFFICIAL_IMAGE=$(cat "$RUN_ROOT/manifests/s1-tag.txt")
if [[ ! -f "$RUN_ROOT/official.oci/index.json" ]]; then
    EXPORT_OCI=1 MIN_DISK_GIB=100 bash docker/exl3/source-build/build-official-source.sh
fi
jq -e --arg config "$(jq -er '.[0].Id' "$RUN_ROOT/manifests/s1-image.json")" \
    --arg index "$(sha256sum "$RUN_ROOT/official.oci/index.json" | cut -d' ' -f1)" \
    '.status == "passed" and .configDigest == $config and .indexSha256 == $index
    and .allDiffIdsVerified == true' "$RUN_ROOT/manifests/official-oci-validation.json" >/dev/null
CONTEXT="$RUN_ROOT/final-context"
RUNTIME_WHEEL_INPUT=${RUNTIME_WHEEL_INPUT:-$(dirname "$RUN_ROOT")/exllama-runtime-wheels}
mkdir -p "$CONTEXT/runtime-wheels"
while IFS=$'\t' read -r filename checksum; do
    [[ $filename != */* && $(sha256sum "$RUNTIME_WHEEL_INPUT/$filename" | cut -d' ' -f1) == "$checksum" ]]
    cp "$RUNTIME_WHEEL_INPUT/$filename" "$CONTEXT/runtime-wheels/"
done < <(jq -er '.exllamav3.runtimeWheels[] | [.filename,.sha256] | @tsv' "$LOCK_FILE")
jq -er '.exllamav3.runtimeWheels[] | "\(.sha256)  \(.filename)"' "$LOCK_FILE" \
    > "$CONTEXT/runtime-wheels/SHA256SUMS"
[[ $(sha256sum "$SOURCE_ARCHIVE" | cut -d' ' -f1) == "$ARCHIVE_SHA" ]]
cp "$SOURCE_ARCHIVE" "$CONTEXT/exllamav3-source.tar.gz"
cp docker/exl3/source-build/SHA256SUMS.plugin "$CONTEXT/SHA256SUMS"
(cd "$CONTEXT" && sha256sum -c SHA256SUMS)
cp docker/exl3/source-build/Dockerfile.final-full "$CONTEXT/Dockerfile"
cp docker/exl3/source-build/verify-image.py "$CONTEXT/"
cp "$LOCK_FILE" "$CONTEXT/sources.lock.json"
cp docker/exl3/source-build/smoke-kernels.py "$CONTEXT/"
cp docker/exl3/prepare_exllamav3_aarch64.py docker/exl3/exllamav3_aarch64_cuda_only.cpp \
    docker/exl3/run_tp2.sh tools/exl3/tp2_preflight.py tools/exl3/manifest.py "$CONTEXT/"
find "$CONTEXT" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_ROOT/manifests/final-context.sha256"
sha256sum "$RUN_ROOT/manifests/final-context.sha256" "$RUN_ROOT/official.oci/index.json" \
    > "$RUN_ROOT/manifests/final-inputs.sha256"
INPUTS_SHA=$(sha256sum "$RUN_ROOT/manifests/final-inputs.sha256" | cut -d' ' -f1)
TAG="vllm/vllm-openai:v0.32.0-dsv41-s1-full-$(git rev-parse --short=10 HEAD)-${INPUTS_SHA:0:10}"
printf '%s\n' "$TAG" > "$RUN_ROOT/manifests/s1-full-tag.txt"
build_args=(
    --builder "$BUILDER" --progress=plain --platform linux/arm64
    --network host --allow network.host
    --build-context "official=oci-layout://${RUN_ROOT}/official.oci"
    --build-arg OFFICIAL_IMAGE=official
    --build-arg "SOURCE_SHA=$(git rev-parse HEAD)"
    --build-arg "EXLLAMAV3_REVISION=$EXLLAMAV3_REVISION"
    --build-arg "INPUTS_SHA=$INPUTS_SHA"
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL:-}"
    --build-arg "UV_INDEX_URL=${UV_INDEX_URL:-${PIP_INDEX_URL:-}}"
    --build-arg "UV_INDEX=${UV_INDEX:-}"
    --build-arg "HTTP_PROXY=${HTTP_PROXY:-}"
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-}"
)
docker buildx build "${build_args[@]}" --load --target final-full \
    --tag "$TAG" "$CONTEXT" 2>&1 | tee "$RUN_ROOT/logs/s1-final-full.log"

# Reuse only this source-built wheel in later packaging stages. The export
# target hits the same compiler cache and contains no base-image files.
docker buildx build "${build_args[@]}" --target exllama-wheel-export \
    --output "type=local,dest=$RUN_ROOT/exllama-wheels" "$CONTEXT" \
    2>&1 | tee "$RUN_ROOT/logs/exllama-wheel-export.log"
sha256sum "$RUN_ROOT/exllama-wheels/"*.whl \
    > "$RUN_ROOT/manifests/exllama-wheels.sha256"
docker run --rm --gpus all --network none \
    --entrypoint /opt/vllm-exl3/.venv/bin/python "$TAG" \
    /opt/vllm-exl3/verify-image.py --gpu | tee "$RUN_ROOT/manifests/s1-full-check.json"
kernel_cache=$(mktemp -d "$RUN_ROOT/kernel-smoke-cache.XXXXXX")
docker run --rm --gpus all --network none \
    --mount "type=bind,src=$kernel_cache,dst=/var/cache/kernel-smoke" \
    -e FLASHINFER_NO_DOWNLOAD=1 -e CUDA_CACHE_PATH=/var/cache/kernel-smoke/cuda \
    -e TRITON_CACHE_DIR=/var/cache/kernel-smoke/triton \
    -e FLASHINFER_WORKSPACE_BASE=/var/cache/kernel-smoke/flashinfer \
    -e XDG_CACHE_HOME=/var/cache/kernel-smoke/xdg -e MAX_JOBS=2 \
    --entrypoint /opt/vllm-exl3/.venv/bin/python "$TAG" \
    /opt/vllm-exl3/smoke-kernels.py | tee "$RUN_ROOT/manifests/s1-kernel-smoke.json"
docker image inspect "$TAG" > "$RUN_ROOT/manifests/s1-full-image.json"
docker history --no-trunc "$TAG" > "$RUN_ROOT/manifests/s1-full-history.txt"
