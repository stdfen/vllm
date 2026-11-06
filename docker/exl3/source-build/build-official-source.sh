#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=$(git rev-parse --show-toplevel)
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to an absolute directory outside the source tree}
BUILDER=${BUILDER:-vllm-source-20261105}
BUILD_STAGE=${BUILD_STAGE:-s1}
NATIVE_MAX_JOBS=${NATIVE_MAX_JOBS:-6}
[[ $NATIVE_MAX_JOBS =~ ^[0-9]+$ ]] && (( NATIVE_MAX_JOBS > 0 ))
VLLM_MAX_SIZE_MB=${VLLM_MAX_SIZE_MB:-500}
[[ $VLLM_MAX_SIZE_MB =~ ^[0-9]+$ ]] && (( VLLM_MAX_SIZE_MB > 0 ))
[[ $BUILD_STAGE == s1 || $BUILD_STAGE == s2 || $BUILD_STAGE == s3 ]]
if [[ $BUILD_STAGE != s1 && -z ${CUBIN_POLICY_INPUT:-} ]]; then
    echo 'A validated cubin policy is required for a slim stage' >&2
    exit 1
fi
SOURCE_SHA=$(git rev-parse HEAD)
SOURCE_SHORT=${SOURCE_SHA:0:10}
LOCK_FILE="$SOURCE_ROOT/docker/exl3/source-build/sources.lock.json"
BUILD_BASE=$(jq -er .buildBase "$LOCK_FILE")
RUNTIME_BASE=$(jq -er .runtimeBase "$LOCK_FILE")
extra_context=()
extra_build_args=()
if [[ -n ${CUDA_APT_INPUT:-} ]]; then
    [[ $CUDA_APT_INPUT == /* ]]
    (cd "$CUDA_APT_INPUT" && sha256sum -c SHA256SUMS >/dev/null)
    extra_context+=(--build-context "cuda-apt-input=$CUDA_APT_INPUT")
fi
if [[ $BUILD_STAGE == s3 ]]; then
    : "${CUDA_RUNTIME_POLICY_INPUT:?S3 requires reviewed CUDA manifests}"
    : "${CUDA_STATIC_POLICY_SHA256:?S3 requires static manifest hash}"
    : "${CUDA_DEDUPE_POLICY_SHA256:?S3 requires dedupe manifest hash}"
fi
if [[ -n ${CUDA_RUNTIME_POLICY_INPUT:-} ]]; then
    [[ $BUILD_STAGE == s3 && $CUDA_RUNTIME_POLICY_INPUT == /* ]]
    [[ $(sha256sum "$CUDA_RUNTIME_POLICY_INPUT/static.json" | cut -d' ' -f1) == "$CUDA_STATIC_POLICY_SHA256" ]]
    [[ $(sha256sum "$CUDA_RUNTIME_POLICY_INPUT/dedupe.json" | cut -d' ' -f1) == "$CUDA_DEDUPE_POLICY_SHA256" ]]
    extra_context+=(--build-context "cuda-runtime-policy-input=$CUDA_RUNTIME_POLICY_INPUT")
elif [[ -n ${CUDA_STATIC_POLICY_SHA256:-}${CUDA_DEDUPE_POLICY_SHA256:-} ]]; then
    echo 'CUDA_RUNTIME_POLICY_INPUT is required with CUDA policy hashes' >&2
    exit 1
fi
if [[ -n ${NATIVE_BUILD_INPUT:-} ]]; then
    [[ $NATIVE_BUILD_INPUT == /* ]]
    (cd "$NATIVE_BUILD_INPUT" && sha256sum -c SHA256SUMS >/dev/null)
    native_source_files=(CMakeLists.txt cmake csrc setup.py pyproject.toml vllm/envs.py
        vllm/__init__.py tools/setup_deepgemm_pythons.sh tools/build_deepgemm_C.py)
    current_native_inputs=$(git ls-files -z -- "${native_source_files[@]}" | sort -z \
        | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
    jq -e --arg source "$SOURCE_SHA" --arg inputs "$current_native_inputs" --arg base "$BUILD_BASE" \
        '.sourceRevision == $source and .nativeSourcesSha256 == $inputs and .buildBase == $base
        and .cudaVersion == "13.0.3" and .nvccThreads == 2
        and .torchCudaArchList == "12.0" and .buildType == "RelWithDebInfo"' \
        "$NATIVE_BUILD_INPUT/state.json" >/dev/null
    extra_context+=(--build-context "native-build-input=$NATIVE_BUILD_INPUT")
fi
if [[ -n ${CXX_SOURCES:-} ]]; then
    [[ $CXX_SOURCES == /* ]]
    expected_cutlass=$(jq -er .cutlass.revision "$LOCK_FILE")
    [[ $(git -C "$CXX_SOURCES/cutlass" rev-parse HEAD) == "$expected_cutlass" ]]
    [[ -z $(git -C "$CXX_SOURCES/cutlass" status --porcelain --untracked-files=all --ignored) ]]
    git -C "$CXX_SOURCES/cutlass" fsck --no-reflogs >/dev/null
    extra_context+=(--build-context "cxx-sources=$CXX_SOURCES")
    extra_build_args+=(--build-arg VLLM_CUTLASS_SRC_DIR=/opt/vendor-src/cutlass)
fi
if [[ -n ${CUBIN_POLICY_INPUT:-} ]]; then
    [[ $CUBIN_POLICY_INPUT == /* && -n ${CUBIN_POLICY_SHA256:-} ]]
    [[ $(sha256sum "$CUBIN_POLICY_INPUT/flashinfer-cubins.json" | cut -d' ' -f1) == "$CUBIN_POLICY_SHA256" ]]
    extra_context+=(--build-context "cubin-policy-input=$CUBIN_POLICY_INPUT")
elif [[ -n ${CUBIN_POLICY_SHA256:-} ]]; then
    echo 'CUBIN_POLICY_INPUT is required with CUBIN_POLICY_SHA256' >&2
    exit 1
fi
if [[ -n ${RUST_TOOLCHAIN_INPUT:-} ]]; then
    [[ $RUST_TOOLCHAIN_INPUT == /* ]]
    (cd "$RUST_TOOLCHAIN_INPUT" && sha256sum -c SHA256SUMS >/dev/null)
    extra_context+=(--build-context "rust-toolchain-input=$RUST_TOOLCHAIN_INPUT")
fi
if [[ -n ${FLASHINFER_WHEELHOUSE:-} ]]; then
    [[ $FLASHINFER_WHEELHOUSE == /* ]]
    (cd "$FLASHINFER_WHEELHOUSE" && sha256sum -c SHA256SUMS >/dev/null)
    jq -er '.flashinfer.artifacts[] | "\(.sha256)  \(.filename)"' "$LOCK_FILE" \
        | (cd "$FLASHINFER_WHEELHOUSE" && sha256sum -c - >/dev/null)
    extra_context+=(--build-context "flashinfer-wheels=$FLASHINFER_WHEELHOUSE")
fi
if [[ -n ${DEPENDENCY_WHEELHOUSE:-} ]]; then
    [[ $DEPENDENCY_WHEELHOUSE == /* ]]
    (cd "$DEPENDENCY_WHEELHOUSE" && sha256sum -c SHA256SUMS >/dev/null)
    extra_context+=(--build-context "dependency-wheels=$DEPENDENCY_WHEELHOUSE")
fi
if [[ -n ${OFFLINE_BUILD_BASE_OCI:-} ]]; then
    expected_digest=${BUILD_BASE##*@}
    [[ $expected_digest == sha256:* ]]
    jq -e --arg digest "$expected_digest" \
        '.manifests | any(.digest == $digest)' \
        "$OFFLINE_BUILD_BASE_OCI/index.json" >/dev/null
    manifest_blob="$OFFLINE_BUILD_BASE_OCI/blobs/sha256/${expected_digest#sha256:}"
    [[ $(sha256sum "$manifest_blob" | cut -d' ' -f1) == "${expected_digest#sha256:}" ]]
    extra_context+=(--build-context "offline-build-base=oci-layout://$OFFLINE_BUILD_BASE_OCI")
    BUILD_BASE=offline-build-base
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
exec > >(tee -a "$RUN_ROOT/logs/s1-official-source.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_ROOT/s1.exit"' EXIT

[[ $(uname -m) == aarch64 ]]
available_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
available_disk=$(df -B1 --output=avail "$RUN_ROOT" | tail -1)
(( available_kib >= 96 * 1024 * 1024 ))
(( available_disk >= ${MIN_DISK_GIB:-320} * 1024 * 1024 * 1024 ))
[[ $(git branch --show-current) == feature/v032-sudoingx-v20-gb10-tp2 ]]
[[ $(git rev-parse base/v032-dsv41-v36-adapt) == f3e845ca98d32b5646c2671fcf69af4962fe2835 ]]

git diff --binary HEAD > "$RUN_ROOT/manifests/source.patch"
git status --porcelain=v1 > "$RUN_ROOT/manifests/git-status.txt"
git describe --tags --always HEAD > "$RUN_ROOT/manifests/scm-version.txt"
cp "$LOCK_FILE" "$RUN_ROOT/manifests/sources.lock.json"
cp docker/exl3/source-build/build-official-source.sh "$RUN_ROOT/manifests/build-official-source.sh"
printf '%s\n' "$BUILD_STAGE" > "$RUN_ROOT/manifests/stage.txt"
printf 'native_max_jobs=%s\nnvcc_threads=2\nwheel_limit_mib=%s\n' "$NATIVE_MAX_JOBS" "$VLLM_MAX_SIZE_MB" \
    > "$RUN_ROOT/manifests/compile-parallelism.txt"
if [[ -n ${NATIVE_BUILD_INPUT:-} ]]; then
    sha256sum "$NATIVE_BUILD_INPUT/SHA256SUMS" | cut -d' ' -f1 \
        > "$RUN_ROOT/manifests/native-checkpoint.sha256"
else
    printf 'no-checkpoint\n' > "$RUN_ROOT/manifests/native-checkpoint.sha256"
fi
printf '%s\n' "${expected_cutlass:-online-only}" \
    > "$RUN_ROOT/manifests/cxx-sources.txt"
printf '%s\n' "${CUBIN_POLICY_SHA256:-full-architecture}" \
    > "$RUN_ROOT/manifests/cubin-policy.sha256"
printf '%s\n' "${CUDA_STATIC_POLICY_SHA256:-no-static-pruning}" \
    "${CUDA_DEDUPE_POLICY_SHA256:-no-deduplication}" \
    > "$RUN_ROOT/manifests/cuda-runtime-policy.sha256"
if [[ -n ${CUDA_APT_INPUT:-} ]]; then
    sha256sum "$CUDA_APT_INPUT/SHA256SUMS" | cut -d' ' -f1 \
        > "$RUN_ROOT/manifests/cuda-apt-input.sha256"
else
    printf 'online-apt\n' > "$RUN_ROOT/manifests/cuda-apt-input.sha256"
fi
if [[ -n ${RUST_TOOLCHAIN_INPUT:-} ]]; then
    sha256sum "$RUST_TOOLCHAIN_INPUT/SHA256SUMS" | cut -d' ' -f1 \
        > "$RUN_ROOT/manifests/rust-toolchain.sha256"
else
    printf 'online-only\n' > "$RUN_ROOT/manifests/rust-toolchain.sha256"
fi
if [[ -n ${FLASHINFER_WHEELHOUSE:-} ]]; then
    sha256sum "$FLASHINFER_WHEELHOUSE/SHA256SUMS" | cut -d' ' -f1 \
        > "$RUN_ROOT/manifests/flashinfer-wheels.sha256"
else
    printf 'online-only\n' > "$RUN_ROOT/manifests/flashinfer-wheels.sha256"
fi
if [[ -n ${DEPENDENCY_WHEELHOUSE:-} ]]; then
    sha256sum "$DEPENDENCY_WHEELHOUSE/SHA256SUMS" | cut -d' ' -f1 \
        > "$RUN_ROOT/manifests/dependency-wheels.sha256"
else
    printf 'online-only\n' > "$RUN_ROOT/manifests/dependency-wheels.sha256"
fi
printf '%s\n' "${EPEL_BASE_URL:-}" "${PIP_INDEX_URL:-}" \
    "${UV_INDEX_URL:-${PIP_INDEX_URL:-}}" "${UV_INDEX:-}" \
    "${GITHUB_GIT_MIRROR:-}" | sha256sum \
    > "$RUN_ROOT/manifests/dependency-sources.sha256"
sha256sum docker/Dockerfile requirements/cuda.txt \
    "$LOCK_FILE" \
    docker/exl3/source-build/build-official-source.sh \
    docker/exl3/source-build/prune-flashinfer-cubins.py \
    docker/exl3/source-build/slim-cuda-runtime.py \
    docker/exl3/source-build/validate-oci-input.py \
    "$RUN_ROOT/manifests/scm-version.txt" \
    "$RUN_ROOT/manifests/dependency-sources.sha256" \
    "$RUN_ROOT/manifests/dependency-wheels.sha256" \
    "$RUN_ROOT/manifests/flashinfer-wheels.sha256" \
    "$RUN_ROOT/manifests/rust-toolchain.sha256" \
    "$RUN_ROOT/manifests/cubin-policy.sha256" \
    "$RUN_ROOT/manifests/cuda-runtime-policy.sha256" \
    "$RUN_ROOT/manifests/cuda-apt-input.sha256" \
    "$RUN_ROOT/manifests/cxx-sources.txt" \
    "$RUN_ROOT/manifests/stage.txt" \
    "$RUN_ROOT/manifests/compile-parallelism.txt" \
    "$RUN_ROOT/manifests/native-checkpoint.sha256" \
    "$RUN_ROOT/manifests/source.patch" > "$RUN_ROOT/manifests/source-inputs.sha256"
tree_hash=$(sha256sum "$RUN_ROOT/manifests/source-inputs.sha256" | cut -c1-10)
TAG="vllm/vllm-openai:v0.32.0-dsv41-${BUILD_STAGE}-${SOURCE_SHORT}-${tree_hash}"
printf '%s\n' "$TAG" > "$RUN_ROOT/manifests/s1-tag.txt"
extra_output=()
if [[ ${EXPORT_OCI:-0} == 1 ]]; then
    extra_output+=(--output "type=oci,dest=$RUN_ROOT/official.oci,tar=false")
fi

# Full source compilation takes place in csrc-build. The official build stage
# then assembles that freshly compiled local wheel, rather than a downloaded one.
docker buildx build --builder "$BUILDER" --progress=plain --load \
    --platform linux/arm64 --network host --allow network.host "${extra_output[@]}" "${extra_context[@]}" \
    --file docker/Dockerfile --target vllm-openai --tag "$TAG" "${extra_build_args[@]}" \
    --build-arg BUILD_BASE_IMAGE="$BUILD_BASE" \
    --build-arg FINAL_BASE_IMAGE="$RUNTIME_BASE" \
    --build-arg CUDA_VERSION=13.0.3 --build-arg PYTHON_VERSION=3.12 \
    --build-arg max_jobs=6 --build-arg nvcc_threads=2 \
    --build-arg native_max_jobs="$NATIVE_MAX_JOBS" \
    --build-arg torch_cuda_arch_list=12.0 \
    --build-arg DEEPEP_COMMIT_HASH=d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb \
    --build-arg VLLM_USE_PRECOMPILED=0 \
    --build-arg VLLM_BUILD_COMMIT="$SOURCE_SHA" \
    --build-arg VLLM_BUILD_PIPELINE=gb10-official-source \
    --build-arg VLLM_IMAGE_TAG="$TAG" \
    --build-arg INSTALL_KV_CONNECTORS=false --build-arg BUILD_NIXL=false \
    --build-arg INSTALL_RUBIN_PRERELEASE=false --build-arg RUN_WHEEL_CHECK=true \
    --build-arg VLLM_MAX_SIZE_MB="$VLLM_MAX_SIZE_MB" \
    --build-arg EPEL_BASE_URL="${EPEL_BASE_URL:-}" \
    --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-}" \
    --build-arg UV_INDEX_URL="${UV_INDEX_URL:-${PIP_INDEX_URL:-}}" \
    --build-arg UV_INDEX="${UV_INDEX:-}" \
    --build-arg CUBIN_POLICY_SHA256="${CUBIN_POLICY_SHA256:-}" \
    --build-arg CUDA_STATIC_POLICY_SHA256="${CUDA_STATIC_POLICY_SHA256:-}" \
    --build-arg CUDA_DEDUPE_POLICY_SHA256="${CUDA_DEDUPE_POLICY_SHA256:-}" \
    --build-arg GITHUB_GIT_MIRROR="${GITHUB_GIT_MIRROR:-}" \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
    --build-arg NO_PROXY=localhost,127.0.0.1 \
    --metadata-file "$RUN_ROOT/manifests/s1-build-metadata.json" "$SOURCE_ROOT"
docker image inspect "$TAG" > "$RUN_ROOT/manifests/s1-image.json"
if [[ ${EXPORT_OCI:-0} == 1 ]]; then
    HOST_PYTHON=${HOST_PYTHON:-$(dirname "$RUN_ROOT")/.venv/bin/python}
    [[ -x $HOST_PYTHON ]]
    "$HOST_PYTHON" docker/exl3/source-build/validate-oci-input.py \
        --layout "$RUN_ROOT/official.oci" --repair-index \
        --config-digest "$(jq -er '.[0].Id' "$RUN_ROOT/manifests/s1-image.json")" \
        --report "$RUN_ROOT/manifests/official-oci-validation.json"
fi
docker history --no-trunc "$TAG" > "$RUN_ROOT/manifests/s1-history.txt"
printf 'OFFICIAL_SOURCE_IMAGE=%s\n' "$TAG"
