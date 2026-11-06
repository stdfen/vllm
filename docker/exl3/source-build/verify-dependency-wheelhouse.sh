#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:?absolute task root required}
BUILDER=${BUILDER:-vllm-source-20261105}
OFFLINE_BUILD_BASE_OCI=${OFFLINE_BUILD_BASE_OCI:?verified build base required}
DEPENDENCY_WHEELHOUSE=${DEPENDENCY_WHEELHOUSE:?verified wheel directory required}
UV_BINARY=${UV_BINARY:-$HOME/.local/bin/uv}
mkdir -p "$ROOT/wheelhouse-smoke"
cp "$UV_BINARY" "$ROOT/wheelhouse-smoke/uv"
printf 'Default empty wheelhouse\n' > "$ROOT/wheelhouse-smoke/fallback"
cat > "$ROOT/wheelhouse-smoke/Dockerfile" <<'DOCKERFILE'
FROM scratch AS dependency-wheels
COPY fallback /README
FROM offline-build-base
COPY uv /opt/uv-test/uv
RUN --mount=type=bind,from=dependency-wheels,target=/wheels \
    cd /wheels && sha256sum -c SHA256SUMS \
    && /opt/uv-test/uv venv --python /opt/python/cp312-cp312/bin/python3.12 /tmp/.venv \
    && UV_OFFLINE=1 UV_FIND_LINKS=/wheels /opt/uv-test/uv pip install --no-deps --python /tmp/.venv/bin/python flashinfer-python==0.6.18 \
    && /tmp/.venv/bin/python -c "import importlib.metadata; print(importlib.metadata.version('flashinfer-python'))"
DOCKERFILE
docker buildx build --builder "$BUILDER" --platform linux/arm64 --network none \
    --build-context "offline-build-base=oci-layout://$OFFLINE_BUILD_BASE_OCI" \
    --build-context "dependency-wheels=$DEPENDENCY_WHEELHOUSE" \
    --progress plain --output type=cacheonly "$ROOT/wheelhouse-smoke" \
    2>&1 | tee "$ROOT/wheelhouse-smoke/build.log"
