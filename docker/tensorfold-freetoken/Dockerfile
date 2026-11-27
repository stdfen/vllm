ARG BASE_IMAGE=vllm/vllm-openai:v0.33.0

FROM ${BASE_IMAGE} AS gb10-auto-base

LABEL org.opencontainers.image.title="vLLM 0.33.0 GB10 Tree-DFlash2" \
      org.opencontainers.image.version="0.33.0-gb10-auto-20261125" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.33.0" \
      org.opencontainers.image.base.digest="sha256:a0dfdeea1d42d4201c82d9bb44f2f4bb1ac2ef5aea196e9e0d1007a2834d2f70"

ENV PYTHONPATH=/opt/tree-deps \
    CPATH=/opt/tree-cuda-headers \
    NVCC_PREPEND_FLAGS=-I/opt/tree-cuda-headers \
    MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST=12.1 \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TF_VERIFY_GRAPH_METRICS=1 \
    TREE_DFLASH2_PROFILE=gb10-auto

RUN --mount=type=bind,from=gb10-release,target=/mnt/release,readonly \
    set -eux; \
    cd /mnt/release; \
    test "$(uname -m)" = aarch64; \
    printf '%s  context.sha256\n' 78f2ef9d881fc3e23928b68aa99978833d5d180ebf4fd247a9ec35eee29f37b0 | sha256sum -c -; \
    sha256sum -c context.sha256; \
    uv venv --offline --system-site-packages --python /usr/bin/python3 /opt/tree-venv; \
    /opt/tree-venv/bin/python -m zipfile -e tensorfold-0.6.0-py3-none-any.whl /opt/tree-deps; \
    /opt/tree-venv/bin/python -c 'from pathlib import Path; import runpy; p = Path("/mnt/release/overlay"); runpy.run_path(str(p / "boot.py"))["install"](p)'; \
    vendor_headers=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; \
    mkdir -p /opt/tree-cuda-headers /opt/tree-dflash2 /opt/tree-overlay/source; \
    for header in "$vendor_headers"/cusparse*.h "$vendor_headers"/cusolver*.h "$vendor_headers"/cublas*.h; do \
        test -f "$header" || continue; \
        name=${header##*/}; \
        test -e "/usr/local/cuda/include/$name" || ln -s "$header" "/opt/tree-cuda-headers/$name"; \
    done; \
    test -f /opt/tree-cuda-headers/cusparse.h; \
    command -v nvcc; \
    command -v ninja; \
    install -m 0755 serve.sh /opt/tree-dflash2/serve; \
    cp -a profiles release-lock.json context.sha256 verify_image.py verify_auto.py run_cuda_checks.py /opt/tree-dflash2/; \
    cp overlay/boot.py overlay/overlay-manifest.json /opt/tree-overlay/; \
    cp -a overlay/source/tests /opt/tree-overlay/source/; \
    cd /; \
    /opt/tree-venv/bin/python -B /opt/tree-dflash2/verify_image.py

ENTRYPOINT ["/opt/tree-dflash2/serve"]

FROM gb10-auto-base AS gb10-auto-a1-256k

LABEL org.opencontainers.image.version="0.33.0-gb10-auto-a1-256k"

RUN --mount=type=bind,source=configure_gb10_a1.py,target=/tmp/configure_gb10_a1.py,readonly \
    set -eu; \
    cd /opt/tree-dflash2; \
    /opt/tree-venv/bin/python -B /tmp/configure_gb10_a1.py "$PWD"; \
    printf '%s\n' \
        '9d8fd4eef02c37436d43fa59f081c2e99126e94c2cb7dc329ccde8e919ba3ee7  profiles/gb10-auto.yaml' \
        'e7da9f9175c5fe154a82edafac6ef31e9e3bc04ef751a77b08bae7c195f9c415  release-lock.json' \
        'd393f7cf77924a03c1276627b788d4215c560ce4ebe047f3867f5dfb17dbf05e  verify_image.py' \
        | sha256sum -c -; \
    VLLM_TARGET_DEVICE=cpu /opt/tree-venv/bin/python -B verify_image.py --profiles
