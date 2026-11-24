ARG BASE_IMAGE=vllm/vllm-openai:v0.33.0
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="vLLM 0.33.0 TensorFold Tree-DFlash2" \
      org.opencontainers.image.revision="4998676ce7e1400517ec3646b3cce6cfd1d01757"

# Use the verified release directory as the build context.
RUN --mount=type=bind,target=/mnt/tf-release,readonly \
    set -eu; \
    cd /mnt/tf-release; \
    printf '%s  bundle.sha256\n' ce6958e981d0daa7828d2e141ab95f5cde191512f08b9f6778eea641654d24d6 \
        | sha256sum -c -; \
    sha256sum -c bundle.sha256; \
    uv venv --offline --system-site-packages --python /usr/bin/python3 /opt/tree-venv; \
    /opt/tree-venv/bin/python -m zipfile -e tensorfold-0.6.0-py3-none-any.whl /opt/tree-deps; \
    /opt/tree-venv/bin/python -c 'from pathlib import Path; import runpy; p = Path("/mnt/tf-release/overlay"); runpy.run_path(str(p / "boot.py"))["install"](p)'; \
    vendor_headers=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; \
    mkdir -p /opt/tree-cuda-headers; \
    for header in "$vendor_headers"/cusparse*.h "$vendor_headers"/cusolver*.h "$vendor_headers"/cublas*.h; do \
        test -f "$header" || continue; \
        name=${header##*/}; \
        test -e "/usr/local/cuda/include/$name" || ln -s "$header" "/opt/tree-cuda-headers/$name"; \
    done; \
    test -f /opt/tree-cuda-headers/cusparse.h

ENV PYTHONPATH=/opt/tree-deps \
    CPATH=/opt/tree-cuda-headers \
    NVCC_PREPEND_FLAGS=-I/opt/tree-cuda-headers \
    MAX_JOBS=2 \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TF_VERIFY_GRAPH_METRICS=1

ENTRYPOINT ["vllm", "serve"]
