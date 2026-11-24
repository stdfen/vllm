ARG BASE_IMAGE=vllm/vllm-openai:v0.33.0
ARG AMD64_BASE_IMAGE=vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3
FROM ${AMD64_BASE_IMAGE} AS build

ENV MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 TVM_FFI_CUDA_ARCH_LIST=8.9
WORKDIR /tmp/freetoken
COPY freetoken/ .
RUN test -n "$(command -v nvcc)" && \
    test -n "$(command -v ninja)" && \
    test "$(grep -c -- '-std=c++17' setup.py)" -eq 3 && \
    sed -i 's/-std=c++17/-std=c++20/g' setup.py && \
    python3 setup.py build_ext --inplace

FROM ${AMD64_BASE_IMAGE} AS freetoken
ENV MAX_JOBS=4 TVM_FFI_CUDA_ARCH_LIST=8.9
COPY --from=build /tmp/freetoken/python/freetoken /tmp/qwen36-runtime/freetoken
COPY --from=build /tmp/freetoken/LICENSE /usr/share/licenses/freetoken/LICENSE
COPY vllm/ /tmp/qwen36-runtime/vllm
COPY source-manifest.json /usr/share/qwen36-freetoken/source-manifest.json
COPY verify_runtime.py verify_decode_routes.py vllm-0.33.0-qwen36-freetoken.patch /tmp/qwen36-runtime/
COPY flashlib-0.3.0-py3-none-any.whl /tmp/qwen36-runtime/
RUN cd /tmp/qwen36-runtime && \
    printf '%s\n' '9037911f315fef3c9f4453261597363abcb83868813ad1b258ab1308b34fd11a  flashlib-0.3.0-py3-none-any.whl' | sha256sum -c - && \
    uv --no-config pip install --system --no-cache --no-index --no-deps ./flashlib-0.3.0-py3-none-any.whl && \
    mkdir /tmp/qwen36-runtime/build-driver && \
    ln -s "${CUDA_HOME}/targets/x86_64-linux/lib/stubs/libcuda.so" /tmp/qwen36-runtime/build-driver/libcuda.so.1 && \
    LD_LIBRARY_PATH="/tmp/qwen36-runtime/build-driver:${LD_LIBRARY_PATH}" \
        python3 -P /tmp/qwen36-runtime/verify_runtime.py && \
    TRITON_INTERPRET=1 python3 -P verify_decode_routes.py interpret \
        vllm/model_executor/layers/fused_moe/qwen36_freetoken.py && \
    TRITON_INTERPRET=0 TRITON_CACHE_DIR=/tmp/qwen36-runtime/triton \
        python3 -P verify_decode_routes.py compile \
        vllm/model_executor/layers/fused_moe/qwen36_freetoken.py && \
    cd / && \
    rm -rf /tmp/qwen36-runtime

FROM freetoken AS combined
COPY tensorfold-0.6.0-py3-none-any.whl verify_runtime.py /tmp/qwen36-tensorfold/
RUN set -eu; \
    cd /tmp/qwen36-tensorfold; \
    printf '%s\n' '68f8c24002d3efbb1345462e7b7a3df2a462a2d0e06f5ca54ad0ff330ca3cbef  tensorfold-0.6.0-py3-none-any.whl' | sha256sum -c -; \
    uv --no-config pip install --system --no-cache --no-index --no-deps ./tensorfold-0.6.0-py3-none-any.whl; \
    vendor_headers=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; \
    mkdir /opt/tensorfold-cuda-headers build-driver; \
    for header in "$vendor_headers"/cusparse*.h "$vendor_headers"/cusolver*.h "$vendor_headers"/cublas*.h; do \
        if [ -f "$header" ]; then ln -s "$header" "/opt/tensorfold-cuda-headers/${header##*/}"; fi; \
    done; \
    test -f /opt/tensorfold-cuda-headers/cusparse.h; \
    ln -s "${CUDA_HOME}/targets/x86_64-linux/lib/stubs/libcuda.so" build-driver/libcuda.so.1; \
    LD_LIBRARY_PATH="$PWD/build-driver:${LD_LIBRARY_PATH}" python3 -P verify_runtime.py --tensorfold; \
    cd /; \
    rm -rf /tmp/qwen36-tensorfold

ENV CPATH=/opt/tensorfold-cuda-headers \
    NVCC_PREPEND_FLAGS=-I/opt/tensorfold-cuda-headers

# Preserve the verified GB10 TensorFold-only release build.
FROM ${BASE_IMAGE} AS tensorfold-gb10

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

# BuildKit selects the amd64 combination by default.
FROM combined AS release
