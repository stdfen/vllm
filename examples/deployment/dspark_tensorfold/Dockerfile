FROM local/sudoingx-clean-port:tensorfold-full-runtime-v7-20261121
ARG SOURCE_DATE_EPOCH=1795387651
LABEL org.opencontainers.image.title="vLLM DSV41 independent TensorFold DSpark graph24"
LABEL org.opencontainers.image.version="dspark-tensorfold-graph24-v1-20261123"
LABEL org.opencontainers.image.revision="574524ce02affc442738cb4c83aa765d45489950"
COPY payload/tensorfold_dsv41/ /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/tensorfold_dsv41/
COPY payload/model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
COPY payload/jit_guard.py /usr/local/lib/python3.12/dist-packages/vllm/utils/jit_monitor.py
COPY payload/warm_kernels.py /opt/bench-helpers/warm_kernels.py
COPY payload/markov_top_tokens.json /opt/ab/assets/markov_top_tokens.json
COPY payload/check_autotune_guard.py /opt/check_autotune_guard.py
COPY control.json /opt/bench-stage/control.json
COPY release-manifest.json verify_image.py /opt/dspark-release/
ENV VLLM_TENSORFOLD_BIAS_CACHE_ROWS=1024
ENV VLLM_BENCH_STAGE=/opt/bench-stage/control.json
ENV TRITON_PRINT_AUTOTUNING=1
