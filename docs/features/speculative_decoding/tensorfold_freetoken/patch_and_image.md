# Tree-DFlash2 补丁和 GB10 镜像使用指南

## 应用 vLLM 补丁

完整补丁的基线是 vLLM `4ba787ec9959e566d3b587a796dff07673324da5`。
若已检出 `0.33.0-TensorFold-freetoken`，直接使用该分支源码。
将补丁移植到同一基线时，从本分支取得补丁绝对路径，再在干净检出中执行。

```bash
git switch -c my-vllm-tensorfold 4ba787ec9959e566d3b587a796dff07673324da5
git apply --index --check /path/to/vllm-0.33.0-tensorfold-freetoken.patch
git apply --index /path/to/vllm-0.33.0-tensorfold-freetoken.patch
test "$(git write-tree)" = 9ed7a710ec76d4f6f45423f4877d6f99d3915327
```

| 补丁 | 使用条件 | SHA256 |
| --- | --- | --- |
| [vLLM 完整补丁](../../../../docker/tensorfold-freetoken/patches/vllm-0.33.0-tensorfold-freetoken.patch) | 基于干净的 `4ba787ec99`，合入全部技术点 | `c3ebdbc9f64f760ed0492dbdca20adb3a14581595ad75adcf0e8035adfb9368c` |
| [编译 prefill 增量](../../../../docker/tensorfold-freetoken/patches/vllm-compiled-prefill.patch) | 已有组合树 `de1f7b4b9d33a5125ea131c69c6dc13d99d8436b` 到 `4dcf0872a8c0492280a52384095569f5f4871c51` | `4c6756b69a20ddc2828089ac8cfd650bab9cc73c5e10170fc840aa75d18e7935` |
| [多请求 Graph 分桶增量](../../../../docker/tensorfold-freetoken/patches/vllm-multi-request-graph-buckets.patch) | 已有编译 prefill 树 `4dcf0872a8c0492280a52384095569f5f4871c51` 到最终树 `9ed7a710ec76d4f6f45423f4877d6f99d3915327` | `b4714418cf08ecec4d6bbcabadb5514584a8698a0b8337f9ece228760815736d` |

增量补丁用于升级表中的对应源码树。应用完整补丁即可得到截止本次归档的全部改动。

## TensorFold 0.6.0 依赖

GB10 实测使用带设备草稿 Graph 和 GDN 目标槽位写入的 TensorFold wheel，
文件名为 `tensorfold-0.6.0-py3-none-any.whl`，SHA256 为
`68f8c24002d3efbb1345462e7b7a3df2a462a2d0e06f5ca54ad0ff330ca3cbef`。
Dockerfile 从锁定发布包解包此 wheel。

核对或重新构建 TensorFold 源码时，在 `c4646171139ee8a3c38103eaa1699dad226ec12b`
的独立工作树根目录内按顺序应用下面两个补丁。补丁路径以 wheel 内的
`tensorfold/` 为根，源码布局需要 `--directory=src`。

```bash
patch_dir=/absolute/path/to/vllm/docker/tensorfold-freetoken/patches
git apply --check --directory=src "$patch_dir/tensorfold-draft-device-graph.patch"
git apply --directory=src "$patch_dir/tensorfold-draft-device-graph.patch"
git apply --check --directory=src "$patch_dir/tensorfold-gdn-replay-destination.patch"
git apply --directory=src "$patch_dir/tensorfold-gdn-replay-destination.patch"
uv build --wheel
```

补丁已在上述源码提交上通过顺序应用检查。重新打包的 wheel 字节可能受构建工具影响，
镜像复现实验使用下节的已锁定发布包。
[设备草稿 Graph 补丁](../../../../docker/tensorfold-freetoken/patches/tensorfold-draft-device-graph.patch) ·
[GDN 目标槽位回放补丁](../../../../docker/tensorfold-freetoken/patches/tensorfold-gdn-replay-destination.patch)

## 基于 vLLM 0.33.0 构建镜像

同一个 Dockerfile 提供 `tensorfold-gb10` 和 `combined` 两个构建目标。以下
命令用于 GB10 TensorFold 镜像。 [amd64 TensorFold + FreeToken 构建指南](../../qwen36_freetoken/patch_and_image.md)

在 Linux ARM64、Docker BuildKit 环境执行 `tensorfold-gb10`。构建上下文是锁定发布包解出的
`release/`。工作区内的原包为
`artifacts/20261124-vllm-tensorfold-batch-graph-buckets-aba/release.tar.gz`。
发布包含完整 overlay、逐文件 SHA256、TensorFold wheel 及 CUDA 检查脚本。
基础镜像使用 GB10 实测的本机镜像 ID，容器构建时校验发布清单并核对已安装的
vLLM 模块 Git blob。CUDA 扩展在 GB10 上首次使用时编译，构建上下文无需模型权重。
[Dockerfile](../../../../docker/tensorfold-freetoken/Dockerfile)

```bash
set -eu
repo=$(git rev-parse --show-toplevel)
bundle=/absolute/path/to/release.tar.gz
context=$(mktemp -d)
printf '%s  %s\n' 43c6aecb2ac75e51f17b8f53c4fa29317a54eb25f72933d5dc21e8d0128beee3 "$bundle" |
    sha256sum -c -
tar -xzf "$bundle" -C "$context"
(cd "$context/release" && sha256sum -c bundle.sha256)
base=vllm/vllm-openai:v0.33.0
test "$(docker image inspect --format '{{.Id}}' "$base")" = \
    'sha256:a0dfdeea1d42d4201c82d9bb44f2f4bb1ac2ef5aea196e9e0d1007a2834d2f70'
DOCKER_BUILDKIT=1 docker build --target tensorfold-gb10 --platform linux/arm64 --pull=false --network=none \
    -f "$repo/docker/tensorfold-freetoken/Dockerfile" \
    -t vllm-tensorfold-freetoken:0.33.0-gb10 "$context/release"
docker image inspect --format '{{.Id}}' vllm-tensorfold-freetoken:0.33.0-gb10
```

Dockerfile 对应已实测的运行时覆盖层和 wheel。归档时已核对构建上下文及覆盖层，
新镜像构建与 GB10 服务验证待执行。

## 在 GB10 启动与检查

以下命令使用同一套已锁定的 Qwen3.8-27B ModelOpt NVFP4/FP8
和 DFlash2 权重。修改 `model_dir`、`draft_dir` 和 `cache_dir` 指向本机目录。
主、草稿模型以只读方式挂载，GPU 0 和测试端口 18049 需空闲。

```bash
repo=$(git rev-parse --show-toplevel)
model_dir=/data/models/RadixArk-Qwen3.8-27B-NVFP4
draft_dir=/data/models/Qwen3.8-27B-DFlash2
cache_dir=/absolute/path/to/dedicated-cache
mkdir -p "$cache_dir"
docker run -d --name qwen38-tensorfold-freetoken \
    --gpus device=0 --ipc=host -p 127.0.0.1:18049:8000 \
    -v "$model_dir:/model:ro" -v "$draft_dir:/drafter:ro" \
    -v "$cache_dir:/root/.cache" vllm-tensorfold-freetoken:0.33.0-gb10 /model \
    --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 131072 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75 \
    --attention-backend TRITON_ATTN --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 \
    --language-model-only --generation-config vllm \
    --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}' \
    --kernel-config '{"enable_jit_warmup":false}' \
    --no-async-scheduling --no-enable-prefix-caching --enable-chunked-prefill \
    --speculative-config "$(cat "$repo/docker/tensorfold-freetoken/speculative-gb10.json")"
```

服务就绪后检查 `/health`、`/v1/models` 和固定种子的推理回执。
用日志和 `/metrics` 核对目标验树、接受状态提交及实际 Graph 重放。
`vllm:tensorfold_batch_graph_bucket_total` 按实际批大小和执行路径计数。

```bash
curl -fsS http://127.0.0.1:18049/health
curl -fsS http://127.0.0.1:18049/v1/models
curl -fsS http://127.0.0.1:18049/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","prompt":"The capital of France is","temperature":0,"seed":11230,"max_tokens":16,"ignore_eos":true,"return_token_ids":true}'
```

固定冒烟只验证服务可响应。正确性验收仍需通过匹配 ShareGPT 请求的逐 token 门禁，
并在 C2/C4/C6 等场景核对 Graph 及状态提交证据。现有结果见[合入记录](integration.md)。
回滚时用 `docker stop -t 30 qwen38-tensorfold-freetoken` 释放测试 GPU 和端口，
并按原配置恢复已有服务状态。
