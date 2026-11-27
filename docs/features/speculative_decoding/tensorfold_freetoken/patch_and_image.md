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

## 默认构建 GB10 A1 256K 镜像

`docker/tensorfold-freetoken/Dockerfile` 默认构建 `gb10-auto-a1-256k`。历史镜像
`sha256:35dd797ea3526ebcf085a3c867240c95a34344209d0e40144c3f2decd75ade29`
从官方 0.33.0 ARM64 构建 `gb10-auto`，装入 24 个锁定的 vLLM 运行时覆盖文件和
TensorFold 0.6.0 wheel，再替换默认预设、发布清单和验证脚本。当前配方在一个
多阶段构建里完成这两步，校验三个最终文件与历史构建逐字节一致。
Dockerfile 和镜像元数据变化会产生新的镜像 ID。历史性能和逐 token 回执以原镜像 ID 为准。

默认预设使用 262144 上下文、K=15、FP8 KV、FP32 SSM、并发上限 8 和分块 prefill，
关闭前缀缓存。配合历史混合量化权重使用 FP8 Attention 和 NVFP4 MLP。
保留长前缀 C1 独占路由，并启用 A1 的批次候选预算、多请求 GPU 状态提交、
C8 及短前缀通用 Graph 分桶和长前缀 Graph。

在 Linux ARM64 主机上，用支持命名上下文的 Docker BuildKit 执行。将 `bundle` 指向
共享工作区 `artifacts/20261125-gb10-tree-dflash2-auto/release-v2/build-context.tar.gz` 的副本。
模型权重在运行时挂载。 [Dockerfile](../../../../docker/tensorfold-freetoken/Dockerfile)

```bash
set -eu
repo=$(git rev-parse --show-toplevel)
bundle=/absolute/path/to/build-context.tar.gz
context=$(mktemp -d)
recipe="$repo/docker/tensorfold-freetoken"
image=vllm-tensorfold:0.33.0-gb10-auto-a1-256k
printf '%s  %s\n' 78943ec63c3dbb1920f222fcee860eddf74148ec36818ba3cb0e391e9c84431e "$bundle" |
    sha256sum -c -
tar -xzf "$bundle" -C "$context"
test "$(docker image inspect -f '{{.Id}} {{.Architecture}}' vllm/vllm-openai:v0.33.0)" = \
    'sha256:a0dfdeea1d42d4201c82d9bb44f2f4bb1ac2ef5aea196e9e0d1007a2834d2f70 arm64'
DOCKER_BUILDKIT=1 docker build --platform linux/arm64 --pull=false --network=none \
    --build-context "gb10-release=$context/build-context" \
    -f "$recipe/Dockerfile" -t "$image" "$recipe"
docker run --rm --network none "$image" --print-config
```

2026-11-27 已在 GB10 使用默认目标完成构建。验收标签为
`vllm-tensorfold:0.33.0-gb10-auto-a1-256k`，本机镜像 ID 为
`sha256:e3e2763f7e67f0c3c7e06f295111f4a69f1ffebaa8949d2e0d325f593fb520ae`。
离线核对确认默认预设与历史 A1 镜像逐字节一致。

历史 A1 的 105 项 FlashInfer tactic 是宿主机运行缓存。复现实测时，把历史
`20261125-gb10-tree-dflash2-reference-tails/tails/cache/a1/flashinfer/autotune/v2`
复制到专用宿主机缓存目录的 `flashinfer/autotune/v2`，再把该缓存目录挂载到容器
`/root/.cache`。主模型和草稿模型分别挂载到 `/model` 和 `/drafter`。
使用默认预设启动时，镜像名后可以直接传 `--host` 或 `--port` 等覆盖项。

## 在 GB10 启动与检查

以下命令使用上述默认 A1 256K 镜像。请先按上节准备历史 tactic 缓存，
并选用空闲 GPU 和端口。主模型与草稿模型均只读挂载。

```bash
image=vllm-tensorfold:0.33.0-gb10-auto-a1-256k
model_dir=/data/models/RadixArk-Qwen3.8-27B-NVFP4
draft_dir=/data/models/Qwen3.8-27B-DFlash2
cache_dir=/absolute/path/to/dedicated-cache
docker run -d --name qwen38-gb10-auto-a1-256k \
    --gpus device=0 --ipc=host -p 127.0.0.1:18049:8000 \
    --mount "type=bind,src=$model_dir,dst=/model,readonly" \
    --mount "type=bind,src=$draft_dir,dst=/drafter,readonly" \
    --mount "type=bind,src=$cache_dir,dst=/root/.cache" "$image"

curl -fsS http://127.0.0.1:18049/health
curl -fsS http://127.0.0.1:18049/v1/models
curl -fsS http://127.0.0.1:18049/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","prompt":"The capital of France is","temperature":0,"seed":11230,"max_tokens":16,"ignore_eos":true,"return_token_ids":true}'
```

用日志和 `/metrics` 核对目标验树、接受状态提交及实际 Graph 重放。
`vllm:tensorfold_batch_graph_bucket_total` 按实际批大小和执行路径计数。
固定冒烟用于验证服务响应。输出一致性需通过 ShareGPT 逐 token 门禁。

结束测试时用 `docker stop -t 30 qwen38-gb10-auto-a1-256k` 释放测试资源。

[AMD64 / SM89 FreeToken 镜像构建指南](../../qwen36_freetoken/patch_and_image.md)
