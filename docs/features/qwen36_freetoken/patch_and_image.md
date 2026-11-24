# FreeToken 补丁使用和镜像构建

本指南适用于 vLLM 0.33.0 上的 BF16 Qwen3.6-35B-A3B 单卡 FreeToken 专家卸载。源码补丁包含 vLLM 接入及 CUDA Graph 填充路由修复。TensorFold + FreeToken 组合镜像在 `docker/tensorfold-freetoken/Dockerfile` 构建时应用这份补丁。 [技术方案](../qwen36_freetoken.md) · [性能报告](performance.md)

## 取得并应用补丁

| 项目 | 值 |
| --- | --- |
| 补丁文件 | `vllm-0.33.0-qwen36-freetoken.patch` |
| SHA-256 | `b0fb32417fbeffe8ed7733e01fc45d758759d0fac35152c02f80f2111281b508` |
| 官方源码基线 | `4ba787ec9959e566d3b587a796dff07673324da5` |
| 官方基线应用后的 Git tree | `ea7c0d04644e70974d208d27925b30c755360469` |
| 带 TensorFold 的集成基线 | `3487cdb7333ea9a434bba22d0c71e5efd70b09c3` |
| 集成基线应用后的 Git tree | `e9ee04d33387eb8821acad776fcc811c93eb88c9` |

[完整 FreeToken 源码补丁](../../../docker/tensorfold-freetoken/patches/vllm-0.33.0-qwen36-freetoken.patch)

从已经包含 FreeToken 的分支构建时，直接使用当前分支源码。移植补丁时，在目标 vLLM 仓库的根目录选用上述两个基线之一。先确保工作区和暂存区干净，并将 `patch` 改成补丁的绝对路径。

```bash
set -eu
patch=/absolute/path/to/vllm-0.33.0-qwen36-freetoken.patch
base=4ba787ec9959e566d3b587a796dff07673324da5
expected_tree=ea7c0d04644e70974d208d27925b30c755360469
test -z "$(git status --porcelain)"
printf '%s  %s\n' b0fb32417fbeffe8ed7733e01fc45d758759d0fac35152c02f80f2111281b508 "$patch" |
    sha256sum -c -
git switch -c qwen36-freetoken-v033 "$base"
git apply --index --check --whitespace=error "$patch"
git apply --index --whitespace=error "$patch"
test "$(git write-tree)" = "$expected_tree"
```

使用集成基线时，将 `base` 和 `expected_tree` 一起换为表中的集成值。兼容性检查失败时，应先核对基线与本地改动。只需撤销刚应用的补丁时，先执行 `git apply --index --reverse --check "$patch"`，再执行 `git apply --index --reverse "$patch"`。

从干净的官方基线移植两个技术点时，先用 `git apply --index --check` 和 `git apply --index` 应用 TensorFold 完整补丁，再检查并应用 FreeToken 补丁。两份补丁合入后的 Git tree 是 `617ce026317414ef87262126be2b1611300c4961`。源码补丁的接收仓库可从当前分支取得 `docker/tensorfold-freetoken` 下的构建配方和构建助手。 [TensorFold 完整补丁](../../../docker/tensorfold-freetoken/patches/vllm-0.33.0-tensorfold-freetoken.patch)

## 用官方 0.33.0 底镜像构建 TensorFold + FreeToken 组合镜像

使用 Linux amd64 主机、Docker、Git 和 Python 3.12 以上的 `uv` 环境。`combined` 目标以官方 vLLM 0.33.0 Linux amd64 CUDA 镜像为基础，先装入 TensorFold Tree-DFlash2 源码，再对镜像内的 vLLM 包检查并应用 FreeToken 源码补丁，禁止模糊匹配。构建会核对六个补丁文件的结果是否与当前工作区清单一致。

镜像沿用 PyTorch `2.14.0+cu130`、Triton `3.8.0` 和 TVM FFI `0.1.11`。FreeToken 固定在 `af71ba43206e124f5ff6419b47ee36c6e9981078`，扩展以 C++20 为 SM89 编译。FlashLib `0.3.0` 与 TensorFold `0.6.0` 的 wheel 经 SHA-256 校验后离线安装。构建阶段还检查 FreeToken 的扩展 ABI、TVM FFI JIT、路由解释器与 SM89 编译，以及 TensorFold 的 DFlash2 和 vLLM 接入模块导入。 [TensorFold + FreeToken Dockerfile](../../../docker/tensorfold-freetoken/Dockerfile)

从同时包含两项技术点的 vLLM 仓库根目录执行。TensorFold wheel 文件名为 `tensorfold-0.6.0-py3-none-any.whl`，SHA-256 为 `68f8c24002d3efbb1345462e7b7a3df2a462a2d0e06f5ca54ad0ff330ca3cbef`。在共享工作区可从 `framework-code/artifacts/20261124-vllm-tensorfold-batch-graph-buckets-aba/release/` 取得锁定的 wheel。构建上下文使用当前工作区相对于官方基线的 `vllm/` Python 源码差异，暂存的补丁也会纳入。上下文清单记录基线、源码 commit、FreeToken 修订、wheel 哈希和覆盖文件哈希。 [TensorFold wheel 的来源及源码补丁](../speculative_decoding/tensorfold_freetoken/patch_and_image.md#tensorfold-060-依赖)

```bash
set -eu
work_dir=$(mktemp -d)
repo=$(git rev-parse --show-toplevel)
freetoken="$work_dir/FreeToken"
context="$work_dir/image-context"
tensorfold_wheel=/absolute/path/to/tensorfold-0.6.0-py3-none-any.whl
git init -q "$freetoken"
git -C "$freetoken" fetch --depth=1 --no-tags \
    https://github.com/FlashML-org/FreeToken.git af71ba43206e124f5ff6419b47ee36c6e9981078
uv run --no-project --python 3.12 python \
    "$repo/docker/tensorfold-freetoken/prepare_context.py" "$freetoken" "$context" \
    --tensorfold-wheel "$tensorfold_wheel"
docker build --platform linux/amd64 --target combined --network=none \
    -f "$repo/docker/tensorfold-freetoken/Dockerfile" \
    -t vllm-tensorfold-freetoken:0.33.0-amd64-sm89 "$context"
docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' \
    vllm-tensorfold-freetoken:0.33.0-amd64-sm89
```

有本地 `flashlib-0.3.0-py3-none-any.whl` 时可传 `--flashlib-wheel /absolute/path/to/flashlib-0.3.0-py3-none-any.whl`。底镜像已离线导入时，可用 `--build-arg AMD64_BASE_IMAGE=<本机镜像标签>` 指定等价的官方 amd64 镜像。已用于验证的 zstd 官方底镜像 manifest 是 `sha256:0b6e0dcf7982ff46914d4eb87a7405831889704477755b1a54090692730d2186`，它的解压层列表及运行配置与 Dockerfile 锁定的官方 gzip 镜像一致。

只应用 FreeToken 补丁的源码可以省略 `--tensorfold-wheel`，并用 `docker build --target freetoken` 构建 FreeToken 单独镜像。


`combined` 已使用本地官方 amd64 底镜像离线构建，验收标签为 `vllm-tensorfold-freetoken:0.33.0-amd64-sm89-applied-patch`，镜像 ID 为 `sha256:051858e24630e2fe77b204d6ce5a45c496fd3d4755257c53e8eaf986ae546f23`。构建日志记录了对六个 vLLM 源文件的补丁应用和内容校验，以及 FreeToken CUDA 扩展和 TensorFold Tree-DFlash2 的检查结果。

## 启动检查

在目标机上选用空闲 SM89 GPU，将本地 BF16 模型权重以只读方式挂载到 `/model`，按[运行参数](../qwen36_freetoken.md#example)启动镜像中的 `vllm serve`。容器启动时通过 `--gpus device=...` 指定单卡，确保 CPU 内存可容纳至少 60 GiB 的 pinned 专家权重以及加载开销。为验证 A/B，固定同一镜像、模型、启动参数和请求，只切换 `VLLM_QWEN36_FREETOKEN`。核对 `/health`、`/v1/models`、B 的专家池日志、逐 token 一致性及 CUPTI 实际重放，再执行性能测试。 [测试合同和结果](performance.md)

两项技术点在同一镜像交付，按服务配置各自使用。FreeToken 路径使用 `VLLM_USE_V2_MODEL_RUNNER=0`，关闭 speculative decoding。TensorFold Tree-DFlash2 的既有验收在 GB10 上完成，使用 `VLLM_USE_V2_MODEL_RUNNER=1`、兼容的主模型及 DFlash2 草稿权重，并设置 `VLLM_QWEN36_FREETOKEN=0`。amd64 组合镜像中的 TensorFold 已做构建和导入检查，GPU 推理需按目标硬件重新验收。 [TensorFold 的验收配置](../speculative_decoding/tensorfold_freetoken/integration.md)
