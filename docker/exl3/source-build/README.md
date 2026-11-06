# GB10 源码镜像实施脚本

修改在 `feature/v032-sudoingx-v20-gb10-tp2` 上进行；
`base/v032-dsv41-v36-adapt` 固定保留原始基线。

宿主工具要求：Git、Docker/Buildx、jq、zstd、sha256sum、uv，以及到目标节点的 SSH。

1. 将干净 Git bundle 克隆到构建节点，复核 HEAD 和 base ref，再应用当前 feature 的构建改动。
2. 使用 `sources.lock.json` 校验基础镜像 digest、ExLlamaV3 源码和 EXL3 MoE wheel。
3. 使用 `buildkitd.toml` 创建专用 docker-container builder。通过环境变量传入临时代理，禁止写入凭据。
4. 在源码根目录设置 `RUN_ROOT`，执行 `bash docker/exl3/source-build/build-official-source.sh`。
   `csrc-build` 完整编译 native 扩展，官方 `build` 阶段使用刚编译的本地 wheel 组装包。
5. 将锁定的 EXL3 MoE wheel 放入 `$RUN_ROOT/final-context/`，设置 `SOURCE_ARCHIVE`，
   执行 `bash docker/exl3/source-build/build-final-full.sh`。此步骤从源码重编 ExLlamaV3。
6. 对 S1 完整参考镜像完成 GPU、双节点模型和性能基线验证后，才能采集 S2 文件轨迹并生成 allowlist。
   S2/S3 不允许用未经过轨迹验证的删除清单替代实际测试。S4 的可选依赖在官方构建入口已禁用，
   由 `verify-image.py` 进行负向验证。
7. S5 部署缓存使用独立宿主目录，并按 image ID 隔离。参考已有双节点运行配置，
   保留已验证的 128K 上下文、固定 4 GiB KV 和 `FULL_DECODE_ONLY`，不能使用通用脚本默认 512K 替代。
8. 最终阶段通过后，设置 `IMAGE`、`SSH_TARGET`、`QSFP_INTERFACE`、`RUN_ROOT`，
   执行 `bash docker/exl3/source-build/transfer-qsfp.sh`。目标节点只接收镜像，不重新编译。
   传输脚本验证路由和两节点 image ID，并避免在目标节点保存完整压缩归档。

所有新增脚本在实际执行前应先通过 `bash -n`。`verify-image.py` 的 GPU 模式必须在 GB10 上执行。
脚本存在或静态检查通过，不代表各阶段已经构建或模型已经验收。

参考组装会运行 `verify-image.py --gpu`，并以新的空缓存目录、`--network none` 和
`FLASHINFER_NO_DOWNLOAD=1` 执行 `smoke-kernels.py`：检查 vLLM RMSNorm、FlashInfer
prefill/decode、Triton JIT 和 CUDA Graph 的数值结果。这些小型检查不能替代 EXL3 双节点模型验收。

## 离线基础镜像输入

当构建节点下载大层失败时，可在网络可用的机器获取 `sources.lock.json` 中固定 digest 的
`linux/arm64` 镜像。每个压缩 blob 必须校验大小和 SHA256；传输前后还应核对整个归档的 SHA256。
解压后的 OCI layout 应包含 `index.json`、`oci-layout`、`blobs/sha256/`。

`docker load` 用于宿主 Docker 导入；专用 docker-container builder 使用独立存储，
因此构建时另行传入 OCI layout，不依赖宿主本地镜像 tag：

```bash
cd /home/intellif/vllm-source-build-20261105/src
export OFFLINE_BUILD_BASE_OCI=/home/intellif/vllm-source-build-20261105/offline-builder.oci
export RUN_ROOT=/home/intellif/vllm-source-build-20261105/run-03
export EXPORT_OCI=1
bash docker/exl3/source-build/build-official-source.sh
```

构建入口检查该 layout 的 manifest digest 与锁文件一致，并将其绑定为
`offline-build-base` named context。离线基础镜像只替代基础层下载；APT、PyPI、Git、Cargo
等后续构建依赖仍需网络或已经验证的对应离线缓存。

当 EPEL metalink 返回过期的 repodata 文件时，可设置 `EPEL_BASE_URL` 为已验证的
单个 EPEL 仓库根地址，例如 `https://dl.fedoraproject.org/pub/epel/8/Everything/aarch64/`。
构建仅调整 `[epel]` 的地址，保留原有 RPM 签名验证。`PIP_INDEX_URL`、`UV_INDEX_URL`
可覆盖构建时包下载入口；包版本仍由源码 requirements 决定。下载源配置的哈希计入构建 tag，
这些参数不写为最终镜像的运行环境变量。

若需要普通包优先使用镜像源，而 Torch 仍从官方 CUDA index 选择对应本地版本，设置
`UV_INDEX` 与镜像地址相同。只设置 `UV_INDEX_URL` 可能仍让 PyTorch extra index 优先提供
NumPy 等普通包。当前保持 `unsafe-best-match`，并已验证 Torch 选择 `2.13.0+cu130`。
共享 UV 缓存的两条并行安装路径将锁等待设置为 3600 秒，保留 UV 自身的缓存互斥。

在源码根目录，统一启动官方源码构建、资源采样和后续参考组装：

```bash
export RUN_ROOT=/absolute/path/to/a-new-run
export SOURCE_ARCHIVE=/absolute/path/to/exllamav3-source.tar.gz
export EXL3_PLUGIN_WHEEL=/absolute/path/to/vllm_exl3_moe-0.1.0.dev0-py3-none-any.whl
export OFFLINE_BUILD_BASE_OCI=/absolute/path/to/offline-builder.oci
export EPEL_BASE_URL=https://dl.fedoraproject.org/pub/epel/8/Everything/aarch64/
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_INDEX_URL="$PIP_INDEX_URL"
export UV_INDEX="$PIP_INDEX_URL"
export EXPORT_OCI=1
# 按实际网络条件设置 HTTP_PROXY / HTTPS_PROXY。
bash docker/exl3/source-build/start-source-reference.sh
```

启动脚本保留 `s1.pid`、`pipeline.pid` 和退出标记。成功的 S1 参考组装还会导出
`exllama-wheels/` 及 SHA256，后续打包可复用本次源码编译出的同 ABI wheel。
Windows 上传的构建文本应转换为 LF；远端先执行 `git diff --check` 与 `bash -n`。

大型 Python 依赖可先执行 `fetch-dependency-wheels.py /absolute/path/to/download-root`。
该脚本按官方 PyPI 元数据选择 ARM64 wheel，支持分段续传，并检查完整文件的 SHA256。
设置 `DEPENDENCY_WHEELHOUSE=/absolute/path/to/download-root/wheelhouse` 后，构建通过
只读 named context 使用这些 wheel；不会把下载目录复制到最终镜像。
用 `verify-dependency-wheelhouse.sh` 可在禁网条件下验证输入和 FlashInfer Python 安装。
AlmaLinux 构建阶段不兼容 manylinux_2_34 的 Z3/TileLang wheel 时仍会从源码构建；
取消构建遗留的 CMake 临时解释器路径失效时，只清理相应包的 UV 缓存后重试。

FlashInfer 的全量 cubin/JIT wheel 可单独置于 `FLASHINFER_WHEELHOUSE`，目录内必须有
`SHA256SUMS`。构建入口和安装层分别校验文件，安装时设置 `PIP_NO_INDEX=1`，避免再次下载。
此输入保留完整架构，用于 S1 参考镜像；S2 裁剪必须另按实际模型轨迹审批清单执行。
依赖 wheelhouse 和 FlashInfer wheelhouse 的清单哈希均计入构建标识。

Rust 工具链也可作为构建输入复用：`RUST_TOOLCHAIN_INPUT` 目录中放置 `toolchain.tar.gz`
与 `SHA256SUMS`。归档仅包含从官方 rustup 安装并已验证的 `.rustup/` 与 `.cargo/bin/`，
不包含 vLLM 二进制；工具链版本仍由 `rust-toolchain.toml` 决定。输入只在 Rust 构建阶段挂载，
清单哈希计入 tag。首次联网安装仍可不设置该参数；归档应先在禁网容器中验证 rustc/cargo 版本。
后续 Rust relink 阶段显式保留 Cargo PATH，避免重新下载已经安装的 rustup。

原生 CUDA 路径的 `NATIVE_MAX_JOBS` 默认 6，`NVCC_THREADS` 固定 2，官方 setup.py
会进一步相除，因此实际 Ninja 并行度是 3。当前 GB10 构建使用 `NATIVE_MAX_JOBS=16`，
实际为 8 个 Ninja 任务；是否提高应结合内存采样。该参数不改变 DeepEP 等其他阶段的 jobs。
当前 AlmaLinux EPEL ccache 对 CUDA 显示 unsupported source language，不能把它当作有效 CUDA 缓存。

当前 fork 的 Git 包版本为 0.32.1.dev7，wheel 实测 506.39 MiB，略超官方默认 500 MiB 检查。
本地镜像实施设置 `VLLM_MAX_SIZE_MB=512`，使用官方已有参数保留大小检查，参数计入输入指纹；
不是禁用检查或减少了 wheel 内容。镜像标签中的 v0.32.0 表示本项目兼容系列，包仍保留真实 Git 版本。

如需中途调整并行度，可用 `checkpoint-native-build.sh` 指定 `SETUP_PID`、`NINJA_PID`
和新的 `NATIVE_BUILD_INPUT` 目录。脚本短暂停止 Ninja 调度、等待现有编译器完成，
保存本次源码构建的 `.deps/`、`build/` 与生成的 `csrc/`，退出时恢复 Ninja。
接续时设置 `NATIVE_BUILD_INPUT`；入口校验归档、当前 native 源文件、固定基础与编译配置。
这是当前构建增量检查点，不是下载或复用历史 v36 的 native 二进制。首次从零构建不需要它。

`CXX_SOURCES=/absolute/path/to/vendor` 可提供 `cutlass/` 干净 Git checkout。入口核对
`sources.lock.json` 中的 CUTLASS 提交、`git fsck` 与包含 ignored 文件的工作区状态，
再使用 CMake 原生 `VLLM_CUTLASS_SRC_DIR`。源码只进入编译阶段。
`GITHUB_GIT_MIRROR` 可设置公共 HTTPS Git 镜像前缀（末尾带 `/`），仅供 CUDA 编译阶段的
后续 Git 依赖下载；固定的 Git commit/tag 仍由源码 CMake 决定，镜像地址哈希计入 tag。

S2 通过参考验证后，设置 `CUBIN_POLICY_INPUT`（含 `flashinfer-cubins.json`）与
`CUBIN_POLICY_SHA256`。脚本逐项验证架构、文件 SHA、运行轨迹保留项和 wheel RECORD，
只在安装 FlashInfer 的同一个 RUN 中删除批准的非 GB10 cubin，避免历史层继续保留大文件。
未提供策略时保留全量架构；S1 不执行实际裁剪。该参数不能替代参考模型与轨迹验收。

`make-cubin-policy.py` 要求参考镜像的真实 API 成功记录和独立 rank0/rank1 strace 目录，
并用 cuobjdump 交叉核查 CUDA ELF 架构。实际观察到的 cubin、未知格式和 SM120/121 均保留。
参考模型验证用 `verify-model-api.py --container NAME --output /absolute/path/model-api.json`，
在 rank0 宿主 venv 中执行；它等待本地模型列表并发起真实生成，不输出部署凭据。
该基本请求检查不等于吞吐或完整形状矩阵验收。

S2/S3 官方镜像完成后，用 `BUILD_STAGE=s2` 或 `s3`、`RUN_ROOT` 和 `REFERENCE_ROOT`
执行 `build-final-reuse.sh`；它要求参考组装和参考模型成功，复用本次源码编译并校验过的
ExLlamaV3 wheel，使用 `Dockerfile.final-reuse` 在干净官方阶段上安装。组装禁网并重跑 GPU 检查。

参考部署可用 `deploy-from-baseline.py --offline` 禁止 Hugging Face 和 FlashInfer 自动下载。
使用 `--trace-bundle` 与 `--trace-output` 时，诊断工具仅通过 bind mount 注入并跟踪文件访问；
模型验收与性能测量应另做不带跟踪器的运行。双节点保持原来的 host 网络供 NCCL/RDMA 通信，
单节点内核 smoke 另在 `--network none` 下执行。

QSFP 流传输显式关闭 SSH 重复压缩，默认使用 `aes128-gcm@openssh.com`，可由
`SSH_CIPHER` 覆盖。传输记录包含网卡和 cipher；吞吐数值应区分到空设备的通道测试和
包含 Docker 导入的最终端到端传输。
目标节点空间预检按逻辑镜像大小的两倍再加 8 GiB 余量执行，因为 Docker load
会在临时解包与提交层之间产生空间峰值。保留原回滚镜像，仅清理本任务确认可重建的中间产物。

S3 设置 `CUDA_RUNTIME_POLICY_INPUT`（含 static.json、dedupe.json）及对应
`CUDA_STATIC_POLICY_SHA256`、`CUDA_DEDUPE_POLICY_SHA256`。必须来自当前参考镜像、
已通过真实生成并审核两端文件轨迹；在 CUDA apt 安装层删除指定静态库，在 Python NVIDIA
wheel 首次安装层将内容完全相同的动态库改为相对 symlink，保留运行时 JIT 工具和头文件。

ExLlamaV3 正式源码为官方 v1.6.0 提交 `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`，
归档 SHA256 见 `sources.lock.json`。旧 e648f1a 源码实际为 1.4.5，不能用作 1.6.0 源码构建。
`RUNTIME_WHEEL_INPUT` 提供锁文件指定的 marisa-trie ARM64 wheel 与 `SHA256SUMS`；
最终组装从本地安装并校验，不隐式升级 Torch。CUDA-only ARM 适配包含 1.6.0 新增 CPU 接口占位。

OCI 导出后使用宿主 venv 的 `validate-oci-input.py` 校验配置、压缩层 hash 和解压 diffID。
双导出产生缺失索引引用时，仅在唯一候选 manifest 的全部内容通过校验后修复引用并备份原索引。
`HOST_PYTHON` 默认指向 `RUN_ROOT` 父目录的 `.venv/bin/python`；后续组装核对验证收据与当前索引。

`make-cuda-runtime-policy.py` 将当前静态/去重清单、真实生成收据、冷缓存内核检查和两端
文件轨迹绑定到策略；若轨迹出现静态候选文件则拒绝批准，应用时再次核对完整候选内容。

原 `/root/.cache` 还可能含 DSpark 必需的 `markov_top_tokens.json`：部署脚本根据
`VLLM_DSPARK_MARKOV_CACHE` 找到原 bind 中的单个文件，以只读方式挂载并记录 SHA256。
它是模型运行输入，不能随编译缓存一起丢弃。

`CUDA_APT_INPUT` 可提供已通过 APT 校验的 `.deb` 与 `SHA256SUMS`。入口核对清单，
运行时 CUDA 安装层在更新索引后将这些包放入 APT 下载目录，继续使用正式 apt 安装流程；
安装后显式清除 `.deb`。仅复用下载输入，不把包归档写入最终镜像。

`run-slim-stage.sh` 依次执行官方阶段、OCI 验证、源码 wheel 复用组装和 GPU 数值检查，
要求新的 RUN_ROOT、BUILD_STAGE=s2/s3 及已通过真实生成的 REFERENCE_ROOT。
`benchmark-model-api.py` 在无跟踪器部署上测量四种固定文本负载、每种五轮，记录实际 token 数、
输出 hash 与吞吐中位数；可通过 `--reference` 比较 3% 门槛，不覆盖视觉和最大上下文矩阵。

模型验证入口为 `run-model-validation.sh`：设置 RUN_ROOT、CONTAINER，及 S1 的
`REFERENCE_BENCHMARK=/absolute/reference/manifests/model-benchmark.json` 后执行。
结果写入 `manifests/model-api.json`、`manifests/model-benchmark.json`，退出码原子写入
`model-validation.exit`。有对照时，任何测量项低于参考吞吐的 97% 都使入口退出 1。
S3 还需设置 `PRIOR_STAGE_ROOT` 为已验证 S2 目录；先将目标节点的上述三个文件同步回构建节点，
入口核对 S2 image ID、参考 image ID 和性能门槛后再构建。
