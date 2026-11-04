# vLLM 0.32 / DeepSeek V4.1 Flash v35 代码改造与部署资源清单

> 盘点时间：2026-11-04（Asia/Shanghai）  
> 代码仓库：`D:\llm+infer\vllm`  
> 代码分支：`feature/v032-sudoingx-v20-gb10-tp2`  
> 当前提交：`f0b6e96abe`  
> vLLM 基线：`v0.32.0` / `5df63a7203cdb9328d824a27b682e77c7b9928be`

## 1. 结论摘要

当前版本在 vLLM 0.32.0 上完成了 DeepSeek V4.1 Flash、EXL3、双 DGX Spark TP=2、Engram 磁盘卸载和 DSpark 推测解码的完整适配。两端容器均在线，8.96 提供 OpenAI API，8.164 作为 headless TP1 worker；`/v1/models` 在 8.96 返回 HTTP 200。

功能、一致性和双机部署已经验证，但性能仍低于作者 v2.0 公布基准：作者测试脚本 set-b、384 token、9 reps 的中位数为 99.055 / 53.702 / 134.356 tok/s，约为作者 103.7 / 57.8 / 144.4 的 95.5% / 92.9% / 93.0%。因此当前状态是“可运行且优化路径已启用”，不是“性能完全达到作者版本”。

当前最紧急的资源风险是 8.96 根盘只剩约 31 GiB，`df` 已显示 100%；8.164 剩余约 277 GiB。继续构建镜像或生成 trace 前应先释放 8.96 空间。

## 2. 代码基线与提交结构

相对 vLLM 0.32.0 基线共修改/新增 74 个文件，约新增 7,994 行、删除 59 行。功能按四个提交分层：

| 提交 | 主要内容 | 原因 |
|---|---|---|
| `c1c2cba254` | EXL3 量化、checkpoint 映射、GB10/aarch64 构建、TP2 工具和测试 | vLLM 0.32 原生不识别作者 EXL3 权重布局，也不能直接复用作者的 GB10 运行插件 |
| `1b1098e46e` | DeepSeek V4.1 decode 热路径、greedy local argmax、SWA/索引器及 TP2 优化 | 降低每 token 的全词表通信、元数据构造和小矩阵计算成本 |
| `97542fc849` | Engram 异步磁盘读取与 file-backed 支持 | 约 94 GiB Engram 表无法按普通 GPU/内存权重加载，需要本地磁盘按需读取 |
| `f0b6e96abe` | DSpark Markov cache、本地/融合 argmax、置信度深度策略、RowMM 和最终镜像覆盖 | 补齐作者 v2.0 后半段推测解码优化，并兼容 v0.32 API 与 EXL3 loader 差异 |

## 3. v0.32 代码调整内容与具体原因

### 3.1 EXL3 量化和权重加载

主要文件：

- `vllm/model_executor/layers/quantization/exl3/`
- `vllm/models/deepseek_v41/nvidia/model.py`
- `vllm/models/deepseek_v41/nvidia/dspark.py`
- `vllm/model_executor/model_loader/weight_utils.py`
- `vllm/model_executor/weight_staging.py`

调整内容与原因：

1. 新增 EXL3 quantization config、linear、MoE 和运行时适配，使 vLLM 能识别 `trellis/suh/svh/mul1` 权重片段。
2. 支持 `slice.<id>.<piece>` 与旧式 flat loader 两种 `wo_a` checkpoint 合同。作者插件和 v0.32 运行时暴露参数的方式不同，不兼容会造成 KeyError、漏载或错误调用 slice loader。
3. 针对 MTP/DSpark 的分组 `wo_a`、drafter 自有 K4 lm_head、共享专家 padding 和量化 scale 名称增加映射。
4. 增加 host staging 和 aarch64/CUDA-only ExLlamaV3 构建支持，减少大权重装载期间的临时显存/统一内存压力。
5. 增加 checkpoint、linear、MoE、manifest 和 staging 测试，避免仅以“加载成功”代替形状、分片与数值合同验证。

### 3.2 DeepSeek V4.1 Attention、Indexer 与 Decode 热路径

主要文件：

- `vllm/models/deepseek_v41/attention.py`
- `vllm/models/deepseek_v41/compressor.py`
- `vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/sparse_swa.py`
- `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`

调整内容与原因：

1. Compressor 与 indexer 的小批量 BF16 输入投影接入 EXL3 RowMM。GB10 decode 阶段的小矩阵 cuBLAS split-K 启动和读权重成本偏高，逐行 kernel 更适合 1–8 rows。
2. `wo_a` 增加插件 `grouped_bmm` hook，使 EXL3 分组低秩投影直接处理 `[token, group, K]`，避免逐组 Python/ATen 路径。
3. v0.32 的 indexer/compressor 保持原生 BF16 权重而不是错误套用 EXL3 quant config，修复曾出现的 4096×320 与 hidden size 5120 形状冲突。
4. Sparse SWA 的 15 个 decode metadata build 合并为一次 launch，减少每轮大量小 kernel 的启动开销。
5. RoPE table 按实际 262K 上下文上限裁剪，避免按 1M draft 配置预分配不必要的表。
6. 当前强制 `--block-size 64`：v0.32 默认 128 与本部署的 DeepGEMM indexer 路径不兼容，32/64 才满足 kernel 合同。

### 3.3 Greedy Verify 与跨 TP 通信

主要文件：

- `vllm/v1/worker/gpu/sample/greedy_local_argmax.py`
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
- `vllm/v1/worker/gpu/attn_utils.py`

调整内容与原因：

1. Greedy sampling 在每个 TP rank 的词表分片上先求局部最大值，只 all-gather 少量 `(value, token_id)`，不再 all-gather 全词表 logits。
2. 保持 tie 时选择较小全局 token id，确保与完整词表 `torch.argmax` 的确定性语义一致。
3. 通过 ConnectX-7 RoCE 和 EXL3 RDMA all-reduce 传递小于 1024 KiB 的 BF16 TP collective，降低跨节点同步延迟。

### 3.4 DSpark 推测解码

主要文件：

- `vllm/models/deepseek_v41/nvidia/dspark.py`
- `vllm/models/deepseek_v41/nvidia/ops/markov_cache.py`
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
- `vllm/v1/worker/gpu/spec_decode/depth_policy.py`
- `vllm/v1/worker/gpu/model_runner.py`

调整内容与原因：

1. 启用 drafter 自有 EXL3 K4 lm_head，避免每个 draft 都使用更重的 target head。
2. `main_proj` 在 TP2 上做 row-parallel，并对多个 draft layer 的 context KV 投影分组执行，减少单 rank 权重带宽和 launch 数量。
3. 预计算 4,096 个高频 token 的 Markov bias 行，未命中时才计算 `markov_w2 @ markov_embed`；每 rank 只保留自己的词表列。
4. 将 Markov bias、base logits 相加和局部 argmax/second/logsumexp 融合，减少中间全词表张量与 kernel launch。
5. 从两 rank 的小型 argmax record 计算 margin、top-1 probability，并结合 checkpoint confidence head 和 `dspark_conf_feat_head.npz` 估算接受概率。
6. 新增 `conf+feat` 自适应 verify depth，根据每个 draft 位置的接受概率和实测 round cost `[27.6,31.9,35.6,38.9,42.2,44.8]` 选择验证深度；规则计算可融合执行。
7. 深度策略支持 `/root/.cache/depth_policy.txt` 热更新，便于不重启模型做 fixed-depth 与 cost-table A/B。

### 3.5 Engram 磁盘卸载

主要文件：

- `vllm/config/engram.py`
- `vllm/models/deepseek_v41/common/engram.py`
- `vllm/models/deepseek_v41/common/engram_file.py`
- `vllm/models/deepseek_v41/common/engram_disk.cpp`

调整内容与原因：

1. 增加 safetensors 内 Engram 表的 file-backed 定位和按行读取，避免将约 94 GiB embedding 表常驻内存。
2. 使用 native AIO、`O_DIRECT`、256 reader threads 和 spin proxy，在 decode/prefill 查表时并行读取本地 NVMe。
3. 保留 CPU offload，并禁止 DP shared memory；当前 TP2 两 rank 分别读取各自 Engram shard。
4. 增加异常收尾和 reader 生命周期处理，避免服务退出时残留磁盘线程或映射。

### 3.6 构建、部署与验证工具

主要目录：`docker/exl3/`、`tools/exl3/`、`tests/quantization/`、`tests/models/`。

原因是作者仓库的补丁、插件、镜像层与模型资产相互依赖，仅修改 Python 模型类无法形成可复现部署。当前仓库补充了 aarch64 构建、FlashInfer 兼容层、TP2 preflight、NCCL/Engram/GEMM/MoE smoke、长上下文探针及模型 manifest 校验。

## 4. 当前部署拓扑

```text
客户端
  │ HTTP :8000
  ▼
8.96 / spark-7655 / TP rank 0 / API server
  │ RoCE v2, enp1s0f1np1, rocep1s0f1, GID 3, MTU 1024
  ▼
8.164 / spark-ab35 / TP rank 1 / headless worker

torch.distributed master: 10.253.230.201:25000
tensor parallel size: 2，nodes: 2，每节点 1× NVIDIA GB10
```

## 5. 主机资源与当前状态

| 项目 | 8.96（TP0） | 8.164（TP1） | 状态判断 |
|---|---|---|---|
| 主机名 | `spark-7655` | `spark-ab35` | 正常 |
| OS / kernel | Ubuntu 24.04 系，Linux `7.0.0-1019-nvidia`，aarch64 | 同左 | 一致 |
| CPU | 20 cores：Cortex-X925 + Cortex-A725，`taskset 0-18` | 同左 | CPU 19 留给 RDMA proxy |
| 统一内存 | 121 GiB；used 115 GiB；available 6.5 GiB；swap used 3.9 GiB | 121 GiB；used 114 GiB；available 7.4 GiB；swap used 2.9 GiB | 余量较低，符合大模型运行特征 |
| GPU | NVIDIA GB10，driver `580.178.04`，采样时 0%、52°C | NVIDIA GB10，driver `580.178.04`，采样时 0%、47°C | 空闲采样正常；GB10 使用统一内存，`nvidia-smi` 显存字段为 N/A |
| 根盘 | 3.7 TiB，已用 3.5 TiB，剩 31 GiB，100% | 3.7 TiB，已用 3.3 TiB，剩 277 GiB，93% | **8.96 严重告警；8.164 高水位** |
| RoCE | `10.253.230.201/30`，PORT_ACTIVE | `10.253.230.202/30`，PORT_ACTIVE | HCA `rocep1s0f1`，firmware `28.45.4028`，MTU 1024 |

## 6. 容器和镜像资源

| 项目 | 8.96 | 8.164 |
|---|---|---|
| 容器名 | `dsv41-serve` | `dsv41-serve` |
| 运行状态 | Running | Running |
| 镜像 tag | `vllm/vllm-openai:v0.32.0-dsv41-v35` | 同左 |
| image ID | `sha256:aa35d704...` | `sha256:a5fa30e1...` |
| 镜像大小 | 33,412,468,057 bytes（约 31.1 GiB） | 33,412,468,056 bytes |
| network / IPC | host / host | host / host |
| shared memory | 64 GiB | 64 GiB |
| ulimit | memlock unlimited；stack 64 MiB | 同左 |
| RDMA device | `/dev/infiniband` | `/dev/infiniband` |
| restart policy | `no` | `no` |
| 空闲采样容器 RSS | 6.461 GiB，763 PIDs | 5.731 GiB，703 PIDs |
| block I/O 累计 | 209 GB read / 4.51 GB write | 208 GB / 3.95 GB |

注意：两端镜像分别构建，因此 tag 相同但 image ID 不同；镜像大小只相差 1 byte，部署源码来自同一 v35 build context。若要求供应链完全可复现，应在单一构建节点生成镜像并按 digest 分发，而不是两端分别 build。

`docker stats` 的 RSS 不能代表 GB10 的全部模型占用；GB10 CPU/GPU 共享统一内存，主机 `free` 的 114–115 GiB used 更能反映当前模型驻留压力。

挂载：

- 模型：`/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`，只读。
- 启动脚本：`/tmp/dsv41-serve.sh -> /serve.sh`，只读。
- 缓存：`/home/intellif/.cache/dsv41-serve -> /root/.cache`，可写；保存 torch extensions、Markov token 列表和深度策略文件。

## 7. 模型权重与辅助资产

| 项目 | 当前值 |
|---|---|
| 模型 | `sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` |
| 本地路径 | `/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` |
| safetensors | 两端均为 169 个，逻辑总字节均为 `406,999,747,740` |
| 权重目录表观占用 | 8.96 `389G`；8.164 `385G` |
| 目录普通文件数 | 8.96 为 183；8.164 为 187 |
| config SHA256 | 两端均为 `fd44e12...612d` |
| index SHA256 | 两端均为 `b82bde5e...5450` |
| `dspark_conf_feat_head.npz` | 416 bytes；SHA256 两端均为 `b1414d72...ba2a` |
| `exl3_moe_plan.json` | SHA256 两端均为 `24713216...eb0` |

核心权重数量、逻辑字节数、config、index、置信度头和 MoE plan 均一致；目录文件名集合 hash 不同，说明两端存在不同的附加/缓存文件。此前已执行完整权重 manifest 校验并通过，本次实时盘点未再次读取约 407 GB 权重做全量 SHA256，以免影响在线服务。目录 `du` 和文件总数差异不能直接解释为权重不一致，但建议把非权重附加文件清单纳入镜像发布审计。

## 8. 推理框架和启动参数

### 8.1 公共参数

```bash
taskset -c 0-18 vllm serve \
  /data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw \
  --served-model-name dsv41 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --nnodes 2 \
  --master-addr 10.253.230.201 \
  --master-port 25000 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --load-format auto \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false,"disk_offload":true,"disk_threads":256}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --kv-cache-memory-bytes 2684354560 \
  --limit-mm-per-prompt '{"image":0}' \
  --block-size 64
```

节点差异：

- 8.96：`--node-rank 0`，不加 `--headless`。
- 8.164：`--node-rank 1 --headless`。

### 8.2 关键运行环境变量

| 分类 | 参数 | 作用 |
|---|---|---|
| RDMA/NCCL | `NCCL_NET=IB`、`NCCL_IB_HCA=rocep1s0f1`、`NCCL_IB_GID_INDEX=3`、`NCCL_IB_DISABLE=0` | 使用 ConnectX-7 RoCE v2 |
| socket | `NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=TP_SOCKET_IFNAME=enp1s0f1np1` | 控制面和 collective 绑定 fabric 网卡 |
| EXL3 collective | `EXL3_RDMA_AR=1`、`EXL3_RDMA_AR_MAX_KB=1024` | 小型 BF16 all-reduce 走 EXL3 RDMA |
| EXL3 kernel | `EXL3_ROUTER=1`、`EXL3_ROWMM=1`、`EXL3_MROW_GROUPED=1`、`EXL3_MOE_COOP_WIDE=1` | 启用 router、RowMM、分组 mrow 和宽 MoE cooperative kernel |
| 预取 | `EXL3_L2PF=attn,moe,next`、`EXL3_L2PF_ORDER=1`、`EXL3_L2PF_DRAFTER=1`、`EXL3_L2PF_HC=1` | 提前拉取 attention/MoE/drafter 权重到 L2 |
| DSpark | `VLLM_DSPARK_OWN_HEAD=1`、`LOCAL_ARGMAX=1`、`MARKOV_FUSED=1`、`MAIN_PROJ_ROWPAR=1`、`CONTEXT_KV_GROUPED=1` | 启用作者 drafter 主链优化 |
| 深度策略 | `VLLM_DSPARK_DEPTH_POLICY=conf+feat`、`DEPTH_FUSED=1`、`CONF_FEAT_HEAD=...npz` | 按拟合接受概率选择 verify depth |
| Markov | `VLLM_DSPARK_MARKOV_CACHE=/root/.cache/markov_top_tokens.json` | 缓存 4,096 个高频 token bias 行 |
| DeepSeek | `VLLM_DSV41_SWA_BATCH=1`、`VLLM_DSV41_MHC_OVERLAP=1`、`VLLM_ENGRAM_WKV_ROWPAR=1` | 合并 SWA 构造、MHC overlap、Engram projection 行并行 |
| 离线 | `HF_HUB_OFFLINE=1` | 只使用本地权重，不访问 Hugging Face |

## 9. 运行路径实际生效证据

两端日志均确认：

- `DSpark main_proj row-parallel over 2 ranks`；
- TP0 缓存词表列 `0-64640`，TP1 缓存 `64640-129280`；
- `greedy drafts argmax per vocabulary shard`；
- 从 `dspark_conf_feat_head.npz` 加载 draft feature confidence；
- `DSpark depth policy conf+feat (k=5)`；
- RDMA all-reduce rank 0/1 均建立，proxy 使用 CPU 19；
- `15 SWA decode builds in one launch`；
- EXL3 RowMM 用于 decode-size V4.1 attention input projections。

## 10. 验证结果

| 验证项 | 结果 |
|---|---|
| Python `py_compile` / `git diff --check` | 通过 |
| 双机 TP2 模型启动 | 通过 |
| `/v1/models` | 8.96 HTTP 200；模型 `dsv41`；max length 262144 |
| 作者 greedy repeat | 5/5，384 tokens，输出完全一致 |
| 作者 set-b 9 reps | code 99.055；prose 53.702；structured 134.356 tok/s；27/27 文本一致 |
| 固定深度 3/4/5 | 均劣于 `conf+feat` |
| 禁用 Breakable CUDA Graph | 95.173 / 55.081 / 134.285，整体劣于当前配置 |

## 11. 当前风险与建议

1. **P0 磁盘风险**：8.96 仅剩 31 GiB。先清理旧镜像、失败 build cache 和过期测试产物，再进行 CUDA trace 或新镜像构建；删除前应逐项核对，避免误删当前 v35、模型和缓存。
2. **P1 性能差距**：相同接受轮次下，v0.32 每轮仍比作者 v5 慢约 2.4–3.3 ms。下一步应在同一 prompt 上分别采集作者 v5 与 v35 CUDA trace，定位 scheduler/graph boundary/drafter phase 的残差。
3. **P1 镜像可复现性**：相同 tag 对应两个 image ID。建议集中构建、导出 OCI archive 或推送内部 registry，并按 digest 部署。
4. **P1 容器自恢复**：当前 restart policy 为 `no`。若用于持续服务，应在完成稳定性 soak 后改为受控的 systemd/Compose 编排或 `unless-stopped`，同时避免两节点单边自动重启造成 TP rendezvous 不一致。
5. **P2 资源水位**：两端可用统一内存仅约 6.5–7.4 GiB，swap 已使用 2.9–3.9 GiB；不要在在线期间并行构建大型镜像或运行其他 GPU/内存重负载。
6. **P2 模型目录治理**：核心权重一致，但附加文件集合不同。建议生成只覆盖发布资产的 manifest，区分权重、置信度头、MoE plan、运行缓存和测试产物。

## 12. 可追溯证据

- Git：`5df63a7203..f0b6e96abe`。
- 镜像：`vllm/vllm-openai:v0.32.0-dsv41-v35`。
- 运行容器：两端 `dsv41-serve`。
- 最终 benchmark：`/home/intellif/dsv41_recipe/artifacts/v032-v35/decode-set-b-r9-final.json`。
- 既有实施记录：`docs/deployment/deepseek_v41_1017_exl3_gb10_tp2_v032_execution_report_20261103.md`。
