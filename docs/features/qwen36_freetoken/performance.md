# Qwen3.6 FreeToken 单张 RTX 4090 A–B–A 验收

2026-11-24 在一张 RTX 4090（SM89）上完成 vLLM 0.33.0 FreeToken 验收。逐 token 一致性和实际 CUDA Graph 重放通过后，执行了同镜像 A–B–A。固定负载为 C1、8192 输入、1024 输出。 [技术方案](../qwen36_freetoken.md) · [补丁及镜像](patch_and_image.md)

## 镜像与负载

| 项目 | 验收配置 |
| --- | --- |
| 实测镜像 | `vllm-qwen36-sm89-ft:0.33.0-20261124-route-mask`，`sha256:c5ede4ca4596427d4f0a775f718abb26e7495e44b9fa2a639ffb09c948965110` |
| 软件 | 官方 vLLM 0.33.0 amd64 CUDA 13.0 底镜像，PyTorch `2.14.0+cu130`，TVM FFI `0.1.11`，NVIDIA 驱动 `580.105.08` |
| 源码 | vLLM 分支基线 `3487cdb7333ea9a434bba22d0c71e5efd70b09c3`，叠加 FreeToken 源码及填充路由修复。FreeToken 修订 `af71ba43206e124f5ff6419b47ee36c6e9981078` |
| 主模型 | Qwen3.6-35B-A3B BF16，权重索引 SHA-256 `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83` |
| Tokenizer | `tokenizer.json` SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| 服务 | 单卡 TP1，NUMA 1，CPU `16-31,48-63`，V1 模型运行器，无 speculative decoding。最大上下文 12288，单批上限 8192 |
| 缓存和 Graph | 62 GiB UVA 专家卸载预算，B 的 GPU 专家池 13 GiB、2218 槽，FP8 KV 1 GiB，KV 与 Mamba 块大小 4096，`FULL_DECODE_ONLY` 的 Graph batch size 1 |
| 请求 | 同一 tokenizer 生成 `" world" × 8192` 的 token ID。`temperature=0`、`seed=20260909`、`ignore_eos=true`，每请求独立 `cache_salt` |

## 正确性和 CUDA Graph 门禁

先冷启动 A（原生 UVA）和 B（FreeToken）。每边两次预热后采集一个长请求的完整输出 token ID。两边都实际处理 8192 输入及 1024 输出，前缀命中为 0，输出逐 token 一致。长输入实际按 4096 token 分块。固定输入 token ID 的 SHA-256 是 `8684c184abfc6617e76e5ce954b9db4d0bb4286562bf9ef1fe5239b06746a26a`。

两边的短请求 CUPTI trace 各记录 31 次 `cudaGraphLaunch`，服务日志均有单 token 的 `FULL` Decode 统计。B 记录专家池初始化。包括两个门禁样本和 15 个正式性能样本在内，17 个请求的 1024 个输出 ID 全部一致。

此前缺少 route-mask 的镜像在 CUDA Graph 捕获前的 warmup 失败。完整模型诊断确认第 0 层收到 8 个填充 ID `-1`，而 FreeToken LRU 选槽使用有效专家 ID 索引槽表。修复后的镜像通过启动、两道门禁和性能轮。

## A–B–A 方法与结果

A1 和 A2 关闭 FreeToken，B 开启。其余服务、模型、缓存和负载参数相同。各臂重新启动，2 次预热后串行提交 5 个正式请求，单请求超时 600 秒。性能采样期间关闭 profiler。每臂 5 次请求全部成功，前缀缓存命中为 0。

TTFT 为请求提交至首个 token 的时长。TPOT 为首末 token 时间差除以 1023。端到端输出吞吐计至 SSE 结束。下表为中位数及范围。B 与 A1/A2 中位数的算术均值比较，基线漂移为两侧中位数的绝对差除以该均值。

| 指标 | A1 | B | A2 | B 相对基线 | A1/A2 漂移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT（s） | 13.2106（13.2086–13.2118） | 10.5360（10.5335–10.5402） | 13.2231（13.2205–13.2258） | −20.28% | 0.10% |
| TPOT（ms） | 170.1662（170.1630–170.1685） | 9.5385（9.5348–9.5500） | 170.2784（170.2695–170.2809） | −94.40% | 0.07% |
| 端到端输出吞吐（token/s） | 5.4674（5.4674–5.4676） | 50.4543（50.4316–50.4671） | 5.4637（5.4636–5.4640） | 9.23× | 0.07% |
| 峰值显存（MiB，1 秒采样） | 7581 | 20891 | 7581 | +13310 MiB | 0 MiB |

吞吐提升超出两侧基线漂移。此结果适用于已测的固定提示词、BF16 权重、单卡和 C1 配置。amd64 组合镜像随后通过独立构建及依赖检查。TensorFold 的 GPU 性能数据来自单独的 GB10 资格验证。 [GB10 TensorFold A–B–A](../speculative_decoding/tensorfold_freetoken/performance_tensorfold.md)

## 证据及恢复

原始请求、全部 token ID、CUPTI trace、服务日志、GPU 采样和 A–B–A 汇总保存在共享工作区的 `artifacts/qwen36-v033-freetoken-20261124/ft033-route-mask.hsM9IN-evidence.tar.gz`，SHA-256 为 `656987f075dd0f7ee5c14881a8e1a596b489a0200bb00af6a19be438556973d6`。测试退出码为 0，原服务于北京时间 15:35:47 完成恢复核查。
