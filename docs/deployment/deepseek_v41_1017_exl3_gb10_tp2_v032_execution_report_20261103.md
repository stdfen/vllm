# DeepSeek V4.1 Flash EXL3 双 GB10 / vLLM 0.32 续实施报告

## 结论

本轮已将两节点模型与 Engram 全部切换为各节点 `/data/models` 本地加载，并完成可靠基线、131K 上下文与并发 1/2/4 验证。最终在线配置为 vLLM 0.32.0、TP=2、双节点、EXL3、file-backed 本地 Engram、eager、无 DSpark，最大模型长度 131072。

DSpark 与 CUDA Graph 均完成了实现和运行门禁，但当前结果不满足上线条件：DSpark 可启动且确实生成 draft token，但接受率仅 3.85%，性能显著退化；CUDA Graph 在最小捕获集合 `[1,2,3,4]` 上首个 PIECEWISE 捕获即产生 empty graph 并导致 worker 退出。因此两项被明确判定为 no-go，而不是作为完成配置启用。

## 本地加载结果

- 两节点 EXL3 路径：`/data/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`
- 两节点 Engram 来源：`/data/models/DeepSeek-V4.1-Flash-1017`
- 模型清单：`4a450ef6c7c4c96d68daad649188fcf817549c77cbb4fa6c1221a2baea2e5ae8`
- 不再依赖 NFS 读取模型或 Engram。
- 最终 KV Cache 约 126 万 token；131072-token 请求理论并发约 9.62。

## 作者负载基线

本地 eager、无推测解码的中位数：

| 工作负载 | 解码 tok/s | TTFT 秒 | 旧 NFS 解码 tok/s | 改善 |
|---|---:|---:|---:|---:|
| poetry | 11.216 | 1.291 | 5.929 | +89.2% |
| code | 11.530 | 1.079 | 7.290 | +58.2% |
| incident | 11.269 | 2.121 | 8.062 | +39.8% |

原因：EXL3 权重本地化主要缩短装载；Engram 是推理热路径上的 file-backed 查表，本地化同时消除了 NFS 页缺失与远端 I/O 延迟，因此稳态吞吐也显著提高。

## DSpark 门禁

为支持 v0.32 的内置 drafter，修复了以下 EXL3 元数据缺口：

1. 识别运行时重映射的 `model.main_proj` 与 `model.layers.40+`。
2. 按实际 safetensors 形状区分 MTP 6-bit linears 与 4-bit routed experts。
3. 为未写入 `tensor_storage` 的 MTP `wo_a` 恢复 8-slice grouped 描述。
4. 为 stacked loader 增加权重名与形状诊断。

DSpark eager 成功启动并通过 323 冒烟，但最终指标为 189 accepted / 4911 drafted，即 3.85%；诗歌样本仅 5.70–6.06 tok/s，低于 11.216 tok/s 基线。因此保留代码与证据，运行配置禁用 DSpark。后续必须继续核对 MTP packed loader 数值正确性，不能只以“成功装载 147 个参数”作为通过标准。

## CUDA Graph 门禁

无 DSpark、`max_num_seqs=4`、捕获尺寸 `[1,2,3,4]` 时，模型完成装载并进入 Breakable CUDA Graph，但首个 PIECEWISE 捕获报告 empty CUDA graph，worker 随后退出。由于最小集合已失败，未扩大到作者的 `[1,2,3,4,6,8,12,18,24]`。最终配置保持 eager。

## 长上下文与并发

| prompt token | output token | 端到端秒 | 结果 |
|---:|---:|---:|---|
| 8192 | 8 | 52.775 | 通过 |
| 32768 | 8 | 130.437 | 通过 |
| 131000 | 8 | 509.958 | 通过 |

并发测试使用每请求 2048 prompt + 64 output：

| 并发 | wall 秒 | 聚合输出 tok/s | 结果 |
|---:|---:|---:|---|
| 1 | 15.633 | 4.094 | 通过 |
| 2 | 9.092 | 14.079 | 通过，当前最佳 |
| 4 | 31.046 | 8.246 | 通过，但性能回落 |

并发 4 的回落说明当前 grouped-prefill/调度仍有优化空间；建议生产初始并发限制为 2，再用真实业务长度分布调优。

## 最终运行态

- 主节点容器：`v032-dsv41-exl3-final-r0`
- 从节点容器：`v032-dsv41-exl3-final-r1`
- 两端 restart policy：`unless-stopped`
- 健康检查：HTTP 200
- 作者冒烟：`17 * 19 = 323` 通过
- 系统指纹：`vllm-0.32.0-tp2-37657eb2`
- 代码保持未提交状态，未 push、未创建 PR。

## 阶段完成判定

15 阶段的基础部署、EXL3、双节点 TP、DeepSeek V4.1、Engram、本地化、可靠基线、长上下文与并发能力已形成闭环。性能增强阶段中 DSpark 与 CUDA Graph 已执行门禁并得到可复现 no-go 结论；它们不是当前可交付能力。不可把“15 阶段验证完成”解释为“所有优化均成功上线”。镜像目前仍通过候选镜像加只读源码 overlay 运行，尚未生成无 overlay 的最终不可变镜像；该项应在 DSpark/Graph 代码取舍确定后再固化，避免把实验补丁烘焙为生产镜像。

## 2026-11-04 v35 覆盖结论

以上 DSpark no-go 结论已被后续实现取代。当前不可变镜像为
`vllm/vllm-openai:v0.32.0-dsv41-v35`，在 8.96（TP0/API）和 8.164
（TP1/headless）运行，使用 `--block-size 64`。日志确认以下路径同时生效：

- drafter 自有 EXL3 K4 head、TP2 row-parallel `main_proj`；
- 4096 行 Markov cache、本地词表分片 greedy argmax、融合 Markov argmax；
- `dspark_conf_feat_head.npz` 拟合置信度及 `conf+feat` 深度策略；
- EXL3 RowMM、SWA metadata batch、greedy verify local argmax、RDMA all-reduce。

作者脚本 `kit_bench.py` set-b、384 token、9 reps 的最终中位数为：code
99.055 tok/s、prose 53.702 tok/s、structured 134.356 tok/s；27/27 输出在各自
prompt 内完全一致。作者发布基准为 103.7 / 57.8 / 144.4，因此当前分别达到
95.5% / 92.9% / 93.0%，尚未达到“作者效果或更好”的性能门禁。固定深度 3/4/5
均劣于 `conf+feat`；关闭 Breakable CUDA Graph 得到 95.173 / 55.081 / 134.285，
也未形成整体改进，最终运行态已恢复自动 Breakable CUDA Graph 与 `conf+feat`。

功能与稳定性门禁通过：`/v1/models` HTTP 200；作者 greedy repeat check 为 5/5
完全一致。未把未达标的性能结果标记为完成，后续应针对 v0.32 每轮比作者基线多出的
约 2.4–3.3 ms 做 CUDA trace，对 scheduler/graph 边界与 drafter phase 定点优化。
