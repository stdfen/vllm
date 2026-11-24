# vLLM 0.33.0 Tree-DFlash2 合入记录

分支为 `0.33.0-TensorFold-freetoken`，基线为 `release/0.33.0` 的
`4ba787ec9959e566d3b587a796dff07673324da5`。源码提交为
`4998676ce7e1400517ec3646b3cce6cfd1d01757`，树
`9ed7a710ec76d4f6f45423f4877d6f99d3915327` 与 GB10 最新性能和正确性门禁使用的源码快照一致。
Tree-DFlash2 是 V2 Model Runner 的可选路径。

## 合入的执行路径

| 技术点 | 实现及当前配置 |
| --- | --- |
| 草稿生成与树规划 | 复用 Qwen3.8 DFlash2 草稿权重。TensorFold 用上下文环、已接受行的 hidden-state taps、按成本分配的 best-first 候选树和实际树宽推进每轮。服务用 K=15，单请求最多 15 个草稿候选。 |
| 单次目标验树 | 按父子拓扑打包树行。原生 TensorFold 路径执行目标 embedding、NVFP4/FP8 投影、GDN、Attention、MLP 和 LM head，按祖先掩码验证所有候选。旧 `replay`、`packed`、`graph` 模式也保留在源码中。 |
| Attention 与缓存接口 | 验树从 vLLM 分页表读已提交的 FP8 KV，通过层级 scale 解码。树尾按配置计算，C1 启用 FP8 树尾舍入。目标请求调度、缓存页和 GDN 槽位仍由 vLLM 管理。 |
| GPU 分支选择和提交 | C1 及实际 2–8 请求批次在 GPU 选择接受路径，仅提交已接受行的 KV、GDN recurrent 和卷积历史。vLLM 用接受和拒绝计数回收未接受位置，下一轮草稿只消费已接受行的 taps。 |
| C1 Graph 与投影 | 启用定容量验树 Graph、草稿 Graph、提交 Graph、GPU 接受、设备槽位直刷、异步草稿 ID 上传、GDN 目标槽位 replay 覆盖和 C1 QMM split。 |
| 多请求和长前缀 Graph | 保留原 2K/C8 满宽树分桶和 8K/C8 长前缀专用 Graph。新增 2–8 请求短前缀通用分桶，Attention 5/6、7/8 个分块分别共用 6、8 的容量，按 GDN 暂存槽位分桶，并在 replay 前更新位置、页表和槽位。每种实际批大小最多新增一份 Graph。 |
| Prefill 和回退 | 编译 vLLM prefill，含 prefill 的混合批走 vLLM 常规目标前向和线性验证。没有匹配的验树图时保留 eager 路径。运行时配置关闭外层 CUDA Graph，树验证、草稿和状态提交使用专用 CUDA Graph。 |

当前组合的配置见 [GB10 speculative JSON](../../../../docker/tensorfold-freetoken/speculative-gb10.json)。
实验候选限宽、固定宽度、侧流重叠和 fused conv 等选项按详细配置文档选择。
当前归档的性能数据按上述 JSON 的开关记录。[Tree-DFlash2 配置说明](../tree_dflash2.md)

## 验证和已知问题

- Windows 的树规划及候选逻辑测试共 63 项通过。两份测试文件为
  `test_dflash2_tree.py` 和 `test_tensorfold_tree_plan.py`。
- GB10 FP8 Attention CUDA 和 28 项 CUDA 回归通过。新分桶在门禁实测中重放 C2、C4、C6，
  对应批大小的重放为 29/69、42/70、65/72。
- 服务级逐 token 门禁为 14/40 对一致。最早在第 3 个 token 分叉。两条 2K/C1
  分别在第 20、38 个 token 分叉，均未进入新分桶。历史 A/A 也出现过第 3 个 token 的差异，
  首个发生偏差的算子和状态仍待定位。归档 SHA256 为
  `f90e89a43ce3d73c19a55f5f99d8269e1ac257eda723921f33e925b156f04d2a`，
  工作区实验编号为 `20261124-vllm-tensorfold-batch-graph-buckets-aba-gate`。

## 当前版本的单变量性能

GB10 ShareGPT，2048/8192 输入、1024 输出、C1/C8、K=15。A1/B/A2 使用同一构建和编译 prefill，
只有 B 打开新增多请求分桶。216/216 正式请求完成。表内数值顺序为 A1 / B / A2，
括号内为 A1/A2 漂移。

| 场景 | TPOT p50（ms/token） | TPOT 变化（漂移） | 输出 tok/s | 吞吐变化（漂移） |
| --- | ---: | ---: | ---: | ---: |
| 2K/C1 | 22.22 / 20.24 / 20.19 | −4.54%（9.55%） | 49.77 / 52.68 / 52.77 | +2.75%（5.85%） |
| 2K/C8 | 43.26 / 42.39 / 40.91 | +0.73%（5.59%） | 140.63 / 146.55 / 143.63 | +3.11%（2.11%） |
| 8K/C1 | 21.04 / 20.87 / 20.81 | −0.27%（1.07%） | 39.63 / 39.81 / 39.82 | +0.20%（0.47%） |
| 8K/C8 | 60.15 / 59.67 / 55.63 | +3.08%（7.82%） | 90.24 / 89.99 / 90.76 | −0.57%（0.58%） |

2K/C8 总验树重放率为 45.55% / 84.37% / 48.28%，B 的新分桶重放 470/613 步。
该格 TTFT p50 下降 3.47%，漂移为 2.47%。四格 TPOT p50 差异均在漂移内。
2K/C8 服务端 prefill 增加 0.88%，漂移为 0.50%。2K/C1 TTFT p95 增加 3.69%，漂移为 1.36%。
按原实验冻结的全场无退化判据，该轮验收未通过。原始回执在工作区实验
`20261124-vllm-tensorfold-batch-graph-buckets-aba-perf`，归档 SHA256 为
`de216068fb7f314e8590c490bf442be36488739b2800bbdc80d78e926ab345d1`。

与两个外部基线的同场 A–B–A 均使用较早的组合。测试身份和结果见
[TensorFold 0.6.0 对照](performance_tensorfold.md)及[原生 vLLM 0.33.0 对照](performance_vllm.md)。
