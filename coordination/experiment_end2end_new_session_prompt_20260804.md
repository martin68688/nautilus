# 新会话执行 Prompt：Experiment End2End

请在独立 worktree `/Users/haoming/Downloads/nautilus-exp-end2end` 中执行端到端记忆系统对比实验。

当前分支：`codex/experiment-end2end-memory-systems`  
基线提交：`f667132e1c67f0f53f26d7e22d0b4ef9b6dc671b`  
实验计划：`/Users/haoming/Downloads/nautilus-exp-end2end/coordination/end_to_end_memory_systems_pilot_plan_20260804.md`

开始前先完整阅读：

1. `/Users/haoming/Downloads/nautilus-exp-end2end/AGENTS.md`；
2. 上述实验计划；
3. 当前 MLEvolve 的 memory、RunForest、Authority、ProtocolSpec、terminal evaluator 和 run-control 实现；
4. `/Users/haoming/Downloads/nautilus-exp-r` 中与 Dynamic Hybrid 端到端运行直接相关的实现和实验记录。

目标不是再做一个离线小分析，而是准备并执行完整端到端 Pilot：

```text
10 个系统 × 4 个任务 × 1 seed = 40 次完整运行
```

系统包括：

1. No Memory；
2. Flat Retrieval；
3. SOP-only；
4. RunForest-only；
5. Static Hybrid 3/3；
6. Dynamic Hybrid：Draft 5/1、Improve 3/3、Debug 1/5；
7. Reversed Router：Draft 2/4、Improve 3/3、Debug 4/2；
8. GOME-style；
9. MACLA-style；
10. RCR-Router-style。

任务先使用 Aerial、Leaf、Denoising、Taxi，seed=1。

请按以下顺序自主推进：

1. **先做版本差异审计。** 当前 worktree 只包含已提交基线；主目录和 Exp-R 有未提交的新代码。不要整体复制 dirty worktree。列出实验必需、经过测试的差异，选择性移植，并记录来源与理由。
2. **审查 10 个系统的可实现性。** 对三个竞品阅读原论文和可用代码。不能忠实复现时，明确标成 `*-style port`，冻结适配规范，不能用临时 heuristic 冒充官方方法。
3. **实现统一 MemorySystem 接口和冻结 manifest。** 除记忆系统外，固定 Agent、Prompt、Top-k/token budget、搜索步数、重试、GPU、容器、ProtocolSpec、Memory Bundle 和 terminal evaluator。
4. **补充测试。** 至少证明每个配置选择了正确系统，No Memory 没有外部记忆，SOP/RunForest 配额正确，运行日志包含 raw retrieval、Prompt exposure、代码、runtime 和 terminal outcome。
5. **先跑一个任务的全系统 Smoke。** Smoke 只检查完整链路，不进入正式结果。发现问题先修复并回归测试。
6. **生成 40 条正式运行 manifest 和 Job。** 任务内随机化启动顺序，使用不可变 run ID。若需要 NRP/Nautilus GPU，必须使用 `nrp-training` skill 做 preflight、提交、监控和结果拉取。
7. **运行并持续监控。** 基础设施失败可以同配置重跑，但必须保留原失败记录。不得删掉失败条件，不得插值其他系统分数，不得临时增加某个系统预算。
8. **先汇总端到端结果。** 输出每个系统在四任务上的 terminal metric、completion、normalized delta、negative transfer、time-to-first-valid、GPU/token cost 和平均排名。
9. **再从完整日志做机制分析。** 分析 stage-wise SOP/RunForest exposure、suppression、adoption、runtime activation、负迁移和代表性运行路径；不要让细节指标代替端到端主结论。
10. **交付可复核证据。** 保存配置、manifest、代码 hash、容器、Memory Bundle、运行日志、terminal evaluator 输出、失败记录、结果表和一份简单中文报告。

重要约束：

- 不要修改 `/Users/haoming/Downloads/nautilus`、`nautilus-exp-r`、`nautilus-exp-b` 或 `nautilus-exp-c` 中的用户文件；它们只可作为只读参考。
- 不要默认 Seed=1 可以支持显著性或 ICLR superiority claim；这一轮是 exploratory Pilot。
- 不要直接启动昂贵的 40-run 矩阵，直到版本审计、统一接口、测试和全系统 Smoke 通过。
- 遇到缺失信息时优先从仓库、历史 manifest 和已有实验记录中查证；只有会实质改变实验定义的问题才询问我。
- 每完成一个阶段都更新计划和状态，但继续推进，不要只写方案而不实现。

最终目标：形成一个可重复运行的端到端实验框架，并完成或可靠启动第一轮 40-run Pilot，使后续论文能够先报告整体效用，再从真实 RunForest 日志解释原因。
