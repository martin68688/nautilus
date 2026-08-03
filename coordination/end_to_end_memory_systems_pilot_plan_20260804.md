# Experiment End2End：端到端记忆系统对比实验计划

- 日期：2026-08-04
- 独立目录：`/Users/haoming/Downloads/nautilus-exp-end2end`
- Git 分支：`codex/experiment-end2end-memory-systems`
- 基线提交：`f667132e1c67f0f53f26d7e22d0b4ef9b6dc671b`
- 当前阶段：Pilot，seed 暂定为 1

## 1. 为什么做这组实验

论文主实验改为完整端到端运行：让不同记忆系统从 Draft 一直参与到最终 Terminal Evaluation，再比较最终成绩、完成率、负迁移、时间和成本。

原来的细分实验不删除，但不再单独承担主结论：

- Experiment A 继续作为 Authority Gate 是否真实触发的机制证据；
- Experiment B/R 的 adoption、activation、stage routing 等指标，优先从完整运行日志中计算；
- 只有完整日志暴露出关键因果问题时，再选少量真实 decision points 做配对干预。

## 2. 第一轮要回答的问题

1. 使用记忆后，Agent 的最终合法成绩是否优于 No Memory？
2. 哪种记忆系统完成率最高、负迁移最少、最早得到合法结果？
3. Dynamic Hybrid 是否优于 Flat、Static 和 Reversed Router？
4. Dynamic Hybrid 与代表性竞品相比是否仍有优势？
5. 完整 RunForest 日志能否解释提升或失败来自哪条记忆、哪个阶段和哪次实际执行？

## 3. 系统矩阵

所有系统都必须执行完整 MLEvolve 流程，而不是只做离线检索。

| ID | 系统 | Draft | Improve | Debug | 作用 |
|---|---|---:|---:|---:|---|
| S0 | No Memory | 0/0 | 0/0 | 0/0 | 无外部记忆下界 |
| S1 | Flat Retrieval | 混合 Top-6 | 混合 Top-6 | 混合 Top-6 | 不区分阶段与粒度 |
| S2 | SOP-only | 6/0 | 6/0 | 6/0 | 只使用整体方法知识 |
| S3 | RunForest-only | 0/6 | 0/6 | 0/6 | 只使用具体运行经验 |
| S4 | Static Hybrid | 3/3 | 3/3 | 3/3 | 固定混合基线 |
| S5 | Dynamic Hybrid | 4/2 | 3/3 | 2/4 | 主要方法 |
| S6 | Reversed Router | 2/4 | 3/3 | 4/2 | 反向阶段对照 |
| C1 | GOME-style | 结构化执行反馈 + Success Memory | 同左 | 同左 | MLE Agent 直接竞品 |
| C2 | MACLA-style | 层级程序记忆 + reliability selection | 同左 | 同左 | 可靠性感知竞品 |
| C3 | RCR-Router-style | role/stage/token-aware routing | 同左 | 同左 | 阶段路由直接竞品 |

表中的 `SOP/RunForest` 是最终记忆配额，不是模型分数。

竞品优先移植其记忆机制到同一个 MLEvolve Host 中，保持 Agent、模型和预算一致。若官方 Full GOME 可以稳定运行，可额外作为外部完整 Agent 参考，但不能与同框架记忆消融混为同一种比较。

## 4. 任务与运行规模

第一轮使用四个已适配任务：

1. Aerial：图像二分类；
2. Leaf：多模态图像分类；
3. Denoising：图像恢复；
4. Taxi：表格回归。

第一轮规模：

```text
10 个系统 × 4 个任务 × 1 seed = 40 次完整运行
```

如果第三个竞品暂时无法忠实复现，可以先运行 9 个系统，共 36 次；不能悄悄用一个简单 heuristic 冒充论文方法。

Seed=1 只用于发现工程问题和初步趋势，不用于显著性结论。

## 5. 必须冻结的公平条件

除记忆系统外，固定：

- Agent/LLM 版本与基础 Prompt；
- 最大搜索步数、重试次数和 wall-clock budget；
- GPU 类型、GPU 数量和容器；
- Top-k 或等价的 Prompt memory token budget；
- ProtocolSpec、DataView 和 terminal evaluator；
- 代码版本、Memory Bundle 和模型资产；
- 随机种子与任务输入；
- 结果写回、超时和失败保留规则。

前七个内部系统使用相同 Host Authority、Protocol 和 evaluator 配置，避免把路由差异与权限配置差异混在一起。竞品的原始机制与所有适配改动必须单独记录。

禁止：

- 使用当前测试标签或未来运行记录；
- 失败后用其他系统分数补齐；
- 删除失败条件；
- 给某个系统额外步骤、重试或 GPU 预算；
- 因为结果不好而在正式矩阵中临时改方法。

## 6. 执行顺序

### Phase 0：版本与能力审计

新 worktree 来自已提交基线。先比较主目录和 Exp-R/Exp-B 中与 Dynamic Hybrid、Authority、RunForest、terminal evaluator 有关的最新实现，只移植实验真正需要且有测试支撑的修改。

不得直接复制整个 dirty worktree，也不得覆盖其他实验分支的用户改动。

### Phase 1：统一系统接口

每个系统使用同一个配置入口，例如：

```yaml
experiment: end2end_memory_pilot
memory_system: dynamic_hybrid
task: aerial-cactus-identification
seed: 1
top_k: 6
authority_enabled: true
```

运行清单应由一个冻结 manifest 生成，避免手工启动漏项。

### Phase 2：Smoke Test

先选一个小任务，让所有系统运行少量步骤，确认：

- 系统配置确实改变了记忆行为；
- memory-on 系统真实检索并进入 Prompt；
- No Memory 没有外部记忆；
- raw candidates、final Prompt、代码、runtime 和 terminal outcome 都能记录；
- 所有系统使用相同预算。

Smoke 不计入正式结果。

### Phase 3：40 次 Pilot

按 task-system block 生成 Job，任务内随机化系统启动顺序。每次运行使用不可变 run ID，并保存完整失败结果。

基础设施失败可以用完全相同配置重跑，但原失败记录必须保留并标记为 infrastructure retry。

### Phase 4：端到端汇总

先回答“哪个系统最终更好”，再解释原因。主表至少包含：

| 系统 | Aerial | Leaf | Denoising | Taxi | 完成率 | 平均排名 | 时间/成本 |
|---|---:|---:|---:|---:|---:|---:|---:|

不同任务的原始 metric 不直接平均；跨任务使用相对 No Memory 的方向归一化 delta 或任务内排名。

### Phase 5：日志机制分析

从完整运行记录中计算：

- Draft/Improve/Debug 的实际记忆组成；
- raw candidate、Prompt-visible 和 suppressed candidate；
- exposure → static adoption → runtime activation；
- time-to-first-valid；
- negative transfer；
- 无效 replay、未触发记忆和浪费的 GPU hours；
- 代表性成功与失败路径。

这些是端到端主结果的解释，不应代替主结果。

## 7. 主要指标

### 端到端主指标

- Completion Rate；
- Host-evaluated terminal metric；
- 相对 No Memory 的 normalized delta；
- Negative Transfer Rate；
- Time-to-first-valid；
- GPU Hours、wall time 和 token cost；
- task-level rank 与平均排名。

### 日志解释指标

- Stage-wise SOP/RunForest exposure；
- Prompt-visible memory count；
- Adoption/Runtime Activation Rate；
- Invalid/Suppressed Candidate Rate；
- 无效 replay 和失败类型分布。

## 8. Pilot 之后

完成 40 次运行后再决定正式矩阵。建议保留：

- No Memory；
- Flat Retrieval；
- Static Hybrid；
- Dynamic Hybrid；
- 最强竞品；
- 必要时保留 Reversed Router。

然后扩大到 3 seeds 和更多任务。Seed=1 的 Pilot 必须标记为 exploratory，不进行论文级显著性宣称。

## 9. Pilot 完成标准

只有同时满足以下条件才算完成：

1. 冻结 manifest 中的所有 task-system 条件都有 terminal outcome；
2. 失败和基础设施重试均保留；
3. terminal evaluator 独立且相同；
4. 每个 memory-on 条件均有真实 retrieval/Prompt 日志；
5. 每个 No Memory 条件均确认没有外部记忆；
6. 输出端到端总表、资源成本表和代表性运行路径；
7. 不把 Seed=1 结果写成统计显著性结论。

## 10. 一句话论文口径

> 我们在相同 Agent、任务、预算与评估器下，让七种内部记忆设计和三种代表性竞品完成完整 MLE 运行；先比较最终效用、可靠性与成本，再用真实 RunForest 日志解释阶段路由、经验采用、权限拦截和负迁移。
