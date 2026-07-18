# RunForest 复合记忆系统低成本 Benchmark 计划

## 1. 研究目标

本 benchmark 用来检验两套相互独立、但在运行时衔接的机制：

1. **三角色根节点策略**：`coldstart_baseline`、`memory_reproduction`、
   `novel_exploration` 是否比同质化的三个根节点更有效；
2. **Stage Hybrid v2**：Draft、Improve、Debug 是否能使用正确粒度的 SOP/Tree
   记忆，并比无记忆、平面记忆和旧 Legacy Gateway 更有效。

它不要求每次开发都运行完整 MLE-Bench。评估分为离线决策、代码采纳、协议修复和
短时真实执行四层，逐层回答：

```text
角色是否正确
    -> 检索是否正确
    -> Agent 是否真正采纳
    -> 代码是否安全可执行
    -> 相同预算下是否产生更好的可信结果
```

本 benchmark 是 MLE-Bench 的低成本代理，不得把结果表述为“已经证明全量
MLE-Bench 优越”。

## 2. 预注册研究问题

### RQ1：三角色是否有独立价值

在总根节点数、LLM 调用、GPU 时间和搜索步数相同的前提下，异质的
Baseline/Replay/Novel 三角色，是否比三个同质根节点提高：

- 至少一个可信候选的成功率；
- 方法族覆盖率；
- time-to-first-trusted-result；
- 最终任务内归一化成绩。

### RQ2：Stage Hybrid 是否有独立价值

固定三角色结构，只改变 Novel 分支的记忆策略时，Stage Hybrid v2 是否比：

- 无外部记忆；
- Flat MiniLM + 同一 clean gate；
- Tree-only；
- Legacy Gateway；

更准确地选择阶段匹配的经验，并提高实际代码采纳和短时执行结果。

### RQ3：两套机制是否产生互补效应

三角色组合与 Stage Hybrid 的组合收益，是否大于两者各自独立使用的收益。主分析只
声称“角色组合的整体效应”；Baseline 与 Replay 的单独贡献通过额外的组合分解条件
估计，不能从 Full 对全 Novel 的单一差值反推。交互通过预注册的
`portfolio x memory_stratification` 四格对照检验，不通过挑选最佳任务或 seed 回答。

### RQ4：安全收益是否以牺牲有效方案为代价

系统是否能把污染、失败、protocol-biased、blocked replay 排除在执行、排名和
positive memory 之外，同时不过度拒绝 clean 成功经验。

### RQ5：Replay 修复是否优于普通 Debug

对于已知存在协议问题的历史方法，staged protocol repair 是否比普通 Debug 或直接
丢弃更能同时实现：协议 clean、方法保真和可信执行。

## 3. Benchmark 总体结构

| 层级 | 测什么 | 是否调用 LLM | 是否训练 GPU | 预计规模 |
|---|---|---:|---:|---:|
| T0 合同层 | 角色隔离、门禁、provenance、RRF 权重 | 否 | 否 | 全量单测 |
| T1 决策层 | Draft/Improve/Debug 检索与路由 | 否或固定生成 | 否 | >=120 episodes |
| T2 采纳层 | Agent 是否按检索内容写出对应方案 | 是 | 否 | >=60 episodes |
| T3 Replay 层 | 五阶段协议修复与保真 | 是 | 仅最终 clean 候选 | >=48 defect cases |
| T4 微执行层 | 相同预算下的可信训练结果 | 是 | 是 | 12 tasks x 3 seeds |

T0-T3 用于快速开发回归；T4 只在候选版本冻结后运行。这样多数代码修改无需重跑
全量训练，而最终结论仍包含真实执行证据。

## 4. 主要实验条件

所有条件固定相同的 Agent 模型、温度、prompt 基座、工具、三个根槽位、总搜索步数、
最大模型调用数、最大输出 token、训练时限和 GPU 型号。输入 memory token 是方法本身
的成本，不通过给无记忆条件填充无效文本来伪造相等；运行时记录实际输入/输出 token、
延迟和 API 成本，并设置相同 wall-clock/GPU 上限与每条件成本上限。

### 4.1 主四格因子条件

`N` 表示 Novel，`B` 表示 cold-start Baseline，`R` 表示 Replay。所有条件始终只有三个
根槽位。

| ID | 根节点组合 | Novel 记忆 | 用途 |
|---|---|---|---|
| F00 | N + N + N | Flat MiniLM + 同一 clean gate | 同质角色 + 平面记忆 |
| F01 | N + N + N | Stage Hybrid v2 | 同质角色 + 分阶段记忆 |
| F10 | B + R + N | Flat MiniLM + 同一 clean gate，仅 N 可用 | 三角色 + 平面记忆 |
| F11 | B + R + N | Stage Hybrid v2，仅 N 可用 | 完整系统 |

这四格直接估计角色组合、Stage Hybrid 和二者交互。Replay source 只在 F10/F11 出现，
所以 `F11-F01` 只能解释为整个 B+R+N portfolio 的效应，不能声称是 Baseline 或 Replay
任一角色的独立贡献。

### 4.2 角色组合分解

固定 Novel 使用 Stage Hybrid，再加入两个等槽位的组合条件：

| ID | 根节点组合 | 可识别的比较 |
|---|---|---|
| P0 | N + N + N | portfolio 参考 |
| P1 | B + N + N | `P1-P0`：用一个 Baseline 替换一个 Novel |
| P2 | R + N + N | `P2-P0`：用一个 Replay 替换一个 Novel |
| P3 | B + R + N | `P3-P1/P2`：组合增量；等同完整 F11 |

P0-P3 仅在存在冻结 replay manifest 的 replay-eligible 任务上比较。没有合法 replay
source 的任务不伪造 R，也不把 fail-closed 当成方法失败；它们进入单独的 no-replay
stratum。正式 P0-P3 比较要求至少 8/12 个任务具有预先冻结、通过 provenance 检查的
replay source；不足 8 个时不得报告 Baseline/Replay 独立贡献，必须先扩展 manifest 或
把角色分解降为 exploratory。No-replay stratum 单独报告 B/N 条件、任务数和 CI，不与
replay-eligible stratum 合并成一个平均数；只有两个 stratum 都有至少 8 个 task cluster
时才做 eligibility interaction/meta-regression。P1/P2 默认跑 T1-T3，只有 portfolio
总效应在开发集非退化时才进入 T4。

### 4.3 诊断性与强基线对照

| ID | 条件 | 目的 |
|---|---|---|
| D1 | 三角色 + Legacy Gateway | 防止把旧词面门禁当成 Stage Hybrid |
| D2 | 三角色 + Tree-only | 测 Tree 单通道贡献 |
| D3 | 三角色 + SOP-only | 测 SOP 单通道贡献 |
| D4 | 三角色 + Stage Hybrid，去掉 stage gate | 测阶段门禁贡献 |
| D5 | 三角色 + Stage Hybrid，去掉 task identity | 测任务身份贡献 |
| D6 | 三角色 + Stage Hybrid Flat-Twin | 仅作为 geometry 消融 |
| D7 | Stage Hybrid 打分但关闭 safety gate | 只在离线候选池测安全贡献，禁止执行/写回 |
| D8 | 预先限定为 clean universe，再运行无 safety gate 的 Stage Hybrid | 在相同可采纳集合上测语义/阶段排序贡献 |
| B0 | 三个 cold-start 根 + 简单顺序演化 + greedy best-valid | 强无记忆执行基线 |
| B1 | 从 clean memory 中等量随机抽取 | 随机安全记忆基线 |
| B2 | Flat MiniLM + 同一 clean gate | 强平面检索基线 |
| B3 | 对所有 clean 可兼容候选做 greedy best-valid 选择 | 简单选择策略基线 |
| B4 | 同一任务内的 clean candidate ensemble | 仅在预测可组合时使用，不强行跨任务统一 |
| O1 | 相同 memory/candidate universe 的安全 oracle upper bound | 测过度过滤与可达上限 |

D1-D8 默认只跑 T1-T3；D7 的不安全候选永不上 GPU、不排名、不写回。B0、B2 是 T4
必跑基线，B1/B3/B4 在适用层报告。Geometry 结果不得自动升级为双曲几何主张。

B4 的可组合性在 task manifest 中预注册：候选必须针对同一 task/holdout、使用完全相同
的 sample ID 与顺序、相同 target schema/label order、兼容的 prediction shape，并能在
不读取 holdout label 的情况下融合。分类概率还必须具有相同类别顺序并通过数值有效性
检查。满足条件的任务少于 3 个时，B4 只作案例附录，不进入跨任务主比较。

### 4.4 Replay 专项条件

| ID | 条件 |
|---|---|
| R0 | 发现协议问题后直接 reject/drop |
| R1 | 普通 Debug Agent，自由修复 |
| R2 | staged protocol repair，无 preservation contract |
| R3 | 完整 staged protocol repair + preservation + runtime provenance |

Replay 对照与 Novel 检索对照分别统计，禁止把 repair 成功率混入 Novel 的 strategy
retrieval 指标。

## 5. 任务与 Episode 设计

### 5.1 任务覆盖

至少覆盖 6 个任务 family，每个 family 至少 2 个独立任务：

1. 文本分类；
2. 图像二分类/多分类；
3. 图像恢复；
4. 表格多分类；
5. 表格回归；
6. group-aware 或 time-aware 任务。

推荐总计 12 个任务。Spooky 只能是其中一个文本任务，任何主结论都不得只由 Spooky
支持。任务选择、数据版本、metric 方向、训练预算和排除理由在运行前冻结。

### 5.2 决策 Episode

每个任务至少构造以下 6 个 episode：

- 2 个 Draft/L1 整体路线决策；
- 1 个 model_design/L2 决策；
- 1 个 Improve 决策；
- 2 个 Debug/L3 决策。

每个任务扩展到至少 10 个 episode，使 12 个任务共至少 120 个 held-out episode，并保证
每个 family 至少 20 个。每个 episode 只描述当前任务状态和决策需求，
不得包含目标 SOP、目标节点、历史 run ID、metric 后验或父子轨迹坐标。

### 5.3 Replay 缺陷矩阵

至少构造 8 类协议缺陷，每类覆盖至少 3 个任务 family、2 个实现变体：

1. preprocessing/vectorizer 在 validation/test 上 fit；
2. early stopping 与最终报告复用同一 validation；
3. 普通 validation prediction 冒充 OOF；
4. ensemble 权重在最终 holdout 上搜索；
5. group/entity 跨 split 泄漏；
6. 时间顺序被随机 split 破坏；
7. target encoding 或特征选择看见 validation label；
8. deduplication 在切分后执行导致近重复跨集合。

每个 defect case 都包含一个 frozen preservation contract，但不把正确修复代码提供给
Agent。缺陷注入器、静态 oracle 和 runtime oracle 必须分离实现，防止 evaluator 与
generator 共享同一模板答案。

## 6. 数据和记忆隔离

### 6.1 三重切分

所有 run、SOP、Tree evidence 和 episode 按来源 run 做不可交叉的三重切分：

- `memory_train`：允许进入可检索记忆；
- `benchmark_dev`：允许开发和调参；
- `benchmark_test`：冻结后一次性盲测。

同一个 source run、父子 transition、代码 hash 或近重复 SOP 不得跨 split。先按
`source_run_id` 分组，再按时间切分；禁止先按单条 SOP 随机切分。

### 6.2 两种泛化轨道

- **Same-family new-run**：任务 family 已见，但 run 和节点完全未见；
- **Cross-task transfer**：完整 task ID 未进入 memory，只允许同 family 的其他任务提供
  SOP/Tree 证据。

两个轨道单独报告。Replay 因定义上需要指定历史 source，单独使用 replay split，不与
Novel 的 cross-task retrieval 混合。

### 6.3 最终 holdout 隔离

T4 的 outer holdout 由外部 evaluator 持有：

- Agent 进程只能读取 outer train；
- holdout 路径不进入 prompt、环境变量或工作目录；
- 训练结束后由独立评分进程加载冻结 artifact；
- 每个 candidate 只允许一次最终评分；
- Agent 不接收单个 holdout 分数后继续修改代码。

因此手工固定 split 只解决“看不到最终答案”；fold-local fit、OOF、selection freeze 等
内部协议仍由 runtime provenance 和专项 oracle 检查。

## 7. Gold 与人工标注

### 7.1 T1 Gold

每个决策 episode 使用 multi-gold，而不是把历史实际 child 当成唯一答案：

- 可接受的 `method_family` 集合；
- 可接受 SOP 集合；
- 必须/禁止出现的 abstraction level；
- 允许的 clean Tree evidence；
- 不可接受的失败、污染和阶段错位候选。

### 7.2 盲标流程

- 两名不查看方法输出的独立标注者；
- 标注候选顺序随机化，不显示系统名称；
- 分歧由第三人 adjudication；
- 报告 raw agreement 和 Krippendorff's alpha；
- `alpha < 0.67` 时只能报告诊断结果，不能开放 relevance claim。

### 7.3 代码采纳 Gold

采纳不能只依赖 Agent 自报。对生成代码做结构化提取并与 prompt trace 对照：

- checkpoint/model family；
- preprocessing/feature family；
- split/fold；
- loss/optimizer/scheduler；
- ensemble 成员与权重选择位置；
- 修复 API 与 runtime provenance 事件。

抽取器不确定时进入人工盲审，不把字符串出现直接等同于实际采纳。

## 8. 指标

### 8.1 角色层

- `Role Slot Accuracy`：三个根槽位是否严格按协议生成；
- `Role Isolation Violation`：Baseline/Replay 接收 Novel L1/L2 的比例；
- `Replay Source Accuracy`：是否命中指定历史 source 和代码 hash；
- `Novel Family Diversity@3`：Novel 候选方法族覆盖；
- `Duplicate Root Rate`：多 worker 是否生成重复根角色。

### 8.2 检索层

- graded `nDCG@10` 和 `Adoption AP@10`；
- `Strategy Precision@3`；
- `Distinct Method Families@3`；
- `Detail Intrusion@3`；
- `Stage Routing Accuracy`；
- `Task Compatibility@k`；
- `Clean Evidence@k`；
- `Eligible Recall@k` 和 `Clean Rejected Rate`，防止系统靠过度拒绝取得零 escape；
- `Blocked/Failed/Protocol-biased Escape@k`；
- `Provenance Completeness`。

### 8.3 采纳层

- `Strategy Adoption Precision/Recall`；
- `Code-Family Alignment`；
- `L2 Timing Accuracy`：L2 是否仅在 Novel model_design 前触发；
- `Frozen-Route Violation`：Improve/Debug 是否擅自更换 Draft family；
- `Repair Feedback Utilization`：上一轮拒绝原因是否被下一轮代码实际处理。

### 8.4 Replay 层

- clean repair success；
- static false acceptance / false rejection；
- runtime provenance pass；
- preservation pass；
- stage completion 和 attempts-to-clean；
- intermediate GPU execution rate，目标为 0；
- invalid ranking / positive writeback rate，目标为 0；
- repaired metric retention，相对同方法合法参考实现的性能保持率。
- safety oracle recall：可修复且 clean 的方法是否被错误永久丢弃。

### 8.5 微执行层

两个共同主要终点，不合并成一个可任意调权的总分：

1. **Trusted Success Rate**：在预算内至少产生一个同时满足 static clean、runtime clean、
   preservation clean、rank eligible 且有有效 metric 的候选；
2. **Task-Normalized Final Score**：把所有任务 metric 转为 higher-is-better，再使用预先
   冻结的简单基线分布中位数和 IQR 标准化。

同时报告原始任务 metric、best-valid、time-to-first-trusted、time-to-best、GPU-hour、
LLM tokens、API cost、失败节点数和 trusted improvement 的单位成本。

不得使用测试臂自身的均值/方差做归一化，也不得跨任务直接比较原始 LogLoss、accuracy
或 RMSE。

## 9. T4 固定预算协议

### 9.1 每个 task-seed-condition 的资源

- 三个根槽位；
- 相同最大搜索步数；
- 相同最大 LLM 调用与输出 token；输入 token 作为处理成本记录，不做人为 padding；
- 20 分钟 GPU 上限作为默认微执行预算；
- 同一 GPU 型号与容器镜像；
- 相同训练数据、outer holdout 和 evaluator；
- 不允许某条件因失败自动获得额外总预算。

若某模型无法在 20 分钟完成，记录 timeout 和已有 checkpoint，不临时为完整系统延长
时间。可另设统一 40 分钟 sensitivity analysis，但所有条件必须一起重跑。

### 9.2 规模与成本

主 T4a 实验运行 B0、F00、F01、F10、F11：
`12 tasks x 3 seeds x 5 conditions = 180` 个微执行单元。按每单元最多 20 分钟计，
最坏约 60 GPU-hours。P1/P2 属于可选 T4b，只有 T4a 显示非退化 portfolio 效应且功效
模拟允许时才增加，避免在基础机制尚未成立时扩大 GPU 消耗。

若资源不足，允许预注册的最小版本：

- 8 tasks，仍覆盖 6 family；
- 3 seeds；
- B0、F00、F10、F11 四个条件；
- 共 96 单元，最坏约 32 GPU-hours。

最小版本只能作为系统验证，不能支持广泛任务泛化主张。

正式开跑前用 T2/T3 dev pilot 实测每 episode 的 LLM 调用、输入/输出 token、失败重试、
人工标注秒数和工程吞吐，再冻结以下硬预算：

- GPU-hours 上限；
- LLM API 总金额和每条件金额上限；
- 每个 episode 最大调用/重试次数；
- 盲标候选池大小和预计 30-50 annotator-hours；
- benchmark 工程开发和复核工时单独记录，不混入方法运行成本。

若 pilot 显示 T2/T3 成本超过预算，优先删除 D1-D6 架构消融；保留 B0/B2、D7/D8、O1、
任务 family 和 test gold 质量。

## 10. 统计分析

### 10.1 分析单位

统计单位是 `task x seed` 的配对 episode，不是检索候选、搜索节点或 fold。多个节点不得
伪装成独立样本。

### 10.2 主要检验

- Trusted Success Rate：task-blocked 配对 permutation/cluster bootstrap 为主；
- Task-Normalized Final Score：task-blocked paired permutation/cluster bootstrap 为主；
- 固定对比：portfolio、memory stratification 及二者交互；混合模型仅作 sensitivity；
- 主要比较：F11-B0、F11-F10、F11-F01；
- Holm 校正三个主要比较；
- 同时报告配对 bootstrap 95% CI 和每任务散点，不只报告 p-value。

12 个 task cluster 对复杂混合模型仍偏少，因此不把模型渐近 p-value 作为主证据。
任务数优先于重复 seed 数，用 leave-one-task-out、逐任务 forest plot 和去除 Spooky 后的
sensitivity analysis 检查单任务驱动。

### 10.3 缺失与失败

- crash、timeout、无 metric 不得从分析中删除；
- Trusted Success 记为失败；
- final score 使用预注册 worst-valid score；
- 另做 survivor-only sensitivity analysis，但不能作为主表；
- 所有重跑必须有 reason code，不能静默替换失败 seed。

### 10.4 功效与停止规则

使用冻结前 dev pilot 的任务间、seed 间方差做层级功效模拟，报告在既定 12x3 设计下的
minimum detectable effect；不预设一个方便的 d=0.5，也不使用 test split 调整样本量。
预注册可接受上限为标准化 MDE <=0.8。若 12x3 的 MDE >0.8，则优先扩展到 16 个任务；
如果资源不足，T4 只能标记 exploratory，不能开放 superiority claim。
正式测试没有“看到显著就停止”。如果 CI 仍跨过最小实际效应，结论写作 inconclusive，
而不是继续挑任务。

## 11. Claim Gates

### 11.1 机制 claim

只有满足以下条件才允许声称“机制按设计工作”：

- test episodes >=120、每 family >=20，且覆盖 >=6 task families；
- 正式 test 中 `insufficient_strategy_coverage` episode = 0；覆盖不足必须先扩充冻结
  memory snapshot，不能用细节 SOP 补位或从主要指标中静默删除；
- 两人盲标与 adjudication 完成，alpha >=0.67；
- Role Isolation Violation = 0；
- blocked/failed/protocol-biased escape = 0；
- clean rejected rate 和 eligible recall 完整报告，并对照 O1 oracle；
- provenance completeness = 100%；
- 预注册主要指标和所有失败均完整报告。

### 11.2 小预算下游 claim

只有在机制 gate 通过后，才允许声称“相同小预算下优于 baseline”：

- >=10 个任务、每臂 >=3 seeds；
- F11 相对 B0 的 Task-Normalized Final Score 配对 CI 下界 > 0；
- Trusted Success Rate 不低于 B0 超过预注册的 5 个百分点 non-inferiority margin；
- 无显著增加 leakage escape 或无效 positive writeback；
- GPU、token、API 成本完整披露。
- 正式设计的标准化 MDE <=0.8；否则只允许报告 exploratory effect estimate。

### 11.3 禁止自动升级的 claim

即使以上通过，也不能自动声称：

- 全量 MLE-Bench 排行优势；
- 所有任务类型普适；
- Poincare/双曲几何优于 Flat-Twin；
- 长预算最终成绩一定更好；
- Replay 历史最高分已被合法复现。

这些需要各自独立实验。

## 12. 工程实现结构

建议新增独立目录，避免修改生产搜索代码来迎合 benchmark：

```text
paper-skills/eval_composite_memory/
  manifests/
    task_manifest_v1.yaml
    condition_manifest_v1.yaml
    memory_snapshot_manifest_v1.json
    claim_gates_v1.yaml
  episodes/
    decision_dev_v1.jsonl
    decision_test_v1.jsonl
    replay_defects_v1.jsonl
  annotations/
    blind_packet_a.jsonl
    blind_packet_b.jsonl
    adjudicated_gold_v1.jsonl
  runners/
    run_offline_decisions.py
    run_agent_adoption.py
    run_replay_repairs.py
    run_micro_execution.py
  evaluators/
    score_retrieval.py
    score_adoption.py
    score_protocol.py
    score_downstream.py
    statistical_analysis.py
  reports/
    composite_benchmark_v1.json
    composite_benchmark_v1.md
```

每次运行保存：代码 commit、memory graph SHA、taxonomy SHA、episode SHA、condition、seed、
模型、prompt hash、容器、GPU、预算、完整 navigation trace、adoption trace、authority
receipt、runtime provenance、metric 和失败 reason code。

## 13. 防止 benchmark 被系统“学会”

- benchmark test prompt 和 gold 不进入检索 corpus；
- 任何 test episode 的生成代码不得写回正式 positive memory；
- 调参只能使用 dev split；
- test 只允许冻结版本执行一次；
- benchmark 版本升级必须更换 SHA 和 changelog；
- 报告所有分析版本，禁止只保留最有利的一版；
- evaluator、oracle 和 Agent prompt 分仓或至少分模块，避免共享修复模板；
- D7 使用独立离线 evaluator 进程；该进程不导入 executor、不挂 GPU/训练凭据、没有
  memory 写权限，只读取不可变候选快照并输出 ranking；
- 公开失败案例和 per-task 结果，使平均分不能掩盖单任务灾难。

## 14. 实施顺序

### Phase 0：低成本非退化与 evaluator 验证

1. 使用仅属于 dev 的 20 个新 episode 检查 gold、candidate pool、指标是否有区分度；
2. 当前 2-query 三角色测试的 MRR=0 作为负面先导结果完整保留，但不把 N=2 当成系统
   无效证据；
3. 复核现有 29-query Stage Hybrid 结果的盲标可行性；其 CI 跨零，所以只用于估计方差，
   不作为扩展 benchmark 的成功前提；
4. Phase 0 的定量通过项为：source-run/ID 泄漏为 0；每个 episode 至少有 3 个可接受
   gold 和 20 个 eligible distractors；安全 oracle 的 nDCG@10 >=0.90；随机基线
   nDCG@10 <=0.50 且 oracle-random 配对均值差 >=0.30；至少 30% episode 的非 oracle
   方法 Top-3 不完全相同；两名 pilot 标注者的
   alpha >=0.67；成本 p95 不超过拟议单 episode 上限；
5. 泄漏、gold、oracle、候选池、标注可靠性任一失败属于 benchmark defect，必须修复后
   重做 Phase 0。Ours 输给 baseline、方法差值很小或成本较高属于方法结果，不允许借此
   修改 gold 或取消正式盲测；
6. 若所有方法因 evaluator 缺陷恒为零，先修 benchmark，不进入正式 test；若 evaluator
   非退化但 ours 不占优，仍进入冻结 test 做预注册否证。

### Phase A：冻结协议与 manifests

1. 冻结 F00-F11、P0-P3、D1-D8、B0-B4、O1 和 R0-R3；
2. 冻结 12 个任务、metric 方向和微执行预算；
3. 生成 memory snapshot 并记录所有 SHA；
4. 建立 source-run 级 split 检查器；
5. 写 claim gate 配置，禁止报告脚本自行决定结论。

### Phase B：完成 T0-T1

1. 新建 >=120 个 held-out episodes；现有 29-query 只作 dev 方差参考；
2. 加入 Draft/L2/Improve/Debug 分层；
3. 完成两个盲标 packet；
4. 跑四格 F00-F11、P0-P3、D1-D8 和 B/O 基线；
5. 报 clean rejected rate、eligible recall 和 O1 upper bound；
6. 若角色隔离或安全 escape 非零，停止，不进入 GPU。

### Phase C：完成 T2-T3

1. 固定 Agent 模型与 prompt；
2. 生成代码但不训练，检查 adoption/code alignment；
3. 运行 48+ replay defect cases；
4. 对 false acceptance/false rejection 做逐例审计；
5. 只有 R3 安全和保真 gate 通过才进入 T4。

### Phase D：完成 T4

1. 先跑 2 tasks x 2 seeds 的 infrastructure smoke，不进入正式统计；
2. 清空 smoke 输出，不写入 memory；
3. 冻结代码和容器；
4. 并行运行正式 `task x seed x condition` 矩阵；
5. 由独立 evaluator 一次性评分 outer holdout；
6. 生成完整 per-task、成本、失败和统计报告。

## 15. 成功、失败与决策规则

### 成功

如果 F11 在机制 gate 全部通过的情况下，相对 B0/F10/F01 获得稳定的可信成功率和归一化
成绩改善，可以支持：

> B+R+N 角色组合与阶段化安全记忆，在固定小预算下提高了 MLE Agent 产生可信改进的
> 概率。

### 部分成功

如果检索、采纳和安全显著改善，但微执行成绩无显著变化，应把贡献定位为
protocol-safe decision support，而不是性能提升方法。

### 否证

出现以下任一结果，应收缩或放弃主张：

- F11 不优于 F10：Stage Hybrid 在三角色内没有独立下游价值；
- F11 不优于 F01：B+R+N portfolio 没有整体下游价值；
- P1/P2 不优于 P0：不能声称 Baseline 或 Replay 各自有独立贡献；
- F11 仅靠更多 token/GPU 获胜且不在预注册成本-Pareto 上：不能称为算法收益；
- Improve 阶段持续低于 Flat Memory：阶段权重或 Tree 投影需要重做；
- R3 无法同时提高 clean repair 与 preservation：staged repair 主张不成立；
- leakage escape 非零：安全主张直接关闭。

## 16. 与现有结果的关系

- 现有 29-query strict benchmark 保留为 T1 的开发诊断，不作为最终 gold；
- `Stage Hybrid v2 = 0.4522` 对 `Legacy = 0.3543` 只是点估计，CI 跨零；
- 现有 2-query Layered Strategy 结果只作为单元测试；
- 现有 three-role test 的两条 test episode 对 Tree-only 和旧 Stage Hybrid 都得到 MRR=0，
  作为 underpowered negative diagnostic 保留，不解释为三角色已被否证；
- 旧 Tree-only MRR 高于旧 Stage Hybrid 的负结果继续保留；
- Spooky replay 的多轮失败作为 T3 defect case 来源，不作为成功证据；
- benchmark 完成前，`offline_retrieval_claim_allowed` 和
  `online_downstream_claim_allowed` 继续保持 false。

## 17. 最小交付物

正式宣称 benchmark 完成前，必须同时提供：

1. 冻结 manifests 和所有 SHA；
2. source-run 级无泄漏 split 报告；
3. 两名盲标者和 adjudication 报告；
4. T0-T4 原始 receipts 与 per-episode 结果；
5. F00-F11、P0-P3 与 B0-B4 的等机会预算及实际成本证明；
6. R0-R3 的逐缺陷结果；
7. 完整失败和重跑清单；
8. 主统计、敏感性分析和成本表；
9. 自动生成的 claim gate 报告；
10. 一份明确列出“可以说什么、不能说什么”的最终研究结论。

## 18. ClaudeAgent 独立审查与处置

独立只读审查 session：`ff2cf272-7cfe-411b-9517-0a3c8d2ee451`。ClaudeAgent 的原始
verdict 是：**设计可识别，但必须修订后实施**。

| 审查意见 | 处置 |
|---|---|
| Replay 只存在于三角色条件，会混淆角色与 replay 收益 | 接受。新增 F 四格和 P0-P3 组合分解；主比较只声称 portfolio 总效应。 |
| 冷启动、Replay、Novel 的输入 token 天然不同，不能声称 token 完全相等 | 接受。改为等 wall-clock/GPU/调用/输出上限，实际输入 token 和成本作为结果。 |
| 缺少随机、简单选择和 ensemble baseline | 接受。新增 B0-B4，并限制 ensemble 只用于输出可组合任务。 |
| 安全 gate 可能贡献了大部分收益 | 接受。新增 D7/D8 candidate-matched 安全消融，D7 永不执行。 |
| 可能通过过度过滤获得零 escape | 接受。新增 Clean Rejected Rate、Eligible Recall 和 O1 oracle。 |
| Spooky 可能单任务驱动 | 接受。加入 leave-one-task-out 和去除 Spooky sensitivity。 |
| 应先做功效分析和小规模验证 | 接受。新增 Phase 0 和基于 dev 层级方差的 MDE 模拟。 |
| 2-query MRR=0 证明三角色存在 P0 缺陷 | 不接受该严重度。N=2 只能是负面诊断；计划保留结果并用 20 个 dev episode 检查 evaluator 非退化。 |
| 29-query 大于计划 T1，Stage Hybrid 未显著所以不应扩展 | 事实修正：29 小于修订后的 120。CI 跨零意味着 claim 关闭，恰好说明需要冻结的盲标扩展，而非要求开发集先显著。 |
| 建议 6 tasks x 5 seeds 替代 12 x 3 | 不采用为主设计。研究目标包含跨任务泛化，task cluster 比重复 seed 更重要；主分析使用 task-blocked 方法，混合模型仅作 sensitivity。 |

修订后的实施门槛是：先通过 Phase 0、盲标、split 和安全 gate，再决定是否投入 T4。任何
开发集不利结果都保留，不以“先显著再正式测试”的方式制造 winner's curse。

### 最终复审

ClaudeAgent 对修订版再次只读审查后给出的 verdict 是：**无 P0，设计可识别，修订后可
实施，可以进入 Phase 0**。最终复审确认：

- F 四格能够识别 portfolio、Stage Hybrid 和交互效应；
- P0-P3 不再把 Replay 的收益错误归因给整个三角色；
- 等机会预算和实际成本报告是可执行的公平口径；
- B0-B4、D7/D8、O1 足以分离简单 baseline、安全过滤、排序收益和过度过滤；
- task-blocked 推断、leave-one-task-out 和显式 claim gate 与当前样本规模相容。

复审剩余的四项 P1 已写入正文：replay-eligible 至少 8/12、标准化 MDE <=0.8、Phase 0
定量检查清单、B4 可组合性定义。D7 也被限制为不加载 executor 的独立只读进程。

## 19. 实际执行状态与停止边界（2026-07-17）

本节记录真实结果，不修改前述成功标准。

### 已完成

- T0/T1：冻结 120 个 test episode、20 个 dev episode、22 个检索/消融条件和 2640 条
  test receipt；source-run split 无交叉，unsafe primary escape 为 0。
- T1 负结果：F11 Stage Hybrid `nDCG@10=0.4382`，低于 SOP-only `0.5222`；Poincare
  `0.4382`，低于 Flat-Twin `0.4431`。这两项明确否定当前几何/复合排序优越性解释。
- Replay frozen fixture：48 case、8 类缺陷、16 个结构变体；fixture 内 issue recall 与
  pre-execution block rate 都是 `1.0`。
- Replay 独立挑战：全新 ClaudeAgent 在禁止读取 detector/fixture 的条件下独立生成 16 个
  held-out 变体。一次性结果为 issue recall `0.125`、pre-execution block rate `0.1875`。
  评估后禁止据此调 detector；该结果关闭泛化安全主张。
- T2 bounded pilot：1 个 Spooky Draft episode x 4 条件。首次调用成功 `1/4`，保留失败
  receipt；重试后四条件均有候选，成功候选 family alignment 与 provenance completeness
  均为 `1.0`。因样本仅 1 且无人审，adoption claim 关闭。
- T3 bounded pilot：1 个 transductive defect。R1 静态修复 clean；R2 五阶段结构 clean 但被
  static taint 误拒；R3 static、preservation、stage structure 均 clean，可送独立 runtime，
  但尚无 evaluator-owned runtime provenance，故 clean repair 仍为 0。
- 测试：benchmark/retrieval/leakage 相关测试分进程共 `84 passed`。

### 尚未完成且不得伪装完成

- 两名盲标者、第三人 adjudication 和 Krippendorff alpha；
- 60+ episode 的完整 T2 Agent 采纳实验；
- 48 case x R0-R3 的完整生成与 evaluator-owned runtime；
- T4 的 12 task x 3 seed 外部 holdout 微训练；
- 任何 downstream superiority、universal replay safety 或 Poincare 优越性主张。

### 科学停止规则

独立 held-out safety challenge 已触发硬失败：存在大量 planted leakage 未在 GPU 前阻断。
因此当前版本不得进入 claim-bearing T4。继续运行完整 T2/T3 或 GPU 微训练只能作为诊断，
不能挽救本版本的安全主张；下一版本必须在不复用这 16 个 held-out 样例调参的前提下，
预注册新的外部挑战集后重新评估。

### v1 终态判定

v1 已按预注册停止规则记为 `completed_stopped_fail_closed`，机器可读账本位于
`paper-skills/eval_composite_memory/reports/benchmark_terminal_report_v1.json`。这里的
“completed”只表示负面实验已经触发停止规则并完成证据封账，不表示未运行的 T4 通过，
也不开放任何机制、采纳、Replay 成功或下游优越性 claim。

覆盖根因审计位于 `reports/coverage_audit_v1.json`：120 个 test episode 中 70 个满足冻结
覆盖，50 个为显式 gap。其中 30 个在完整历史图中也少于 3 条 clean 兼容 SOP，必须新增
独立 clean 证据；另 20 个的补充证据只存在于被 source split 留出的 run 中，把它们移回
memory 会破坏冻结的来源隔离。因此 v1 不通过重划 split、降级到 L3 或跨 family 补位来
消除这些负结果。
