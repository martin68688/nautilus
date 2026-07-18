# Nautilus / RunForest 研究定位与 Novelty 审计

> 分析日期：2026-07-14  
> 语料：本地 `文献/`、`coordination/` 当前研究记录、ICLR 2026 Recursive Self-Improvement Workshop 110 篇录用论文，以及补充的 arXiv / AAMAS 正文。

## 结论先行

当前最有希望的论文，不应再以“Hyperbolic Skill Memory”或“hierarchical procedural memory”作为主标题。前者没有正向实验支撑，后者已被 MemP、SkillRL、SkillGraph、MACLA、Trace2Skill 和 SKILL-DISCO 密集覆盖。

更可辩护的定位是：

> **面向机器学习工程 Agent 的协议安全分层决策记忆（protocol-safe stratified decision memory）**：系统在不同决策阶段只允许不同抽象层级的经验进入候选集；每条经验必须携带可核验的执行证据；对 protocol-biased 但方法有价值的候选，不做自由重写，而执行 preservation-constrained staged repair；只有通过 runtime provenance 的结果才能参与排名并回写正向记忆。

这条主线把项目从“又一个 agent memory”移到“self-improving MLE agent 的 decision governance / evidence eligibility / safe repair”上。

当前 novelty 的保守判断是 **Level 2 — High Overlap**：不存在一篇论文覆盖你的完整闭环，但 GOME、POLARIS、MACLA 分别占住了三个相邻面。如果把 claim 收窄到上述四件套，并做出因果消融，可形成更稳健的 **Level 3 — Medium Overlap** 定位。

## 1. 我从仓库还原出的当前研究

### 1.1 研究问题

现有 MLE Agent 会积累大量搜索树、代码、metric、失败和修复轨迹，但“检索到了经验”并不等于“在正确决策阶段使用了正确粒度的经验”。低层 API/路径修复可能挤掉完整方法路线；被污染或 protocol-biased 的高分方案可能进入排名；普通 Debug Agent 又可能在修协议时破坏原本有价值的方法。

### 1.2 当前机制

1. 三角色根节点隔离：`coldstart_baseline`、`memory_reproduction`、`novel_exploration`。
2. Novel-only 分层检索：
   - L1：完整 method family / strategy route；
   - L2：选定 family 后、只在 `model_design` 前注入兼容 tactic；
   - L3：只在 Debug 阶段提供 failure / repair。
3. RunForest execution evidence：run、节点、parent-child transition、metric、code、failure、local-best lineage。
4. Evidence eligibility：clean、执行成功、metric 有效、rank-eligible，排除 buggy / quarantined / protocol-biased。
5. Staged protocol repair：按 `data_scope → validation_provenance → cross_fit/OOF → selection_freeze → final_holdout → runtime provenance` 修复；冻结模型、backbone、feature family、loss、optimizer 和预算等 preservation contract。
6. 最终 gate：只有 runtime provenance 证明 split、fit、OOF、selection、freeze、holdout 使用合法，候选才可排名、采纳并写回正向记忆。

### 1.3 已有证据与边界

- 旧 240-query / 21-run 离线结果中，Tree-only MRR 0.3741，Stage Hybrid 0.3670，Flat-Twin 0.3709；旧 hybrid 没有赢。
- 新 Layered Strategy 在 2 个 held-out query 上 Strategy Precision@3 从 0.3333 到 1.0，Detail Intrusion@3 从 0.6667 到 0，但样本只有 2，`claim_allowed=false`。
- Poincaré 没有稳定赢 Flat-Twin 或独立 Euclidean；当前 geometry claim 不成立。
- 工程实现与 no-GPU preflight 很完整，但还没有同期在线三臂下游结果。

## 2. Novelty 四轴分解

### Problem framing

在 MLE tree-search / code-evolution Agent 中，如何让跨 run 经验真正改善决策，同时避免抽象层级错配、数据泄漏、验证协议偏差与错误正向记忆写回。

### Core mechanism

三角色隔离 + L1/L2/L3 决策阶段门禁 + execution-grounded RunForest/SOP 双载体 + preservation-constrained staged protocol repair + runtime provenance rank/writeback gate。

### Key insight

Agent memory 的主要失败不只是召回不足，而是：

1. **decision-stage mismatch**：正确信息在错误阶段也是有害信息；
2. **evidence ineligibility**：真实执行结果不等于合法证据；
3. **repair destructiveness**：自由修复可能修掉协议问题，同时破坏方法主体；
4. **memory poisoning by promotion**：未经证明的候选一旦排名或写回，会递归污染后续自改进。

### Application domain

长时程、可执行、可验证的机器学习工程 Agent，尤其是 MLE-Bench / Kaggle-style code search。

## 3. Recursive Workshop 文献图谱与本研究关系

Workshop 共录用 110 篇。与本研究真正相邻的不是全部论文，而是五组。

### A. Memory design / context evolution

- [ALMA](https://arxiv.org/abs/2602.07755)：用 Meta Agent 在代码空间搜索 memory design；威胁“手工设计 memory architecture”的合理性。
- [SimpleMem](https://arxiv.org/abs/2601.02553)：结构压缩、递归 consolidation、query-aware retrieval；覆盖效率型 memory pipeline。
- [ACE](https://arxiv.org/abs/2510.04618)：把 context 作为持续增长与整理的 playbook，用 Generator / Reflector / Curator 和 execution feedback 更新。

对本研究的含义：不要 claim “structured/evolving memory”本身新。你的 taxonomy 和 gate 必须被证明是由 MLE 决策结构与安全约束推出的，而不是手工 playbook。

### B. Procedural skill learning

- [SkillRL](https://arxiv.org/abs/2602.08234)：成功轨迹 + failure lessons，general/task-specific SkillBank，并与 RL policy 共演化。
- [PRAXIS](https://arxiv.org/abs/2511.22074)：按 environment/internal state 检索 state-action-result procedural exemplars。
- [Knowledge is Not Enough / PaST](https://arxiv.org/abs/2601.11258)：参数空间的 skill vector transfer，与外部 SOP memory 不同。

对本研究的含义：L1/L2 不能只描述为 general/task-specific hierarchy；必须强调它是 **decision-stage / abstraction eligibility**，且候选需有同任务 clean run evidence。

### C. MLE / code self-improvement

- [Towards Execution-Grounded Automated AI Research](https://arxiv.org/abs/2601.14525)：execution-guided evolution 可以产生有效研究方法，也展示了早停滞与 RL mode collapse。
- [GOME](https://arxiv.org/abs/2603.01692)：MLE-Bench 中把 structured diagnostic feedback 当 gradient、success memory 当 momentum；这是应用域最接近的竞争者。
- [Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805)：复杂 code-evolution pipeline 常不胜简单基线；直接规定了你必须采用的评估强度。

对本研究的含义：不能笼统 claim “execution-grounded MLE memory”。与 GOME 的差异必须是：GOME 优化“如何更新当前解”，你的工作约束“什么经验在什么阶段有资格影响决策，以及 protocol-biased 解如何在不改方法的条件下恢复资格”。

### D. Repair / harness / error localization

- [POLARIS](https://arxiv.org/abs/2603.23129)：从失败抽象 repair strategy，生成 minimal code patches，执行检查后运行时更新 policy；是 staged repair 最近邻。
- [AutoHarness](https://arxiv.org/abs/2603.03329)：根据 environment feedback 自动合成 action verifier / policy harness。
- [Structure Enables Effective Self-Localization of Errors](https://arxiv.org/abs/2602.02416)：结构化 thought boundary 使 error localization 和 backtracking 更可靠。

对本研究的含义：不能 claim “failure abstraction + minimal repair”新。真正 delta 是 **protocol-specific repair transaction**：阶段图由代码能力触发，方法主体由 preservation contract 冻结，中间节点零 GPU/零排名/零正向记忆，最终资格由 runtime data provenance 决定。POLARIS 明确承认没有 rollback/强 static validation，并不提供这一合同。

### E. Verifier / provenance / recursive poisoning

- Workshop 的 `Verifying the Verifiers` 讨论 coding-agent failure attribution，区分 agent failure 与 benchmark defect。
- 本地 [From Agent Traces to Trust](../文献/graph文献/2606.04990v2.pdf) 综述把 execution provenance 定义为连接 evidence、tool output、memory、action 和 answer 的 typed execution graph。
- Workshop 的 `Reward Hacking in Self-Improving Code Agents` 表明 self-improvement loop 会利用 evaluator 缺口。

对本研究的含义：provenance 本身不是新贡献；新意应落在 **provenance controls memory eligibility and recursive promotion**，即 provenance 不是日志，而是排名与写回的执行语义。

## 4. 本地文献库对旧定位的修正

本地文献库已经推翻了早期几条“空白”判断：

1. “procedural memory 很少有人做”已不成立：MemP、MACLA、SkillRL、PRAXIS 已覆盖。
2. “中间粒度 skill carrier 是空白”已不成立：SKILL-DISCO 明确抽取介于 raw operator 和 full trajectory 之间的 multi-step operation，并编译为可执行 PFSM skill。
3. “hierarchical skill library 是空白”已不成立：SkillRL 有 general/task-specific 层次，MACLA 有 atomic procedure / meta-procedure playbook。
4. “graph + agent skill retrieval 是空白”已非常脆弱：SkillGraph 做 dependency-aware graph traversal，SAGE/EXG 等也覆盖 graph memory；即便 runtime tool autonomy 的具体实现不同，也不足以单独支撑主会 novelty。
5. “执行证据 / provenance”也已形成独立文献群。

仍相对干净的交叉点是：

> **将 execution provenance 从描述性记录升级为 self-improvement loop 的资格系统，并以 decision-stage abstraction gate 和 preservation-constrained repair transaction 执行这一资格系统。**

## 5. 最接近工作的四轴重合度

| Prior work | Problem | Mechanism | Insight | Domain | Novelty level |
|---|---|---|---|---|---|
| GOME | match | differ | partial/match | match | Level 2 |
| POLARIS | match | partial/match | partial/match | differ | Level 2 |
| MACLA | match | partial/match | partial | differ | Level 3 |
| SkillGraph | match | partial | partial | differ | Level 3 |
| SkillRL | match | partial | partial | differ | Level 3 |
| ACE | partial | partial | match | differ | Level 3–4 |
| Trace2Skill | partial/match | partial | differ | differ | Level 3–4 |
| SKILL-DISCO | partial | partial | differ | differ | Level 3–4 |
| PRAXIS | partial | partial | partial | differ | Level 4 |
| ALMA | partial | differ | partial | differ | Level 4 |

按 scoop-check 的 worst-case 规则，当前总体为 **Level 2 — High Overlap**。这里不是说工作“已被做完”，而是说 broad claim 只剩一个轴可区分，必须收紧。

## 6. 可辩护的一句话 Delta

> Unlike GOME, which uses structured execution feedback to choose how an MLE solution should be updated, and POLARIS, which turns failures into generic validated policy patches, the proposed system controls which execution memories are eligible to influence each decision stage and repairs protocol-biased candidates through a method-preserving transaction whose runtime provenance determines ranking and positive-memory writeback, preventing recursive promotion of invalid improvements.

中文版：

> 不同于 GOME 用结构化执行反馈决定“当前方案怎么改”、POLARIS 将失败抽象为通用 policy patch，本研究解决“哪条执行经验在当前决策阶段有资格影响 Agent”：它用分层门禁限制策略/战术/修复记忆，用 preservation-constrained transaction 修复 protocol-biased 候选，并让 runtime provenance 直接决定排名与正向记忆写回，从而阻断无效改进在递归自提升中的晋升与扩散。

如果这句话最终无法通过消融被验证，就不应强行把三个工程模块包装成一个方法贡献。

## 7. 建议的论文形状

### 推荐主标题

**Protocol-Safe Stratified Memory for Self-Improving Machine Learning Agents**

备选：

- **When Experience Is Eligible: Stage-Aware Memory and Provenance-Gated Repair for MLE Agents**
- **From Execution Traces to Rank-Eligible Experience in Self-Improving MLE Agents**

标题中暂时不要出现 Hyperbolic、Skill Graph、Recursive Skill Memory。它们会把 reviewer 引向你证据最弱、竞争最拥挤的位置。

### 三个贡献，最多三个

1. **Decision-stage stratification**：首次把 MLE experience retrieval 定义为 eligibility-constrained decision problem，而不是相关性 top-k。
2. **Method-preserving protocol repair transaction**：把 validation protocol 修复从自由 Debug 中分离，并提供可检查的 preservation contract。
3. **Provenance-gated recursive promotion**：runtime provenance 决定 rank、adoption 与 positive-memory writeback，研究错误经验如何在 self-improvement loop 中扩散或被阻断。

RunForest、SOP taxonomy、三角色、adoption trace 是实现这些贡献的系统构件，不要每个都单列成 novelty。

## 8. 最小可发表实验

### 实验 A：分层门禁是否真的改善“方法决策”

固定 memory corpus、LLM、tool、预算、candidate count，只改变 retrieval policy：

1. Tree-only；
2. untyped flat/hybrid retrieval；
3. general/task hierarchy；
4. decision-stage L1/L2/L3 gate（ours）。

至少 6 个任务 family、≥60 个 held-out decision episodes；测试 run 与 memory source run 严格隔离。指标：Strategy Precision@3、family diversity、detail intrusion、evidence eligibility、strategy adoption、生成代码 alignment。当前 2-query 结果只能作单元测试。

### 实验 B：分层门禁是否改善下游 MLE 结果

在同 Job / 同模型 / 同预算下做同期控制。最低建议 8–10 个 MLE-Bench 任务，每臂 3 seeds：

- simple sequential / random-search baseline；
- Tree-only RunForest；
- untyped Stage Hybrid；
- Layered Strategy。

报告 normalized score、any-medal、best-valid metric、time-to-best、GPU-hour、token/API cost、adoption-conditioned gain。必须包含 `Simple Baselines are Competitive with Code Evolution` 类型基线。

### 实验 C：protocol repair 的因果价值

构造或收集多种真实 protocol defects：transductive fitting、early-stopping reuse、missing OOF、ensemble selection on holdout、group/time leakage、final-holdout reuse。

对比：

1. reject/drop；
2. ordinary Debug Agent；
3. generic minimal repair（POLARIS-style）；
4. staged preservation-constrained repair（ours）。

指标：clean repair success、method preservation、static audit pass、runtime provenance pass、false acceptance、false rejection、GPU waste、修复后相对原合法方法的性能保持率。

### 实验 D：recursive promotion / memory poisoning

这是最可能把系统论文升格为研究论文的实验。向 memory loop 注入不同类型的 invalid high-score experience，测量经过多轮 search 后：

- invalid memory retrieval rate；
- invalid candidate promotion rate；
- descendant contamination；
- clean-best degradation；
- provenance gate 的阻断率与误伤率。

只有这个实验能直接证明“rank/writeback gate”不是普通日志工程。

### Geometry 的处理

把 Poincaré vs Flat-Twin / Euclidean 放到 appendix 或 negative result。除非在同工具、同 SOP、同 query anchor 下稳定过预注册 gate，否则不要恢复标题级 geometry claim。

## 9. Reviewer 最可能攻击的点

1. **“只是手工 taxonomy。”** 需要跨任务 held-out 标注、inter-annotator agreement、自动/规则分类误差和 taxonomy perturbation。
2. **“把多个工程 safeguard 拼起来。”** 需要以 recursive invalid-promotion 作为统一理论对象，并用因果实验连接三个模块。
3. **“只在 Spooky 有效。”** 必须覆盖文本、图像、tabular regression、group/time split、ensemble 等多种任务与协议。
4. **“评估集太小且后验设计。”** 预注册 task split、query generator、claim gate；当前 2-query 结果不能进入 headline。
5. **“Tree-only 已经更好。”** 不回避旧 MRR 负结果；解释新方法优化的是 strategy eligibility，而非 generic retrieval，并在下游同期对照验证。
6. **“provenance guard 只是 Agent 自报。”** 必须加强 fit-call / selection-call 与 guard-call 的静态或动态绑定，否则不能称为 proof。
7. **“与 GOME / POLARIS 的差异是文字。”** 加两个直接 transfer baseline：GOME-style structured update without stage gate；POLARIS-style generic repair without preservation/provenance gate。
8. **“复杂方法不如简单基线。”** 报完整成本与 Pareto，而不是只报成功率。

## 10. 现在最值得做的三件事

1. **冻结论文主 claim**：从今天起把 Hyperbolic 降为支线，主线改成 protocol-safe stratified decision memory。
2. **先做实验 D + C，再扩大在线 benchmark**：如果无法证明 invalid promotion 被显著阻断且合法方法被保留，这条论文主线不成立；先用较便宜的诊断实验 falsify。
3. **重新整理 related work**：以 GOME、POLARIS、MACLA 为三篇“almost prior”，SkillGraph/SkillRL/ACE 为第二圈；不要再用“无人做 procedural graph memory”作开场。

## 11. 最终判断

工程价值很高，研究主张目前处于“可救但必须收窄”的状态。

- **不建议继续押注**：hyperbolic geometry advantage、generic hierarchical skill memory、graph + agent retrieval。
- **建议押注**：decision-stage eligibility、method-preserving protocol repair、provenance-gated recursive promotion。
- **决定性 falsification**：若分层门禁在充分 held-out 任务上不提高合法 strategy adoption / downstream score，或 staged repair 不比 generic repair 更能同时保持方法与协议合法性，则应把工作降级为系统工程论文，而不是主会 method paper。

