# SkillGraph vs 双曲 Procedural Skill 记忆 · 逐点攻防文档

> **用途**：本研究的 Related Work 弹药库 + Rebuttal 应答模板。逐层拆解 SkillGraph（arXiv:2605.12039）的每个机制，对照本研究"双曲 procedural skill 记忆"，明确**撞车面 / 超越点 / 审稿质疑 / 应答话术 / 必做实验**。
>
> **结论先行**：SkillGraph 是本研究最近的、机制层面正面碰撞的竞品（4 个子层撞车），但本研究在**节点语义、几何结构、检索范式、演化信号、冲突能力**5 个维度全部具备独有件。本研究的风险不在"被它抢先"（它 2026-05 已发表，无几何/无冲突/无 metric 监督），而在"审稿人会拿它质疑你的 novelty 增量"——本攻防文档即用于系统性回应此质疑。

---

## 0. 战场总览：5 层对照

| 层 | SkillGraph 机制 | 本研究机制 | 撞车度 | 本研究超越点 |
|---|---|---|---|---|
| ① 结构 | 平面 skill 图，2 类节点（general/task），3 类 typed edge | 双曲 procedural 图，6 类语义节点，7 类几何化边 | 🟠 中 | 双曲几何 + 多类型语义节点 + 冲突边 |
| ② 检索 | 程序化 BFS+beam+拓扑排序（零 LLM） | agent 自主多跳几何导航（tool call） | 🟠 中 | agent 自主性 + 几何统一检索 |
| ③ 演化 | RL 反馈驱动 Insert/Merge/Split/Deprecate + 边权重演化 | metric_delta 监督半径 + 条件化保留低频 | 🟡 低 | 客观 ground truth + 反 prevalent bias |
| ④ 训练范式 | RL（GRPO，改 policy 权重） | context 注入（不改权重，即插即用） | 🟢 低（范式不同） | 即插即用、跨模型迁移 |
| ⑤ 冲突 | 无 | prevents 几何对跖 + metric 判定 | 🔴 无（本研究独占） | 全新能力 |

**净结论**：撞车集中在 ①②（结构+检索），但本研究在这两层都用"几何 + 语义节点 + agent 自主"三件套实现了差异化；③④⑤ 则是范本级差异或独有能力。下面逐层展开。

---

## ① 结构层攻防

### 撞车面（审稿人会指出的）
> "SkillGraph 也把技能组织成图、用 typed edge 表达依赖关系，你的'技能图'与它有何本质区别？"

SkillGraph 的结构：
- **2 类节点**：general skill（领域无关策略）/ task-specific skill（任务策略）。
- **3 类 typed edge**：`prereq`（前置）、`enhance`（增强）、`co_occur`（共现）。
- 平面有向图，边带权重 `w(e)∈[0,1]`。

### 本研究超越点

**超越点 1-A：双曲几何承载结构（vs 平面）**
SkillGraph 是平面图——节点位置无几何意义，关系全靠显式存的 edge。本研究把 skill 嵌入 Poincaré 球，**7 类关系是几何位置 (Δr, Δθ) 的派生，无需显式存储**。这不只是"换个容器"，而是带来两个 SkillGraph 做不到的能力：
- **低频 skill 的几何保护**：SkillGraph 的 Deprecate 操作（成功率 <0.15 就弃用）会**主动丢弃低频 skill**；本研究双曲边缘指数体积让低频有效 skill 即使频次低也保持检索可达性，**直击 prevalent bias**。
- **关系的几何可判性**：SkillGraph 的 `prereq` 是存出来的，检索时遍历；本研究算 (Δr, Δθ) 即知关系类型，**全空间都能算，不受显式边限制**。

**超越点 1-B：6 类 procedural 语义节点（vs 2 类泛节点）**
SkillGraph 的节点只有 general/task 二分，本质是"技能文本 + 标签"。本研究 6 类节点（Skill/SOP/Condition/FailureMode/Evidence/Implementation）是**procedural 方法学语义**——Condition 编码适用上下文、FailureMode 编码失败模式、Evidence 编码 metric 证据。这让图能表达 SkillGraph 表达不了的关系（如"某 SOP 的适用条件 = 另一 SOP 的失败模式"）。

**超越点 1-C：冲突边 prevents（SkillGraph 完全没有）**
SkillGraph 的 3 类边都是正向/协同关系，**无任何冲突表达**。本研究 `prevents`（对跖）能表达"两个都有效但互斥"的双真冲突——这是 §5 详述的独占层。

### 审稿质疑 + 应答话术

> **Q1（必问）**："你的 6 类节点是不是 SkillGraph general/task 二分的人工细化？增量在哪？"
>
> **A1**：不是细化，是**语义正交分解**。SkillGraph 的 general/task 是单一轴上的类别标签；本研究的 Condition/FailureMode/Evidence 是**独立的关系锚点**——Condition 是 `applies_when` 边的端点，FailureMode 是 `prevents` 边的端点，Evidence 是 `supported_by` 边的端点。删除任何一类，对应的关系边就无处挂载。这是结构必需，不是装饰。**实验支撑**：消融"去 FailureMode 节点"应导致冲突场景召回下降（待跑）。

> **Q2**："双曲几何的好处有量化证据吗？还是只是理论上'更优雅'？"
>
> **A2**：有量化假设，需实验证伪/证实。三个可证伪预测——(a) 低频 skill 召回率：双曲 > 平面（因指数体积）；(b) 深层 patch 区分度：双曲边缘节点间距 > 平面；(c) 检索效率：单空间几何检索 vs 双索引拼接，latency 应更低。**这是 §6 实验清单的核心**。

---

## ② 检索层攻防

### 撞车面
> "SkillGraph 的 graph-aware retrieval 已经证明'依赖感知的有序检索'有效（去掉它 ALFWorld 暴跌 31 分），你的检索和它有什么不同？"

SkillGraph 检索：种子选择 → 向后 BFS（深度 D）+ 向前 beam（宽度 B）→ 拓扑排序输出有序序列。**全程零 LLM、程序化固定管道**。

### 本研究超越点

**超越点 2-A：agent 自主多跳（vs 固定程序化管道）**
SkillGraph 的检索路径是**硬编码**的（BFS 深度 D、beam 宽度 B 都是预设超参），agent 无权决定何时查、查什么、走几跳。本研究 agent 通过 tool `navigate(direction=θ, depth=r, hops=k)` **自主决定方向（选 skill 类别）、深度（选层级）、跳数（选路径）**。这是本研究综述里"图 + 运行时 agent 自主检索"空白象限的具体填充。

**超越点 2-B：几何统一检索（vs 两阶段遍历）**
SkillGraph 虽然是单图，但检索仍是"种子→扩展→排序"三步串行。本研究在同一双曲空间内用单一公式同时算语义+层级+路径：
```
Score(v|q) = w₁·exp(−d_角向(q,v)) + w₂·exp(−|r_v−r_q|) + w₃·exp(−d_双曲(q,v))
```
三项在同一几何内闭环，无串行误差累积。

### ⚠️ 这一层的最大风险（必须诚实面对）

**SkillGraph 的程序化拓扑排序已被其实验证明极强**（消融掉它 ALFWorld 掉 31 分）。本研究的"agent 自主"必须**跑赢**它的程序化管道，否则这个差异点不成立。这是本研究**最高风险的实验**。

### 审稿质疑 + 应答话术

> **Q3（最致命）**："agent 自主多跳比固定管道更贵（要 LLM 决策），你凭什么说它更好？SkillGraph 的固定管道已经很强了。"
>
> **A3**：两个论点。(1) **场景差异**：SkillGraph 的任务是强组合性（ALFWorld 有明确子目标序列），固定拓扑排序天然契合；本研究 MLE 场景的"组合"是方法学组合（特征工程+模型选择+调参），**没有固定执行顺序**，固定管道会强行排序出伪依赖。(2) **效率补偿**：agent 自主虽单次贵，但能**按需检索**（不需要时跳过），SkillGraph 是每步强制检索。**实验支撑**：报告 token 成本曲线，论证"agent 自主减少无效检索步数、端到端 token 更低"（待跑）。**这是必跑的保命实验**。

> **Q4**："几何统一检索 vs SkillGraph 的 BFS+beam，本质区别是什么？"
>
> **A4**：BFS+beam 是**离散跳数**的，深度 D 是硬上限；几何检索是**连续距离**的，可以返回"接近但不完全相邻"的 skill（双曲距离连续可排）。这解决了 SkillGraph 的"路径长度永远 ≤D，信息不够也不能再深挖"的局限。

---

## ③ 演化层攻防

### 撞车面（较弱）
> "SkillGraph 也有图演化（Insert/Merge/Split/Deprecate），你的'写入'新在哪？"

SkillGraph 演化：节点级（失败触发插入、Jaccard 合并、成功率 0.15-0.4 拆分、<0.15 弃用）+ 边级（路径强化、共现发现、衰减剪枝）。**信号源：RL task reward 的成功率 p̂**。

### 本研究超越点

**超越点 3-A：metric_delta 客观监督（vs RL reward）**
SkillGraph 的演化信号是 RL 训练的 task reward 转化成的成功率 p̂——**受 reward hacking 影响**（如 Reward Hacking in Self-Improving Code Agents 所证，RSI 系统 30-50% 的"改进"是 hacking）。本研究用 `metric_delta`（MLE 执行的真实指标变化）监督半径，是**蒸馏时的客观 patch 验证**，有 ground truth。这让本研究的图演化**可验证、可解释、不被 hacking 污染**。

**超越点 3-B：反 prevalent bias（vs 主动弃用低频）**
SkillGraph 的 Deprecate（`p̂<0.15` 弃用）是**主动丢弃低频 skill**——这恰恰是 Trace2Skill 的 prevalent bias 问题（综述主贡献1 要解决的）。本研究**条件化保留低频**：低频但 metric 有效 → 保留并放在双曲边缘（几何保护可达性），不弃用。

### 审稿质疑 + 应答话术

> **Q5**："RL reward 也是客观信号（任务成功与否），为什么你的 metric_delta 更优？"
>
> **A5**：两者客观性层级不同。RL reward 是**任务级二值信号**（成功/失败），无法区分"这个 skill 对最终成功贡献多大"；metric_delta 是**patch 级连续信号**（这个改动让指标变化多少），能定位**单个 skill 的边际贡献**。Reward Hacking 文献已证 RSI 系统在任务级 reward 下系统性作弊；patch 级 metric_delta 因颗粒度更细，hacking 难度更高。**实验支撑**：对比"RL reward 监督的演化" vs "metric_delta 监督的演化"，后者在 holdout 上 hacking 率更低（待跑，呼应 Reward Hacking 论文的检测协议）。

---

## ④ 训练范式层攻防

### 撞车面（最低，范式不同）
> "SkillGraph 用 GRPO 训练，你的方法用什么？"

SkillGraph：GRPO 训练 policy，**改 model 权重**，需 GPU 训练、cold-start SFT。

### 本研究超越点

**超越点 4：context 注入，不改权重（即插即用）**
本研究 skill 图以 **context 注入**方式喂给 agent，**不改任何模型权重**。优势：
- **跨模型迁移**：同一 skill 图可用于 DeepSeek、Qwen、GPT 等任意模型，SkillGraph 的训练绑定单一 base model。
- **零训练成本**：SkillGraph 需 GRPO 训练（消融显示无 cold-start SFT 掉 17 分）；本研究即插即用。
- **可解释**：context 里的 skill 是人类可读文本，权重的隐式记忆不可读。

### 审稿质疑 + 应答话术

> **Q6**："不改权重的 context 注入，是不是只是 prompt engineering？"
>
> **A6**：不是。纯 prompt engineering 是静态的（一个固定 prompt）；本研究的 skill 图是**动态、可检索、可演化**的结构化记忆——agent 运行时按当前任务从双曲图检索相关子图注入，不同任务注入不同 skill。这等同于"外部化的、可成长的参数记忆"。**实验支撑**：对比"静态全量注入" vs "动态检索注入"，后者 token 成本低且性能不降（待跑）。

---

## ⑤ 冲突层攻防（本研究独占，最强护城河）

### 撞车面
**无。** SkillGraph 完全没有冲突处理机制。

### 本研究超越点

**超越点 5：prevents 几何对跖 + metric 判定（独占）**
两个 `metric_delta` 都为真但互斥的 skill（如"增大数据量" vs "精简数据集"），本研究放在双曲对跖位置（角度差≈π）。检索到 A 时，对跖的 B 自动被几何识别为"要警惕的对立面"。这解决的是 MLE 场景的**双真冲突**（两次真实执行结果矛盾，无时间先后，LLM 无先验知道信谁）——只有 metric + 执行条件能判，MemConflict/STALE 文献已证现有系统在此失效。

### 审稿质疑 + 应答话术

> **Q7**："冲突判定为什么需要几何化？逻辑规则也能判（if A and B 同条件 then conflict）。"
>
> **A7**：逻辑规则需要**穷举冲突对**（人工或 LLM 标注），不可扩展且易漏。几何化让冲突**从位置自动涌现**——任何两个被放在对跖位置的 skill 自动是冲突，无需显式枚举。且几何对跖有**可调阈值**（角度差 > θ_conflict 判冲突），比硬规则更平滑。**实验支撑**：对比"逻辑 prevents" vs "几何 prevents"，后者在未见冲突对上的泛化召回更高（待跑）。

---

## ⑥ 审稿人 Rebuttal 弹药库（高频质疑速答）

| # | 审稿质疑 | 一句话应答 | 支撑实验 |
|---|---|---|---|
| 1 | 和 SkillGraph 区别？ | SkillGraph 是平面+RL+程序化检索+无冲突；本研究是双曲+context注入+agent自主+几何冲突，5 维全异 | 5 层消融 |
| 2 | 为什么不普通 GNN？ | GNN 编码图结构但无几何层级；双曲的径向-角向分解让层级+语义在同一空间，且边缘指数体积保护低频 | 双曲 vs 欧氏 GNN |
| 3 | agent 自主比固定管道强？证据？ | MLE 场景无固定执行序，固定管道造伪依赖；agent 自主按需检索降总 token | token 成本曲线 |
| 4 | metric_delta 比 RL reward 强？ | patch 级连续信号 vs 任务级二值；颗粒度细→hacking 难度更高 | holdout hacking 率 |
| 5 | 双曲只是更优雅？ | 有可证伪预测：低频召回、深层区分度、检索效率三指标 | 三项量化实验 |
| 6 | context 注入 = prompt eng？ | 静态 prompt vs 动态可检索可演化的结构化外部记忆 | 静态 vs 动态注入 |
| 7 | 几何冲突比逻辑规则强？ | 几何冲突从位置自动涌现+阈值可调，无需穷举冲突对 | 泛化召回对比 |
| 8 | novelty 会不会被 SkillGraph 抢？ | SkillGraph 无几何/无冲突/无 metric 监督/改权重，本研究独占三件 | （无需实验，文献事实） |

---

## ⑦ 必做实验清单（按优先级）

### P0（不做就被拒）
1. **agent 自主导航 vs SkillGraph 程序化拓扑排序**（回应 Q3，本研究最高风险点）
   - 设计：同 skill 图，分别用"agent tool 自主多跳"和"BFS+beam+拓扑排序"检索，比最终 metric + token 成本
   - 预期：MLE 场景 agent 自主 ≥ 程序化，且 token 更低（因按需检索）
   - **若失败**：agent 自主差异点降级，改打"几何统一检索"这张牌

2. **双曲 vs 欧氏 GNN**（回应 Q2，审稿必问）
   - 设计：同 skill 库，分别嵌入 Poincaré 球和 ℝⁿ，比低频 skill 召回率 + 深层区分度
   - 预期：双曲在低频召回上显著优（指数体积保护）

### P1（做强差异化）
3. **metric_delta 监督半径 vs 无监督 depth**（回应 Q5）
   - 设计：半径分别用 metric_delta 和 HyperbolicRAG 式无监督预测，比 holdout 上 hacking 率
   - 预期：metric 监督 hacking 率更低

4. **几何 prevents vs 逻辑 prevents**（回应 Q7）
   - 设计：冲突判定分别用对跖阈值和硬规则，比未见冲突对的泛化召回
   - 预期：几何泛化更好

### P2（锦上添花）
5. **静态全量注入 vs 动态检索注入**（回应 Q6）
6. **低频 skill 保留率：双曲 vs SkillGraph Deprecate**（直击 prevalent bias）

---

## ⑧ 风险红线（必须诚实写入 Limitations）

1. **双曲数值稳定性**：Poincaré 球靠近边界 arcosh 溢出 → 用 Lorentz 模型 + ε 截断。
2. **agent 自主检索延迟**：单次比程序化贵 → 靠"减少无效步数"补偿，需实测。
3. **最高风险**：若实验 1（agent 自主 vs 程序化）失败，本研究核心差异点从"agent 自主"降级为"几何统一检索"，需在论文里诚实说明。
4. **被抢先风险**：HyperbolicRAG（2025-11）已近，建议优先占坑"metric 监督半径"与"prevents 几何化"两个独有件。

---

## ⑨ Related Work 可直接粘贴段落

> "SkillGraph (Li et al., 2026) 与本研究最为接近，同样将技能组织为图并以 typed edge 编码依赖。但 SkillGraph 限于平面几何与 RL 协同训练（GRPO），其节点为 general/task 二分的泛技能，层级信号来自 task reward（受 reward hacking 影响），检索为程序化的 BFS-beam-拓扑排序管道，且缺乏冲突处理机制。本研究在五点上形成差异：(1) 采用双曲 Poincaré 几何承载结构，使 7 类关系成为几何位置 (Δr, Δθ) 的派生，并以边缘指数体积几何性地保护低频有效技能；(2) 节点为 6 类 procedural 语义单元（含 Condition/FailureMode/Evidence），支持条件与冲突关系的挂载；(3) 检索由 agent 在双曲空间自主多跳导航，而非固定管道；(4) 演化半径由 metric_delta 客观监督而非无监督预测；(5) 首次将冲突关系几何化为双曲对跖，实现 MLE 双真冲突的客观判定。其中 (1)(4)(5) 在现有双曲记忆与技能图工作中均未出现。"

---

## ⑩ 一句话总纲

> **SkillGraph 是本研究最近的正面竞品，但它在"平面几何 + 泛节点 + 程序化检索 + RL reward + 无冲突"五个维度均与本研究不同。本研究的护城河不在任何单一维度，而在"双曲几何 + metric 监督 + 几何冲突"三独有件的组合——三者任一被审稿质疑，另有两者兜底。最大实验风险是"agent 自主检索能否跑赢程序化拓扑排序"，此为 P0 必跑项。**
