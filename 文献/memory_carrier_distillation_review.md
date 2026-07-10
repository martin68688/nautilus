# 记忆载体设计与方法学蒸馏综述：从搜索轨迹到可建图 Skill 单元

> **定位**：本综述聚焦本研究**最核心的方法贡献**——"记忆载体应该是什么形态，以及如何从 MLE 搜索轨迹蒸馏出这种载体"。这是 `hyperbolic_skill_memory_review.md`（双曲结构层）与 `agent记忆综述_技术细节版.md`（四环节全景）的**上游方法学文档**：双曲几何是承载结构，但结构里装什么、怎么造出来，才是方法主体。
>
> **核心论断**：当前所有 skill/记忆系统的载体粒度都走极端——要么是 raw SearchNode（太碎，每次尝试），要么是 1 个巨型 SKILL.md（太粗，整个任务的知识塞一个文件）。本研究提出**介于两者之间的"结构化细粒度 skill 单元"**（带条件 + 证据 + 失败模式），并设计了从搜索轨迹蒸馏这种单元的改进管线（条件化保留而非合并消除）。
>
> **真实数据支撑**：本综述的所有设计均基于对项目 124 个无泄露 SearchNode（6 个后期 spooky run）的真实分析，并附一次完整的手工蒸馏演示（从 1 条路径蒸馏出 7 个结构化 skill）。

---

## 0. 一页纸速览

| 层 | 问题 | 本研究的回答 |
|---|---|---|
| **载体是什么** | 记忆单元该多细/多粗 | **结构化细粒度 skill**（SOP + 条件 + 证据 + 失败模式），介于 raw node 和巨型 SKILL.md 之间 |
| **载体怎么造** | 如何从轨迹蒸馏 | **改进 Trace2Skill**：成功/失败 analyst 并行提 patch → **条件化聚类保留**（非合并消除）→ 每个 patch 带 metric_delta |
| **载体的关键差异** | 为什么不用现有方案 | raw node 无条件无证据；SKILL.md 丢失转折与冲突；本研究**保留条件/证据/失败/冲突** |
| **产出规模** | 蒸馏出多少 | 1 条路径 → ~7 个 skill；全量 ~50-150 个（vs raw 423 / SKILL.md 1 个） |
| **如何验证** | 凭什么说这个载体好 | 用真实 run 手工蒸馏，展示**冲突对、refines 链、失败 plan 原文**如何被保留 |

---

## 1. 为什么"载体设计"是研究的真正核心

### 1.1 一个被忽视的事实：现有系统的载体都走极端

通览所有 skill/记忆系统，载体粒度呈现**两极化**，无人占据中间地带：

| 系统 | 载体粒度 | 数量 | 问题 |
|---|---|---|---|
| **MLEvolve raw SearchNode** | 每次代码尝试（code_summary + metric） | 423 个（spooky） | 太碎，无独立条件/失败模式，无法直接检索 |
| **Trace2Skill SKILL.md** | 整个任务的知识塞一个文件 | 1 个（244 行） | 太粗，13 条 insight 混在一起，丢失转折/冲突/独立条件 |
| **SkillGraph skill 节点** | general/task 二分泛技能 | 几十个 | 假设 skill 已存在，**不做蒸馏** |
| **GoS skill 节点** | executable skill | 数百-数千 | 同上，**不蒸馏、不演化、不验证** |

**关键观察**：做蒸馏的（Trace2Skill）合并成 1 个文件；做图的（SkillGraph/GoS）假设 skill 已存在不蒸馏。**"蒸馏 + 图"之间缺一个"细粒度结构化 skill 单元"作为桥梁**。

### 1.2 本研究的位置：结构化细粒度 skill 单元

本研究的载体是**介于 raw node 和巨型 SKILL.md 之间的中间粒度**：

```
raw SearchNode（太碎）──→ ★ 结构化 skill 单元（本研究）★ ──→ 巨型 SKILL.md（太粗）
  423 个                     50-150 个                      1 个
  无条件/证据                带条件/证据/失败/冲突           丢失转折/冲突
```

这个中间粒度**正是双曲图的可建图节点**——每个单元有明确的 SOP（→角度θ）、metric（→半径r）、条件（→applies_when 边）、失败模式（→prevents 边端点）。

### 1.3 为什么这是方法主体（而非双曲几何）

双曲几何是**承载结构**，但"结构里装什么、怎么造出来"才是方法 novelty 主体。理由：

1. **SkillGraph/GoS 都不做蒸馏**——蒸馏是本研究的独占环节。
2. **Trace2Skill 做蒸馏但合并成 1 个文件**——本研究的改进是"条件化分解"。
3. **raw node 太碎、SKILL.md 太粗**——本研究找到中间粒度。
4. 载体的质量直接决定双曲图的检索质量——"垃圾进，垃圾出"。

---

## 2. 载体设计：结构化细粒度 skill 单元

### 2.1 载体的字段 schema

一个记忆载体单元应包含以下结构化字段（对应综述里的 6 类节点）：

```
SkillUnit = {
  // 核心内容
  sop:            "小数据集上用 LoRA 微调 DeBERTa 替代渐进解冻",
                    // 程序性方法学规则（procedural），不是声明性事实
  
  // 适用条件（Condition 节点）
  conditions:     ["数据集 < 5000 样本", "transformer backbone", "val loss plateau"],
                    // 这个 skill 在什么前提下才该被召回
  
  // 客观证据（Evidence 节点）
  evidence: {
    metric_delta:   -0.703,        // log loss 变化（越负越好，spooky 是 log loss）
    source_trace:   "step10→16, run 20260516_104127",
    direction:      "maximize=False"  // 指明 metric 方向
  },
  
  // 失败模式（FailureMode 节点）
  failure_mode:   "全参数微调时此 skill 不适用（见冲突 skill）",
                    // 什么情况下会失败，以及为什么
  
  // 可执行实现（Implementation 节点）
  implementation: "<代码模板片段或参考文件路径>",
  
  // 元数据
  source_stage:   "improve",        // 蒸馏自哪个 stage
  confidence:     "high",           // 由 metric_delta 量级决定
  trace_ids:      ["step10@run_...", "step16@run_..."]
}
```

### 2.2 与现有载体的字段对比

| 字段 | raw SearchNode | Trace2Skill SKILL.md | **本研究 SkillUnit** |
|---|---|---|---|
| SOP | code_summary（半结构化） | ✅ 有，但混在正文 | ✅ **独立字段** |
| 条件 | ❌ 无 | ❌ 混在正文 | ✅ **独立 conditions** |
| metric 证据 | ✅ 有 metric 值 | ❌ **无**（合并丢失） | ✅ **保留 metric_delta** |
| 失败模式 | ❌ 无（只有 is_buggy） | ❌ **无**（prevalent bias 丢弃） | ✅ **独立 failure_mode** |
| 可执行实现 | ✅ 有 code | 部分 | ✅ **独立 implementation** |
| 来源追溯 | ✅ step/run | ❌ 丢失 | ✅ **保留 trace_ids** |

### 2.3 为什么每个字段都不可省

每个字段都对应双曲图的一个**结构必需**（删了图就建不起来）：

- **删 SOP** → 没有节点内容，无法 embedding 算角度 θ。
- **删 conditions** → `applies_when` 边无处挂载，无法做条件化检索。
- **删 metric_delta** → 半径 r 无监督信号，退化成 HyperbolicRAG 的无监督 depth。
- **删 failure_mode** → `prevents` 边无端点，无法做冲突判定。
- **删 implementation** → `has_implementation` 边无端点，agent 检索到 skill 后无法直接复用代码。

**这不是"字段越多越好"，而是"每个字段都是图的承重柱"。**

---

## 3. 蒸馏方法：改进 Trace2Skill 管线

### 3.1 原版 Trace2Skill 回顾（蒸馏的起点）

原版（arXiv:2603.25158）三阶段：
1. **轨迹生成**：ReAct agent 跑任务，按成功/失败分轨。
2. **并行 analyst 提 patch**：success analyst 提成功行为；error analyst 多轮 ReAct 找根因。
3. **无冲突合并**：层级合并算子 M 做去重 + **冲突消除**（三道编程栏杆）+ **prevalent pattern bias**（低频当噪声丢弃）。

**原版的两个致命局限**（本研究切入点）：
- ❌ **丢弃低频正确经验**：prevalent bias 把低频但有效的 patch 当 idiosyncratic 过滤掉。
- ❌ **仅物理冲突检测**：只查文件/行号/格式，**不检测语义冲突**（两条 patch 对同一流程给出矛盾逻辑建议）。
- ❌（附）**无图**：skill 写成线性 Markdown，丢失 SOP 间隐含联系。

### 3.2 本研究的改进：条件化保留而非合并消除

核心思想：**Stage 3 不消除冲突，而是按条件聚类保留**。三档处理：

```
矛盾候选对 (A, B) ──→ 抽取条件 ──→ 条件相同？
                                    │
                      ┌─────────────┼─────────────┐
                      ▼             ▼             ▼
                 同条件+同方向   同条件+反方向   不同条件
                      │             │             │
                      ▼             ▼             ▼
                  合并成一条    保留两边         分开保留
                  （去重）    + prevents 边    + refines 边
                             （冲突建模）     （条件分支）
```

**关键差异**：
- 同条件 + 反方向（如"小数据集用 LoRA" vs "小数据集全参数微调"）→ **不合并，加 prevents 边**。这是原版完全丢失的双真冲突。
- 不同条件（如"小数据集用交叉验证" vs "大数据集用 holdout"）→ **分开保留，加 refines 边**。原版可能因 prevalent bias 丢弃低频的那个。

### 3.3 蒸馏管线的完整设计

```
输入：N 个 raw SearchNode（有 step/stage/metric/code_summary/plan）

Stage 1 — 演化路径抽取
  按 node2parent 关系重建 root→leaf 路径
  每条路径 = 一个完整的"尝试→改进→成功/失败"故事
  ★ 用 node2parent 而非 parent 字段（项目实测 parent 字段为空）

Stage 2 — metric_delta 计算
  对每条路径的相邻节点，计算 metric 变化量
  ★ 注意 metric 方向：log loss 任务用 Δ = metric_B - metric_A（负=改善）
  ★ 排除数据泄露节点（metric < 0.1 视为泄露，不参与）

Stage 3 — 并行 patch 提取（analyst）
  对每条路径跑两个 analyst：
    success_analyst: 成功步骤（metric 改善）→ 提取 sop + conditions
    failure_analyst: 失败步骤（metric 退化）→ 提取 failure_mode + 根因
  每个 patch 继承：metric_delta + source_trace_ids + stage

Stage 4 — 条件化聚类（★ 核心创新，替代原版合并消除）
  按 conditions 聚类 patch：
    - 同条件 + 同方向 → 合并（去重）
    - 同条件 + 反方向 → 保留两边 + prevents 边
    - 不同条件 → 分开保留 + refines 边
  ★ 低频但 metric 改善大的 patch → 保留（反 prevalent bias）

输出：50-150 个结构化 SkillUnit（带 conditions/evidence/failure_mode）
```

### 3.4 与原版的逐点对比

| 维度 | 原版 Trace2Skill | 本研究改进 |
|---|---|---|
| Stage 3 冲突处理 | **消除合并**（三道编程栏杆） | **条件化保留**（同条件反方向→prevents） |
| 低频 patch | **prevalent bias 丢弃** | **保留**（metric 改善大即留） |
| patch 验证 | LLM 主观判断价值 | **metric_delta 客观验证** |
| 产出形态 | 1 个线性 Markdown | **50-150 个结构化单元**（为建图准备） |
| 语义冲突 | ❌ 仅物理冲突检测 | ✅ **prevents 边表达语义冲突** |
| 注入方式 | 反检索（整块拼 prompt） | **每个 skill 独立可检索** |

---

## 4. 真实蒸馏演示：从 1 条路径到 7 个结构化 skill

> 本节用项目真实数据（run `20260516_104127`，已修复数据泄露）手工蒸馏，展示产物的真实样貌。完整版见 `distilled_skills_demo.html`。

### 4.1 蒸馏原料：一条完整的演化路径

来自健康 run（0 个泄露节点），log loss 越低越好：

```
step10 draft  log loss=1.090（欠拟合，手工特征淹没 transformer）
   │
   ▼ step16 improve ↓0.703 ★大幅改善
step16 improve log loss=0.387（删手工特征，用原生 AutoModel）
   │
   ▼ step22 debug ↓0.068
step22 debug  log loss=0.320（修 head_mask bug + mean pooling）★本路径最优
   │
   ▼ step46 debug ↑0.293 ★退化
step46 debug  log loss=0.613（为避免超时换 deberta-v3-small）
   │
   ▼ step49 improve ↓0.185 部分恢复
step49 improve log loss=0.428（加 attention pooling 补救，但未回到 0.32）
```

**为什么这条路径是蒸馏金矿**：同时包含成功转折（step16）、持续优化（step22）、失误退化（step46）、部分恢复（step49）。原版 Trace2Skill 会合并成 1 个 skill 文件，**丢失转折点和冲突**。

### 4.2 蒸馏出的 7 个 skill + 1 条件 + 1 冲突

| ID | 类型 | 内容 | metric_delta（log loss） |
|---|---|---|---|
| **S1** | ✅ 正向 | 删手工特征，用原生 AutoModel | ↓0.703 |
| **S2** | ✅ 正向 | mean pooling + multi-sample dropout 防过拟合 | ↓0.068 |
| **S3** | ❌ 失败 | 为避免超时换 deberta-v3-small（退化） | ↑0.293 |
| **S4** | ⚠️ 部分 | multi-head attention pooling 补救小模型 | ↓0.185（但未回到最优） |
| **C1** | 条件 | 6h 超时限制（triggers S3, applies_when S4） | — |
| **冲突** | prevents | S2↔S3（large 模型 vs small 模型，同任务相反策略） | 客观判定 S2 胜 |
| **refines** | 关系 | S1→S2→S4（原生微调→pooling→attention） | 证据链 1.09→0.39→0.32 |

### 4.3 演示揭示的四个关键洞察

**洞察 1：工程约束是冲突的根源（C1 的价值）**
S2（large 模型，0.32）和 S3（small 模型，0.61）**共享超时约束 C1**。原版合并成一句"注意超时"，丢失"换小模型是错的"反面知识。改进版用 prevents 边保留对立。

**洞察 2：失败的 plan 原文是金矿**
S3 的 plan 写着"use a smaller, faster model (deberta-v3-small)"——看似合理但**在 spooky 上是错的**（容量不足）。原版不提取这个错误推理，改进版作为 failure_mode 保留。

**洞察 3：refines 边有 metric 证据**
S1→S2 是**有 log loss 证据的 refines 链**（1.09→0.39→0.32，持续下降），不是 LLM 主观判断。这是"metric_delta 验证 patch"的核心价值。

**洞察 4：这些 skill 直接可入双曲图**
每个 skill 有：SOP（→角度θ）、metric（→半径r）、条件（→applies_when）、失败模式（→prevents 端点）。**这才是双曲 skill 图的真实节点数据**——不是 raw SearchNode，是蒸馏后的结构化单元。

---

## 5. 真实数据验证：载体设计的数据支撑

### 5.1 粒度的数据证据（为什么中间粒度是对的）

对 124 个无泄露 SearchNode 的分析：

| 维度 | 数据 | 对载体设计的启示 |
|---|---|---|
| 总节点数 | 124 | raw 粒度太碎，需聚合 |
| stage 分布 | improve/debug 为主 | 蒸馏应聚焦 improve/debug 的转折点 |
| log loss 范围 | 0.19–1.09 | 有效/无效界限清晰（中位数 0.35） |
| 四象限分布 | 各 23-27%（均衡） | 探索充分，四类 skill 都有代表性 |

### 5.2 "新≠更有效"对载体的影响

真实数据验证（step vs log loss 相关 −0.008）：

> 搜索后期（step 大）的 skill 并不比前期 log loss 更低。

**对载体的影响**：skill 的**有效性半径必须由 metric（log loss）主导，recency 仅作 tiebreak**（权重 0.9/0.1）。这否定了"新 skill 自动更靠核心"的直觉，让载体排序**有客观 ground truth 支撑**。

### 5.3 数据泄露的处理

早期 run（如 20260514_171209）有大量 log loss < 0.1 的泄露节点（数据泄露导致虚高）。本研究：
- **蒸馏时排除**：metric < 阈值（如 0.1）的节点不参与，避免泄露污染 skill。
- **用后期 run**：20260516/17 的 run 已修复泄露（0 个泄露节点），作为蒸馏主数据源。
- **载体带 direction 字段**：`evidence.direction = "maximize=False"`，明确 log loss 越低越好，防止方向混淆。

---

## 6. 载体设计与蒸馏的关系图

```
┌─────────────────────────────────────────────────────────┐
│  raw SearchNode（124 个，无泄露）                         │
│  字段：step / stage / metric / code_summary / plan        │
└──────────────────────────┬──────────────────────────────┘
                           │
                  Stage 1: 演化路径抽取（node2parent）
                           │
                  Stage 2: metric_delta 计算（注意方向）
                           │
                  Stage 3: 并行 analyst 提 patch
                           │
                  Stage 4: ★ 条件化聚类（非合并消除）
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  结构化 SkillUnit（~50-150 个）                           │
│  字段：sop / conditions / evidence(metric_delta) /        │
│        failure_mode / implementation / trace_ids          │
└──────────────────────────┬──────────────────────────────┘
                           │
                  双曲嵌入（见 hyperbolic_skill_memory_review.md）
                  sop + conditions → 角度 θ
                  metric_delta      → 半径 r
                  conditions        → applies_when 边
                  failure_mode      → prevents 边
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  双曲 procedural skill 图（6 节点 7 边）                   │
└─────────────────────────────────────────────────────────┘
```

**数据流方向**：raw node → 蒸馏 → SkillUnit → 双曲嵌入 → 图。**蒸馏是上游，双曲是下游**。载体质量决定图质量。

---

## 7. 与现有蒸馏/载体工作的 novelty 边界

| 对比对象 | 它的载体/蒸馏 | 本研究差异 |
|---|---|---|
| **Trace2Skill** | 蒸馏成线性 Markdown + 反检索 + 合并消除 | 蒸馏成**结构化单元**（带条件/证据/失败）；**条件化保留**（非消除）；保留低频 |
| **SkillGraph** | 假设 skill 已存在，**不做蒸馏** | 本研究做蒸馏；且产出是结构化单元（非 general/task 二分） |
| **GoS** | 离线构建，**不蒸馏不演化** | 本研究在线蒸馏 + metric 验证 |
| **SkillRL** | RL 蒸馏进权重 | 本研究 context 注入（不改权重）；skill 是显式可读单元 |
| **EXG** | case 节点（轨迹快照，episodic） | 本研究是 SOP（方法规则，procedural），抽象层级高一级 |
| **A-MEM** | Zettelkasten 笔记（声明性） | 本研究是 procedural skill + 条件 + 证据 |

**核心 novelty**：本研究的载体是**第一个同时具备"procedural SOP + 客观 metric 证据 + 条件 + 失败模式 + 冲突关系"的结构化单元**。现有工作的载体要么缺证据（A-MEM/EXG），要么缺条件/失败（SkillGraph/GoS），要么合并丢失（Trace2Skill）。

---

## 8. 风险与开放问题

### 8.1 蒸馏质量依赖 LLM analyst
success/failure analyst 的 patch 提取质量依赖 LLM 推理能力。
- **对策**：用 metric_delta 反向校准——如果 analyst 提的 sop 与 metric 方向矛盾（说"有效"但 log loss 上升），降权或丢弃。

### 8.2 条件抽取的难度
conditions 的抽取依赖 LLM 从 plan/code_summary 中归纳，可能不准。
- **对策**：用 metric 反向校准——如果两个被标"同条件"的 skill metric 方向相反，说明条件抽取有遗漏（它们其实条件不同）。

### 8.3 规模问题
1 条路径 → 7 个 skill；全量 124 节点可能产出 50-150 个。这个规模是否足以建有意义的双曲图？
- **对策**：双曲空间在小规模层级图上反而优势最大（指数体积保护低频）。50-150 个 skill 是 ℍ² 的甜点规模。

### 8.4 跨任务蒸馏
spooky 蒸馏出的 skill 能否迁移到其他 MLE 任务（如 tabular/CV）？
- **开放问题**：需跨任务实验验证。条件化保留（conditions 字段）是迁移的基础——不同任务的条件不同，skill 自然分开。

---

## 9. 实验框架（针对载体与蒸馏）

### 9.1 蒸馏质量评估（离线，不烧算力）
以 `experience_kb/small-data-transformer-finetuning/insight.md` 的手工 15 条为 ground truth：
- **召回率**：自动蒸馏出的 skill 覆盖手工版的多少条（≥10/15）。
- **新发现**：自动蒸馏是否发现手工版没有的 skill（1-2 条）。
- **冲突保留**：是否保留了 2-3 对矛盾 skill（留给建图）。
- **条件标注**：100% 的 skill 有 conditions 字段。

### 9.2 载体粒度消融
对比三种载体粒度在下游检索/求解任务上的效果：
- (a) raw SearchNode（太碎）
- (b) **结构化 SkillUnit（本研究）**
- (c) 巨型 SKILL.md（太粗，反检索注入）

### 9.3 蒸馏策略消融
对比 Stage 4 的不同策略：
- 合并消除（原版 Trace2Skill）
- **条件化保留（本研究）**
- 全保留（不聚类，每个 patch 一个 skill）

---

## 10. 结论

记忆载体设计与蒸馏方法是本研究的**方法主体**。本研究提出"结构化细粒度 skill 单元"作为介于 raw SearchNode（太碎）和巨型 SKILL.md（太粗）之间的中间粒度，每个单元携带 procedural SOP + 条件 + 客观 metric 证据 + 失败模式。蒸馏采用改进 Trace2Skill 管线——核心创新在 Stage 4 的"条件化保留"（替代原版的合并消除），保留低频有效 skill 并建模双真冲突。

真实数据（124 个无泄露节点 + 1 条路径手工蒸馏出 7 个 skill）验证了：① 中间粒度的可行性；② 条件化保留能捕获合并丢失的冲突与转折；③ 每个 skill 单元可直接作为双曲图的节点。双曲几何是承载这些单元的结构，但**载体的形态与蒸馏方法是 upstream 的方法 novelty**——没有好的载体，再好的几何也无用。

---

## 参考文献

### 蒸馏 / skill
1. **Trace2Skill**: arXiv:2603.25158（蒸馏起点，本研究改进对象）
2. **SkillRL**: arXiv:2602.08234（flat skill bank + RL，前作）
3. **SkillGraph**: arXiv:2605.12039（skill 图，假设 skill 已存在不蒸馏）

### 记忆载体
4. **EXG**: arXiv:2605.17721（case 节点，episodic）
5. **A-MEM**: arXiv:2502.12110（Zettelkasten 笔记，declarative）
6. **GoS / Graph-of-Skills**: OpenReview HfGDY8mV67（executable skill，不蒸馏）

### 双曲结构（下游）
7. **HyperbolicRAG**: arXiv:2511.18808（双曲 RAG，无监督 depth）
8. 见 `hyperbolic_skill_memory_review.md`（双曲结构层综述）

### 冲突 / benchmark
9. **MemConflict**: arXiv:2605.20926（冲突评估）
10. **Reward Hacking**: ICLR 2026（metric hacking 对验证的影响）

### 项目内部
- `mlevolve/runs/20260516_104127_spooky-author-identification`：蒸馏主数据源（健康 run）
- `paper-skills/experience_kb/trace2skill-distilled/SKILL.md`：原版蒸馏产物（1 个巨型文件，对比对象）
- `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md`：手工 15 条 ground truth
- `distilled_skills_demo.html`：本研究蒸馏演示（7 个结构化 skill）
- `real_skill_poincare.html`：124 节点四象限验证
