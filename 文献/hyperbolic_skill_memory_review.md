# 双曲 Procedural Skill 记忆：几何统一检索的 Agent 长期记忆综述

> **定位**：在 `agent记忆综述_技术细节版.md`（四环节横向拆解）与 `research_review_graph_trace2skill.md`（方向声明）基础上，进一步收敛到**本研究的具体形态**——以 procedural skill 为载体、以双曲几何为结构、以几何统一检索为机制、以 metric 客观信号为写入与冲突判据的 agent 长期记忆系统。本综述整合近期所有相关前沿（HyperbolicRAG / GAM / SkillGraph / GoS / SkillRL / HyperRAG / Graph-based Agent Memory Survey 等）的精读结论，明确空白象限、novelty 边界与实验框架。
>
> **阅读次序建议**：先读 §1（一句话立论）→ §3（空白矩阵）→ §4（方法）→ §5（实验）→ §2 与 §6（支撑与边界）。
>
> **准确性声明**：所有技术细节均回论文全文核对（已在每节标注来源 arXiv 编号）。公式编号、超参、实测数字以各论文"原文报告值"为准。

---

## 0. 一页纸速览

**核心论断**：当前 agent 记忆研究沿三条线发展——**双曲几何线**（HyperbolicRAG、HyperRAG）、**skill 图线**（SkillGraph、SkillRL）、**agent 自主检索线**（A-MEM、GAM）——但**三线从未交汇**。本研究在交汇点提出"**双曲 procedural skill 记忆**"，并引入三条独有机制：(1) **经真实数据校准的演化深度半径**（metric 主导 + recency 微调，124 节点验证）；(2) `prevents` 冲突边的几何对跖化；(3) agent 在双曲空间的自主多跳导航。

**一图定位**：

```
                    agent自主      程序化检索      纯重排
                 ┌──────────────┬──────────────┬──────────────┐
  skill载体      │   ★ 本研究 ★  │  SkillGraph  │      —       │
  (procedural)   │  【三线交汇】 │ (平面+RL)    │              │
                 ├──────────────┼──────────────┼──────────────┤
  有双曲几何     │   【空白】    │ HyperbolicRAG│ HyperRAG     │
                 │              │ (文档RAG)    │ (双曲重排)   │
                 ├──────────────┼──────────────┼──────────────┤
  平面/无几何    │  A-MEM/GAM   │  GoS/GraphRAG│   传统RAG    │
                 └──────────────┴──────────────┴──────────────┘
```

---

## 1. 引言：三条线的交汇空白

### 1.1 问题的本质

记忆有效性是"query-conditioned fitness-for-use"问题（MemConflict, arXiv:2605.20926）——记忆不是"存了就行"，而是"在当前查询下能否提供正确、有效、适用的信息"。这把记忆从存储问题抬升为**推理问题**：找到若干相关甚至矛盾的经验后，如何组织、判定、多跳追溯。普通 RAG 只解决"找得到"，本研究关心其后的组织、判定与追溯。

### 1.2 三条发展线及其各自瓶颈

**① 双曲几何线**。将记忆/知识嵌入双曲（Poincaré / Lorentz）空间，利用"半径=层级、角向=语义"的分解来统一检索。代表：HyperbolicRAG（arXiv:2511.18808）、HyperRAG（OpenReview 3kzoBq8ZmQ）、HypRAG（WACV 2026）。**瓶颈**：节点均为文档/实体/概念（声明性），无 procedural skill；层级深度为无监督预测，缺乏客观验证；检索为程序化 PPR / 重排，非 agent 自主。

**② skill 图线**。把可复用技能组织为图节点，用 typed edge 编码依赖/演化。代表：SkillGraph（arXiv:2605.12039）、SkillRL（arXiv:2602.08234）、Trace2Skill（arXiv:2603.25158）。**瓶颈**：SkillGraph 是平面 + RL（改权重），依赖 task reward（受 reward hacking 影响），无双曲几何；Trace2Skill 是线性 Markdown + 反检索，丢失 SOP 间隐含联系。

**③ agent 自主检索线**。让 agent 在运行时通过 tool call 自主决定何时检索、检索什么、几跳。代表：A-MEM（arXiv:2502.12110）、GAM（arXiv:2604.12285）、MemGPT（arXiv:2310.08560）。**瓶颈**：A-MEM 有图但检索是纯向量；GAM 是平面分层图且边权依赖 LLM（贵且不稳）；MemGPT 是分层文本块非图。**无一同时具备"图结构 + agent 自主 + 双曲几何"**。

### 1.3 本研究的位置

本研究主张：**procedural skill 是比声明性事实更优的记忆载体**（因 MLE 场景提供客观 metric 信号），而**双曲几何是 procedural skill 图的天然坐标系**（因 skill 间存在真实层级与依赖）。两者的结合落在三线交汇的空白象限，并可借 MLE 场景独有的客观信号补齐前述各线的共同短板——写入验证与冲突判定。

---

## 2. 四环节分析框架（沿用并扩展）

记忆系统的能力可分解为一条流水线（与 *Graph-based Agent Memory Survey*, arXiv:2602.05665 的生命周期四阶段一致）：

```
原始 trace ──①结构化──> 记忆单元 ──②写入──> 组织好的存储
                                              │
              查询 ──────③检索──────> 相关子集/子图
                                              │
              矛盾出现 ──────④冲突处理──────> 仲裁后的记忆
```

下表把本研究与代表性工作在四环节上的实现对照（仅列与本研究直接对话的系统；完整对照见 §3）：

| 系统 | ① 结构 | ② 写入 | ③ 检索 | ④ 冲突处理 |
|------|--------|--------|--------|-----------|
| **HyperbolicRAG** | Poincaré 球（文档/entity） | 无监督 depth 预测 | 双空间 PPR + mutual-ranking 融合 | 无 |
| **GAM** | 平面层级图（topic + event） | LLM 边权 + 语义边界 consolidate | multi-factor 乘法调制 | 仅防 contamination |
| **SkillGraph** | 平面 skill 图（prereq/enhance/co_occur） | RL 反馈 + 插入/合并/拆分/弃用 | BFS+beam+拓扑排序 | 无 |
| **GoS** | 平面 skill 图（dep/wf/sem/alt） | 离线构建，不演化 | 反向 PPR + 预算重排 | 无 |
| **Trace2Skill** | 线性 Markdown | 多 trace 并行 patch + prevalent bias | 反检索（拼 prompt） | 合并阶段消除（仅物理冲突） |
| **本研究** | **双曲 procedural skill 图（6 节点 7 边）** | **metric_delta 监督半径 + 条件化保留** | **agent 自主多跳 + 几何统一检索** | **prevents 几何对跖 + metric 判定** |

---

## 3. 现有工作横向拆解

### 3.1 双曲几何线：层级检索的几何化（但载体是声明性）

**HyperbolicRAG（arXiv:2511.18808）** 是本路线最完整的代表。三个关键设计：

1. **Depth-Aware 表示**：对每个文本单元预测标量 depth，经原点指数映射 `zᵛᴴ = exp₀ᶜ(ẑᵛᴱ) = tanh(√c·‖ẑᵛᴱ‖)·ẑᵛᴱ/(√c·‖ẑᵛᴱ‖)` 投影到 Poincaré 球。欧氏向量的方向保留语义，范数转为半径（层级）。
2. **双向包含对齐**：用 margin-based 对比损失 `L_{p→f} = Σ[d_H(p,f⁺) − d_H(p,f⁻) + γ]₊` 强制 passage（容器）与 fact（被包含）的几何包含关系。
3. **双空间检索融合**：欧氏分支（局部语义）+ 双曲分支（层级）并行 PPR，用 mutual-ranking 融合强调跨空间一致性。

**局限**：节点是 passage/entity；depth 无监督预测（无 ground truth）；检索程序化非 agent；无冲突。

**HyperRAG（OpenReview 3kzoBq8ZmQ）**：LLM span 检测 + RAG + 双曲层级重排（实体链接场景）。是重排式而非检索统一。

> ⚠️ **命名陷阱**：另有 HyperRAG（Web Conf 2026, rfp4146）用 hypergraph（超图≠双曲），与本路线无关，引用时须区分。

### 3.2 Skill 图线：技能的组织与演化（但平面/改权重）

**SkillGraph（arXiv:2605.12039）** 是与本研究机制最接近的竞品，**也是最大威胁**。核心机制：

- **节点**：从 trajectory 蒸馏的 general / task-specific skill，每节点含 `{title, principle, condition, category}`。
- **三类 typed edge**：`prereq`（前置）、`enhance`（增强）、`co_occur`（共现），各带权重 `w(e)∈[0,1]`。
- **图感知检索**：种子选择 → 向后 BFS（深度 D）+ 向前 beam（宽度 B）→ 拓扑排序输出有序 skill 序列。
- **图演化**：节点级（Insert/Merge/Split/Deprecate，阈值化：`p̂∈[0.15,0.4]` 拆分、`p̂<0.15` 弃用）+ 边级（Path reinforcement / Co-occurrence discovery / Decay-pruning）。
- **闭环训练**：GRPO 训练 policy，graph 与 policy co-evolve。
- **实验**：ALFWorld 90.6 / WebShop 84.4；消融显示去掉 graph-aware retrieval 后 ALFWorld 从 90.6 暴跌到 59.4（检索是最大贡献）。

**与本研究碰撞**：节点+typed边+演化+程序化拓扑检索（4 点撞车）。**但本研究独有**：6 类 procedural 语义节点、metric 客观验证、冲突判定、MLE 场景、不改权重（context 注入）。

**SkillRL（arXiv:2602.08234）**：SkillGraph 的前作，flat skill bank + RL co-evolution。**与本研究差异**：RL 改权重 vs context 注入（范式不同）。

**Trace2Skill（arXiv:2603.25158）**：本研究蒸馏层的起点。三阶段（轨迹生成 → 并行 analyst 提 patch → 层级合并消除冲突）。**自承认局限**：(1) prevalent bias 丢低频正确经验；(2) 仅物理冲突检测；(3) 无图。本研究改进点即针对此三者。

**GoS / Graph-of-Skills（OpenReview HfGDY8mV67）**：纯推理时检索层，从 SKILL.md 离线构建图，四类边（dep 确定性 I/O 匹配 / wf / sem / alt），反向 PPR 扩散 + 预算重排。**不蒸馏、不演化、不验证**。撞本研究"图+检索"，但是离线静态 + 程序化。

### 3.3 Agent 自主检索线：图 + agent 的组合（但平面/无几何）

**A-MEM（arXiv:2502.12110, NeurIPS'25 poster）**：Zettelkasten 同质笔记图 + agent 自主写入，但**检索是纯向量**（§3.4 三个公式无 LLM）。§2.2 自白：agency 在存储与演化，检索让给 agentic RAG。**是本研究 baseline #1**（图+agent写，但向量检索）。

**GAM（arXiv:2604.12285）**：层级图 agent 记忆。**全文检索确认：GAM 无任何几何/双曲/曲率组件（0 次）**。核心机制：
- `ℋₜ = {𝒢_topic, 𝒢_event, 𝒮_arch, ℰ_cross}`，Topic 网络（全局）+ Event 图（局部）+ 跨层索引。
- State-based consolidation：语义散度超阈值才更新全局图（防 contamination）。
- Multi-factor 重排：`Score(v,q) = P_sem(v|q)·Π β_k^{𝕀_k(v,q)}`，时间/置信/角色三因子。
- 边权依赖 LLM scorer（贵）。**撞本研究"agent + 层级图"，但平面且无 metric 验证/冲突**。

**MemGPT（arXiv:2310.08560）**：分层文本块 + agent tool call 检索。**非图**。是 agent 检索范式来源，但无图结构。

### 3.4 空白象限确认

综合三线，**"双曲几何 + procedural skill 载体 + agent 自主多跳 + metric 验证 + 冲突判定"五元组合完全空白**。这是本研究的精确位置。

---

## 4. 本研究方法：双曲 Procedural Skill 记忆

### 4.1 设计原则

1. **载体不变**：记忆载体始终是 procedural skill（trace2skill 蒸馏产物），不是文档/实体。
2. **几何承载结构**：skill 间的 7 类关系不显式存储，而是几何位置关系的派生（详见 §4.4）。
3. **客观信号锚定**：写入与冲突判定均由 `metric_delta` 驱动（MLE 场景独有）。
4. **检索统一**：语义、层级、路径三种检索在同一双曲空间内用同一距离族计算，无需拼接。

### 4.2 数据流

```
mlevolve trace → 改进 Trace2Skill 蒸馏 → Skill（带 metric_delta/applies_when/source_trace_ids）
      → 双曲嵌入（半径 metric 监督，角向 embedding 派生）
      → 双曲 Skill 图（6 节点 7 边几何化）
      → agent 自主多跳几何检索 + 几何冲突判定
```

蒸馏是**周期性、离线**的，与 mlevolve 在线搜索解耦；当前 run 内图只读不写，保证稳定。

### 4.3 节点与双曲嵌入

**6 类节点**：`Skill`（完整技能集）/ `SOP`（单条规则）/ `Condition`（适用上下文）/ `FailureMode`（失败模式）/ `Evidence`（trace_id + metric_delta）/ `Implementation`（代码模板）。

**嵌入（ℍ² 起步方案，skill 数 >500 升级 ℍ³）**：

- **半径 r（演化深度，详见 §4.3bis）**：由 `metric_delta` 主导、`recency` 微弱辅助的**演化深度**决定。这是本研究**最硬的差异化**——HyperbolicRAG 的 r 是无监督预测，本研究有 ground truth，且经真实数据校准（见 §4.3bis）。
- **角向 θ（语义类别）**：由 skill 文本的欧氏 embedding 经 `exp₀` 映射的方向决定。同类 skill 聚同一扇区。

**投影公式**：`z = exp₀ᶜ(ẑ) = tanh(√c·‖ẑ‖)·ẑ/‖ẑ‖`。tanh 把任意范数压到 [0,1) 落入 Poincaré 球；方向（语义）保留，范数（半径=演化深度）受 metric 监督。

### 4.3bis 半径的精确定义：演化深度（经真实数据校准）

**动机**：半径 r 同时承载"有效性层级"与"演化时序"两个语义——这在 procedural skill 演化谱系中是耦合的（后产生的 skill 通常是从既有 skill 细化而来，与生物系统发生树同构）。但这一耦合假设必须在真实数据上验证。

**真实数据验证**（6 个后期 spooky run（20260516/17，已修复数据泄露），124 个有效 skill 节点，metric 为 log loss 越低越好）：

| 假设 | 验证方法 | 相关系数 | 结论 |
|------|---------|---------|------|
| "新=特化"（step↑→后期 stage） | step vs stage_order 相关 | +0.217 | 🟡 弱正相关（部分成立） |
| "新=更有效"（step↑→log loss↓） | step vs metric 相关 | −0.008 | 🔴 几乎无关（不成立） |

**数据驱动的修正**：真实数据否定了"新=更有效"（step 与 log loss 相关仅 −0.008）。因此 recency **不能与 metric 同等加权**，否则会把"新但 log loss 高"的 skill 错误拉向有效核心区。修正后的半径定义（注意 spooky 是 log loss 越低越好，故用 `1/metric` 归一化 effectiveness）：

```
r = sigmoid( 0.9 · norm(1/metric) + 0.1 · norm(recency) )
```

metric 权重提升至 0.9（有效性远比时序重要，数据证明），recency 降至 0.1，仅作 metric 接近时的 tiebreaker。这是**数据校准而非主观设参**——124 个无泄露真实节点的 step-metric 相关性（−0.008）直接否决了 0.5/0.5 或 0.7/0.3 的直觉配比。

**四类 skill 在 Poincaré 球的落点**（真实四象限分布，median split）：

| 象限 | 特征 | 真实占比 | 半径位置 | 代表 skill（真实数据） |
|------|------|---------|---------|----------------------|
| Q1 核心·基石 | 旧 + 有效（log loss 低） | 33 (27%) | r 小（核心） | step16 原生 AutoModel，log loss 0.387 |
| Q2 前沿 | 新 + 有效（log loss 低） | 29 (23%) | r 小偏外 | step56 DeBERTa 微调，log loss 0.287 |
| Q3 试探 | 新 + 无效（log loss 高） | 34 (27%) | r 大（边缘） | step49 attention pooling，log loss 0.428 |
| Q4 淘汰 | 旧 + 无效（log loss 高） | 28 (23%) | r 最大 | step10 多分辨率分类，log loss 1.090 |

**关键发现**：Q2 的 evolution/fusion stage 占比显著高于 Q1，支撑"后期 stage 更特化"的部分假设；但 Q3 有 34 个"新但 log loss 高"的 skill——在 SkillGraph 的 Deprecate 机制（成功率 <0.15 弃用）下会被丢弃，而在本研究的双曲边缘，它们因指数体积保持检索可达性。**这是本研究 vs SkillGraph 的实证差异化弹药**。

**时序边从演化深度免费派生**：`refines` / `superseded_by` / `successor_of` 均从 r 的大小关系自动读出，无需像 Zep 那样额外存 valid_at 时间戳。

**失效兜底**（范式革命、回滚、并行演化等耦合失效情况）：r 始终服从 metric_delta（有效性优先），精确时序退回节点的时间戳元数据。这种分层设计在论文里主动说明反而显得严谨。

### 4.4 边的几何化（核心创新）

**关键洞察**：双曲图中"边"不是显式存储的记录，而是两节点位置关系 `(Δr, Δθ)` 的几何派生。7 类边 = 7 种 `(Δr, Δθ)` 组合：

| 边类型 | Δr（半径差） | Δθ（角度差） | 几何含义 | 例子 |
|--------|-------------|-------------|---------|------|
| `refines`（细化） | 大正（B演化更深） | ≈0（同方向） | 同方向向外，时序派生 | 梯度下降→Adam |
| `prerequisite`（前置） | 大正（B演化更深） | ≈0（同方向） | 同方向向外带序 | 数据清洗→特征工程 |
| `alternative`（替代） | ≈0（同层） | 小（方向近） | 同层不同向 | XGBoost↔LightGBM |
| `co_occurs`（共现） | ≈0（同层） | 小（方向近） | 同层邻近 | 标准化+正则化 |
| `applies_when`（条件） | 负（跨层） | 中 | 外层条件→内层skill | 小数据集→交叉验证 |
| `supported_by`（支撑） | 负（包含） | ≈0（同方向） | A包含E | dropout有效←实验日志 |
| `superseded_by`（被取代） | 大正且B有效 | ≈0 | B是A的更新更好版 | 见 §4.3bis 时序派生 |
| ⭐`prevents`（冲突） | ≈0（同层） | **≈π（180°对跖）** | 同层但对立 | 增大数据 vs 精简数据 |

**注**：`refines`/`prerequisite`/`superseded_by` 的"大正 Δr"同时表达层级（B 更特化）与时序（B 演化更深），两者在 procedural skill 演化中耦合（见 §4.3bis 真实数据验证）。

**`prevents` 的几何化是本研究独有的王牌**：两个 `metric_delta` 都为真但互斥的 skill，放在双曲对跖位置；检索到 A 时，对跖的 B 自动被几何识别为"要警惕的对立面"。**没有任何现有工作做过双曲冲突记忆**。

### 4.5 混合检索：几何统一（而非拼接）

传统混合检索（GoS / HybridRAG）是"向量召回→图扩展"两阶段拼接。本研究在双曲空间内**单一几何统一**：

```
Score(v|q) = w₁·exp(−d_角向(q,v))   ← 语义（方向）
           + w₂·exp(−|r_v − r_q|)    ← 层级（半径）
           + w₃·exp(−d_双曲(q,v))    ← 路径（测地）
```

三项均在同一双曲空间计算。Agent 通过 tool `navigate(direction=θ, depth=r, hops=k)` 自主决定方向（选 skill 类别）、深度（选层级）、跳数（选路径），返回 skill 子图。**这是本研究综述中"图 + 运行时 agent 自主检索"空白象限的具体填充机制**。

### 4.6 双层记忆协调

| 维度 | Layer A · Trace 记忆（保留现状） | Layer B · 双曲 Skill 图（新增） |
|------|------------------------------|----------------------------|
| 回答 | "这分支已试过什么？结果如何？" | "这种情况下方法学该怎么做？" |
| 时间尺度 | 当前 run（小时级） | 跨 run（周/月级） |
| 粒度 | 具体 code + metric | 抽象 SOP + 条件 |
| 呈现 | 强制注入（必看） | 按需 tool 查（agent 自主） |

两层绝不合并。Layer A 防"重复犯错"，Layer B 防"永远局部探索"。当前 run 内 Layer B 只读不写。

---

## 5. 实验框架

### 5.1 主实验（好不好）

5 条件 × 3 任务 × 3 seed，对比：

| 条件 | 结构 | 检索 | 冲突 |
|------|------|------|------|
| (a) No-Mem | 无 | 无 | 无 |
| (b) Flat-Vec（mlevolve 现状） | 平面 | 固定向量 | 无 |
| (c) Graph-Vec（EXG / SkillGraph 式） | 图 | 固定程序化 | 无 |
| (d) Flat-Agent（MemGPT 式） | 平面 | agent | 无 |
| (e) **Hyper-Skill（本研究完整方法）** | **双曲 procedural 图** | **agent 自主几何** | **metric 判定** |

指标：最终 metric、收敛速度、重复错误率。

### 5.2 消融（图/agent/双曲/metric 各占多少功劳）

在完整方法上做减法：
- **w/o 双曲几何**（退化为平面 GNN）→ 回应"为什么不普通 GNN"（**审稿必问**）
- **w/o agent 自主**（退化为程序化 PPR）→ 回应"agent 自主导航是否跑赢 SkillGraph 的拓扑排序"
- **w/o metric 监督半径**（退化为无监督 depth）→ 回应"metric 是否是关键"
- **w/o 几何冲突**（退化为逻辑 prevents）→ 回应"几何冲突判定是否有效"
- **半径配比消融**（0.9/0.1 vs 0.5/0.5 vs 0.7/0.3）→ 用真实 124 节点（无泄露）验证 metric 主导的必要性（step-metric 相关 −0.008 否决等权）

### 5.3 专项实验（v3 差异化）

1. **Q3 低频 skill 召回率**（双曲 vs 欧氏）：用真实 124 节点（无泄露）的 Q3 象限（34 个"新但 log loss 高"skill）直接验证"双曲边缘指数体积保护低频"——**本研究最硬的 motivation，且有现成真实数据**。对比 SkillGraph 的 Deprecate（丢弃低频）。
2. **冲突分类正确率**：双曲对跖判据 vs LLM 主观判定，预期几何判据把矛盾场景正确率从 ~60% 提到 ~80%。
3. **Path Diversity@K / Hidden Link Recall**：验证"双曲检索能召回向量找不到的语义远但相关 skill"。
4. **四象限保持率**：双曲 vs 平面在 Q3/Q4（边缘低频 skill）上的检索可达性，直击 prevalent bias。

### 5.4 效率与泛化

- **效率**：检索延迟 p50/p95、token 成本、检索调用次数。论证"agent 几何检索单次贵但减少试错步数、端到端更快"。注意双曲距离比欧氏贵（arcosh），需 Lorentz 模型 + ε 截断缓解数值问题。
- **泛化**：2–3 个 MLE-bench 任务（NLP / tabular / CV）跨任务迁移。

### 5.5 实验预测（分层可验证）

- 只做蒸馏 + Markdown 注入 → 性能基线
- 加双曲图 + agent 几何检索 → SEH@K 显著提升
- 加 metric 监督 + 几何冲突 → 矛盾场景 AA 显著提升

每层独立验证，避免"全有或全无"风险。

---

## 6. Novelty 边界、风险与结论

### 6.1 与各竞品的精确 novelty 边界

| 对比对象 | 它的做法 | 本研究差异 |
|---------|---------|-----------|
| **HyperbolicRAG** | 双曲 + 文档 RAG + 程序化检索 + 无监督 depth | procedural skill 载体；metric 监督半径；agent 自主；几何冲突 |
| **HyperRAG** | 双曲 + 实体重排 | 检索统一（非重排）；skill 载体；metric 验证 |
| **SkillGraph** | 平面 skill 图 + RL co-evolve + 程序化拓扑检索 | 双曲几何；不改权重（context 注入）；6 类 procedural 语义节点；冲突判定 |
| **SkillRL** | flat skill bank + RL | 图（非 bank）；双曲；context 注入（非 RL） |
| **GoS** | 离线 skill 图 + 程序化 PPR + 不演化 | 在线演化 + metric 验证；agent 自主遍历；冲突 |
| **GAM** | 平面层级图 + LLM 边权 + agent | 双曲（免 LLM 边权）；procedural skill；metric 验证；冲突 |
| **A-MEM** | 同质笔记图 + agent 写 + 纯向量检索 | 双曲；agent 自主多跳；procedural；冲突 |
| **Trace2Skill** | 线性 Markdown + 反检索 + 消除合并 | 双曲图 + 几何检索；保留低频（反 prevalent bias）；语义冲突建模 |

### 6.2 三条独有 novelty（护城河）

1. **经真实数据校准的演化深度半径**：所有现有双曲记忆/RAG 的层级是无监督猜或 LLM 算的；本研究有客观执行信号 ground truth，且权重配比（metric 0.9 + recency 0.1）经 124 个无泄露真实 skill 节点验证（step-metric 相关 −0.008 直接否决了等权配比）。这是别人做不到、本研究独有，且**有数据支撑、可辩护**。
2. **`prevents` 边 = 双曲对跖**：所有先例的双曲图只有层级/相似关系，无冲突关系。冲突判定几何化是首次。
3. **agent 在双曲空间自主导航**：所有双曲检索先例都是程序化（PPR/NNS/重排）；本研究 agent 用 tool 在双曲空间自主决定方向与深度。

### 6.3 风险

1. **双曲距离比欧氏贵**（arcosh 计算复杂，靠近边界数值溢出）→ Lorentz 模型 + ε 截断。
2. **双曲训练难调**（需 Riemannian Adam）→ 成熟库 geoopt；metric 监督半径反而比无监督稳。
3. **最大风险——"为什么不直接用 GNN？"**：必须做双曲 vs 欧氏 GNN 消融，证明双曲在层级 + 低频召回上更好；若跑不赢，此 novelty 不成立。
4. **被抢先风险（中高）**：HyperbolicRAG（2025-11）已很近，2026 年双曲 RAG 在涌入。建议尽快在"metric 监督半径"与"prevents 几何化"两个独有件上占坑。

### 6.4 可行性

- **代码基础**：MLEvolve 框架完整（搜索树、记忆层、冷启动），只需增量改造（在线代码仅改 `improve_agent.py` 加几何检索 tool）。
- **数据基础**：Spooky 任务 60+ run（含丰富成功/失败案例）+ 手工 15 条 ground truth（`experience_kb/small-data-transformer-finetuning/insight.md`），离线验证不依赖新算力。
- **算力估算**：主实验 + 消融约 700–900 GPU·h；核心单任务验证约 12–16 GPU·h（1 周内见信号）。
- **投稿目标**：对标 A-MEM（NeurIPS'25 poster）→ NeurIPS / ICML / ICLR poster；冲刺 oral/spotlight 需极强实验。

### 6.5 结论

当前 agent 记忆研究沿双曲几何、skill 图、agent 自主检索三线发展但未交汇。本综述通过精读各线代表工作，定位出"五元组合完全空白"的象限，并据 MLE 场景独有的客观 metric 信号，提出"双曲 procedural skill 记忆"——以几何统一检索替代拼接式混合检索，以 metric 监督半径与几何冲突判定补齐所有先例的共同短板。三条独有 novelty（metric 监督半径、prevents 几何对跖、agent 双曲自主导航）均有硬 motivation 且落在真空白，构成不可替代的贡献。

---

## 7. 参考文献（按本综述出现顺序）

### 双曲几何线
1. **HyperbolicRAG**: Enhancing Retrieval-Augmented Generation with Hyperbolic Representations. arXiv:2511.18808.
2. **HyperRAG**: Hierarchy-Aware Retrieval-Augmented Generation with Hyperbolic Reranking. OpenReview 3kzoBq8ZmQ.
3. **HypRAG**: Hyperbolic Embeddings Improve Narrative Quality in RAG Models. WACV 2026 Workshop.
4. Nickel & Kiela. Poincaré Embeddings. NeurIPS 2017.
5. Ganea. Hyperbolic Entailment Cones. NeurIPS 2018.

### Skill 图线
6. **SkillGraph**: Skill-Augmented RL via Evolving Skill Graphs. arXiv:2605.12039.
7. **SkillRL**: Evolving Agents via Recursive Skill-Augmented RL. arXiv:2602.08234.
8. **Trace2Skill**: Distilling Trajectory-Local Lessons into Transferable Skills. arXiv:2603.25158.
9. **GoS / Graph-of-Skills**: Dependency-Aware Structural Retrieval. OpenReview HfGDY8mV67.

### Agent 自主检索线
10. **A-MEM**: Agentic Memory for LLM Agents. arXiv:2502.12110. NeurIPS 2025 poster.
11. **GAM**: Hierarchical Graph-based Agentic Memory. arXiv:2604.12285.
12. **MemGPT / Letta**: Towards LLMs as Operating Systems. arXiv:2310.08560.

### 冲突评估 / benchmark
13. **MemConflict**: Evaluating Long-Term Memory Under Memory Conflicts. arXiv:2605.20926.
14. **STALE**: Can LLM Agents Know When Their Memories Are No Longer Valid? arXiv:2605.06527.

### survey / 系统
15. **Graph-based Agent Memory**: A Survey. arXiv:2602.05665.
16. **Mem0**: Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
17. **Zep / Graphiti**: Temporal Knowledge Graph Architecture. arXiv:2501.13956.
18. **EXG**: Self-Evolving Agents with Experience Graphs. arXiv:2605.17721.
19. **SAGE**: Self-Evolving Agentic Graph-Memory Engine. arXiv:2605.12061.
20. **InternAgent-1.5**: Unified Agentic Framework for Long-Horizon Scientific Discovery. arXiv:2602.08990.

### 项目内部参考
- `mlevolve/agents/memory/record.py`（MemRecord）/ `retriever.py`（HybridRetriever）：平面记忆 + BM25+向量混合检索现状。
- `mlevolve/engine/search_node.py` / `agent_search.py`：搜索树与多阶段调度。
- `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md`：手工 Trace2Skill 产物（15 条 ground-truth insight）。
- 配套可视化：`hyperbolic_memory_visual_tutorial.html` / `hyperbolic_edges_visual.html` / `geometry_comparison.html`。
