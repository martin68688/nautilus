# Agent 记忆系统综述：现有方案全景与 MLE 经验记忆研究方向

> 面向博士导师汇报的文献综述。系统梳理当前 agent 记忆的四种核心能力（结构化 / 写入 / 检索 / 冲突处理），定位现有工作的空白象限，并基于项目代码现状（MLEvolve + paper-skills）提出可发表的研究方向。

---

## 摘要

大语言模型（LLM）agent 在长周期任务中的核心瓶颈不是单步推理能力，而是**记忆**——它无法跨任务、跨会话、跨失败有效地积累、组织和复用经验。近两年（2025–2026）涌现了大量 agent 记忆工作，但它们在**记忆结构、写入自主性、检索机制、冲突处理**四个维度上呈现高度碎片化：没有一个系统同时做到"图结构 + 真 agent 自主检索 + 基于客观信号的冲突判定"。

本综述首先沿四个维度对代表性工作进行横向拆解，揭示出三个关键空白：(1) 图结构记忆与运行时 agent 自主检索的组合尚无人做；(2) 记忆冲突处理在 static（0.44）和 dynamic（0.50）场景下普遍失败，核心原因是"找到矛盾记忆后无法客观判定信谁"；(3) 现有冲突解决的评估指标（SEH@K vs AA 的 gap）证明检索不是瓶颈，判定才是。

基于此，结合项目已有的 MLEvolve 搜索框架（含搜索树、GlobalMemoryLayer、experience_kb）与 MLE 场景独有的客观执行信号（metric + 执行条件），提出三层互补的研究贡献：**(1) 蒸馏层**——以 Trace2Skill 范式为起点，从大量 trace 中自动蒸馏出 skill，并改进其冲突消除策略以保留低频正确经验；**(2) 组织层**——将蒸馏产出的 skill 文档结构化为 **Skill Graph**（含 Skill / SOP / Condition / FailureMode / Evidence / Implementation 多类节点），作为 skill 之间关系的可遍历记忆系统；**(3) 检索层**——agent 在 mlevolve 搜索过程中通过 tool call 自主多跳查询 Skill Graph。三层构成"trace → skill → graph → agent 检索"的完整流水线，**图节点是蒸馏产物（不是 SearchNode 本身）**，蒸馏是周期性的、跨 run 的，与 mlevolve 在线搜索解耦。

---

## 1. 引言：为什么 agent 记忆是当前的核心问题

LLM agent 已经能够在单次任务中完成代码生成、工具调用、多步推理。但真实部署中的 agent 是**长周期、多任务**的：它需要记住上次怎么失败的、哪个方法有效、哪些条件变了。这正是当前 agent 从"单次执行器"升级为"持续学习者"的瓶颈所在。

记忆问题的本质，可以用 MemConflict（arXiv 2605.20926）的一句话概括：

> **Memory validity is a "query-conditioned fitness-for-use" problem.**（记忆有效性是"取决于查询的适用性"问题）

也就是说，记忆不是"存了就行"，而是"在当前查询下能否提供正确、有效、适用的信息"。这个定义把记忆从"存储问题"上升为"推理问题"。

现有工作沿四个维度展开，但各自只解决了一部分：

| 维度 | 核心问题 | 代表性失败 |
|------|---------|-----------|
| **结构化** | 记忆以什么形式组织？平面 / 分层 / 图？ | 平面向量库无法表达因果关系 |
| **写入** | 谁决定记什么、怎么连、何时更新？ | 固定管道写入缺乏自适应 |
| **检索** | 检索由谁触发、如何构造查询、几跳？ | 固定向量检索无法做多跳因果追溯 |
| **冲突处理** | 矛盾记忆如何解决？覆盖 / 消除 / 保留？ | LLM 主观判定在 static/dynamic 上准确率仅 0.44/0.50 |

本综述按这四个维度组织，逐一定位空白，最后给出研究方向。

---

## 2. 记忆方案全景：四个维度的横向拆解

### 2.1 维度一：记忆结构（平面 / 分层 / 图）

记忆的物理组织形式决定了能表达什么关系。

#### (a) 平面记录库（Flat Records）

最基础的形式：每条记忆是一个独立记录，靠向量相似度检索。

- **代表**：早期 RAG、ReasoningBank（arXiv 2509.25140）
- **特点**：简单、快，但无法表达"记忆 A 是记忆 B 的改进版"这种关系。
- **局限**：在需要因果追溯的任务上（如"这个错误以前怎么修的"）召回不全。

#### (b) 分层文本块（Tiered Blocks）

借鉴操作系统的内存层级，把记忆分成 in-context / recall / archival 三层。

- **代表**：**MemGPT / Letta**（arXiv 2310.08560）
- **结构**：core memory（始终在上下文内的 persona/human 块）→ recall memory（对话历史）→ archival memory（外部向量库）。
- **特点**：解决了上下文窗口有限的问题，agent 可以在层间搬运信息。
- **局限**：**不是图**。层内仍是文本块，无法表达节点间的因果或相似关系。多跳推理能力弱。

#### (c) 图结构（Graph）

把记忆组织成节点和边，节点是记忆单元，边编码关系。

- **代表**：**A-MEM**（arXiv 2502.12110，NeurIPS 2025 poster）、**SAGE**（arXiv 2605.12061）、**EXG**（arXiv 2605.17721）、**Mem0g**（arXiv 2504.19413）、**Zep/Graphiti**（arXiv 2501.13956）

图结构内部又分**同质图**和**异构图**：

| 系统 | 节点 | 边类型 | 同质/异构 |
|------|------|--------|----------|
| A-MEM | Zettelkasten note | similar_to（语义链接） | 同质 |
| EXG | case + task anchor | contain / similar_to / fixed_by | 异构 |
| SAGE | entity + memory fragment | 异构关系边 + 结构角色 | 异构 |
| Mem0g | entity | 关系三元组 | 同质 |
| Zep | entity + event | 时序关系边 | 异构 |

**关键观察**：图的强弱取决于**边类型的信息量**。A-MEM 的 similar_to 是最弱的（纯语义相似）；EXG 的 fixed_by（错误-修复）已经有因果意味；SAGE 的结构角色（bridge/hub/community）最强但依赖训练。**没有一个系统用 MLE 执行派生的因果边**（如 `improves_over` 由 metric delta 决定、`fails_due_to` 由错误根因决定）。

---

### 2.2 维度二：写入自主性（固定 / agent 写 / 训练写）

"写入"指记忆如何被创建、链接、更新。

#### (a) 固定管道写入

系统按预设规则提取记忆，agent 不参与决策。

- **代表**：标准 RAG、EXG（case 抽取是固定流程）、本项目 mlevolve 的 `GlobalMemoryLayer`（每个 SearchNode 自动入库）
- **特点**：稳定、快，但写入质量不可控，且无法自适应。

#### (b) Agent 自主写入

LLM 通过 tool call 决定记什么、怎么连、何时更新。

- **代表**：**MemGPT**（`core_memory_append` / `core_memory_replace` tool）、**Mem0**（ADD/UPDATE/DELETE/NOOP tool call）、**A-MEM**（LLM 自主构造 note + 判断建边 + 演化更新）
- **关键区分**：MemGPT 和 Mem0 的写入 tool 是显式 function call；A-MEM 的写入是 LLM 在记忆管理阶段的自主决策（构造 keywords/tags/links）。
- **A-MEM 的自白（§2.2）**：它明确指出自己的 agency 在"存储和演化"，而非"检索"——这句话划清了写入 agent 和检索 agent 的边界，对本研究方向的定位至关重要。

#### (c) 训练式写入

用强化学习训练一个 writer 策略，根据下游反馈优化写入。

- **代表**：**SAGE**（writer 是 policy model，reward 来自 reader 的检索效果）、**Trace2Skill**（writer 是层级合并算子 M，用 prevalent pattern bias 过滤）
- **特点**：写入质量最高（有 reward 闭环），但需要训练，且推理时是固定流程。

#### 三种写入策略的对比

| 策略 | 写入质量 | 自适应性 | 训练成本 | 代表 |
|------|---------|---------|---------|------|
| 固定管道 | 低 | 无 | 无 | EXG, mlevolve现状 |
| Agent 自主 | 中 | 高 | 无 | MemGPT, Mem0, A-MEM |
| 训练式 | 高 | 中 | 高 | SAGE, Trace2Skill |

**对本项目的启示**：MLE 场景天然有 writer——搜索循环本身就是序列写入过程（每个 SearchNode 是一步）。不需要额外训练 writer，但可以借鉴 A-MEM 的 agent 自主写入（让 LLM 判断建什么边）+ SAGE 的 reward 闭环（用检索反馈评估写入质量）。

---

### 2.3 维度三：检索机制（固定向量 / agent 自主 / 训练式）

检索是记忆系统的"读"端，决定了记忆能否被有效利用。这是本研究方向最核心的维度。

#### (a) 固定向量检索（非 agent）

系统按固定管道触发检索，用向量相似度返回 top-k。

- **代表**：**A-MEM**（§3.4，公式 8-10：all-minilm 编码 + 余弦相似度 + top-10）、**EXG**（FAISS 种子 + 图遍历 rerank）、**Mem0**（嵌入相似度）、**Zep**（后端三步 pipeline）
- **关键事实**：A-MEM 虽然标题叫"Agentic Memory"，但**检索是纯向量**（§3.4 三个公式无 LLM 参与）。它在 §2.2 明确承认：检索的 agency 让给了 agentic RAG，自己的 agency 在存储和演化。
- **局限**：无法做多跳因果追溯（"错误 → 根因 → 修复 → 适用条件"这种链式推理）。

#### (b) Agent 自主检索（运行时 tool call）

LLM 在运行时通过 tool call 自主决定何时检索、检索什么、检索几次。

- **代表**：**MemGPT**（`archival_memory_search` tool，agent 自己决定调不调）、**LongMemEval-V2 的 AgentRunbook-C**（coding agent 在沙箱里写代码检索轨迹文件）
- **特点**：灵活性最高，agent 能根据当前情况动态构造检索策略。
- **局限**：MemGPT 的记忆不是图（分层块）；AgentRunbook-C 的记忆是文件不是图。**两者都没做"图 + agent 检索"的组合**。

#### (c) 训练式检索器

训练一个检索模型，推理时做一次前向传播。

- **代表**：**SAGE**（Graph Foundation Model 做软寻址 + 结构传播）
- **特点**：训练时 agent 化（有 writer-reader reward 闭环），但**推理时不是运行时 agent 决策**，而是模型前向。快但不灵活。
- **优势**：能识别图结构角色（bridge/hub/community），比纯向量强。
- **局限**：无法根据当前失败的具体特征动态调整检索策略。

#### 检索机制的空白象限分析

把"记忆结构"和"检索自主性"做成二维矩阵：

```
                    记忆结构
            平面/分层          图结构
        ┌──────────────┬──────────────┐
   系统  │  mlevolve现状 │   A-MEM      │
   固定  │  GlobalMemory │  (图+agent写 │
   检索  │  (向量+BM25)  │  但向量检索) │
        ├──────────────┼──────────────┤
检索 →   │   MemGPT     │              │
agent    │ (agent检索   │  ★ 空白 ★    │
自主     │  但分层块)   │ (图+agent检索)│
        └──────────────┴──────────────┘
                          ↑
                    SAGE 在此象限上方
                    （训练式检索器，非运行时agent）
```

**结论**：图结构 + 运行时 agent 自主检索的组合，目前在文献中是空白。这是本研究方向主贡献的精确位置。

---

### 2.4 维度四：冲突处理（覆盖 / 消除 / 预防 / 保留）

当新记忆与旧记忆矛盾时，系统如何处理？这是 MemConflict（arXiv 2605.20926）系统评估的维度。

#### MemConflict 的三类冲突定义

| 冲突类型 | 含义 | 例子 | 正确处理 |
|---------|------|------|---------|
| **Dynamic** | 后来的**真实更新**取代早期状态 | 用户搬家了 | 覆盖旧值 |
| **Static** | 后来的**错误矛盾**不应覆盖不变属性 | 出生地被误说错 | 坚持旧值 |
| **Conditional** | 多个值在**不同条件下**都有效 | 早上咖啡/晚上牛奶 | 按条件选 |

#### 四种冲突处理路线

| 路线 | 思想 | 代表 | 局限 |
|------|------|------|------|
| **A. 写入时预防** | 阻止错误/矛盾进入存储 | TMMA（NeurIPS 2025 Workshop） | 防 false memory，但会把条件化正确矛盾也挡掉 |
| **B. 更新时覆盖** | 新来时 LLM 决定 UPDATE/DELETE | Mem0, Letta | 强制二选一，丢失条件 nuance |
| **C. 消除合并** | 批量合并成一条无冲突记忆 | Trace2Skill（层级合并 + prevalent bias） | 丢弃低频正确经验；只做物理冲突检测（行号），漏语义冲突 |
| **D. 检索时选择** | 都保留，检索时按条件选 | **（无人做）** | 需要条件建模，现有系统不建模 |

#### MemConflict 的实测数据（6 个系统，Table 3）

| 系统 | Dynamic AA | Static AA | Conditional AA |
|------|-----------|----------|---------------|
| A-Mem | 0.36 | 0.26 | 0.71 |
| LangMem | 0.50 | 0.19 | 0.16 |
| Letta | 0.40 | 0.22 | 0.84 |
| MemOS | 0.38 | 0.44 | 0.84 |
| Mem0 | **0.12** | 0.19 | 0.77 |
| Memobase | 0.41 | 0.42 | 0.24 |

**三个关键发现**：

1. **Static（平均 0.44）和 Dynamic（平均 0.50）是重灾区**，Conditional 反而较好（最好的系统达 0.84）。原因是 conditional 只要保留 (条件, 值) 对即可，而 static/dynamic 需要**判定"该不该更新"**——这靠 LLM 主观判断极易出错。

2. **检索不是瓶颈，判定才是**。White-box 指标 SEH@3（正确记忆是否进入 top-3）普遍比 AA（最终答案对不对）高 0.10-0.16。说明记忆找到了，但系统在冲突时用错了。这个 gap 是本研究副贡献的核心 motivation。

3. **Mem0 在 Dynamic 上只有 0.12**，因为它的 ADD/UPDATE 靠 LLM 判断，误判率极高。

**对本项目的启示**：static/dynamic 的核心瓶颈是"判定信谁"，不是"检索找没找到"。这意味着图结构和 agent 检索（提升召回）对 static/dynamic 的帮助有限——真正需要的是**客观判定信号**。而 MLE 场景恰好有这个信号（metric + 执行条件），这是本研究方向副贡献的切入点。

---

## 3. 现有工作的定位矩阵

综合四个维度，代表性工作的定位如下：

| 工作 | 结构 | 写入 | 检索 | 冲突处理 | 场景 |
|------|------|------|------|---------|------|
| **A-MEM** (NeurIPS'25) | Zettelkasten 笔记图 | LLM 自主生成笔记 + 自动加链接 | 向量 + 链接传播 (非 agent tool) | 演化式 link-update（新笔记触发旧笔记重写） | 多轮对话 |
| **MemGPT** (ICLR'24) | 分层块 (main / recall / archival) | agent tool (function call) | agent tool (function call) | 覆盖式 (insert / replace / append) | 长对话 |
| **Mem0 / Mem0ᵍ** | NL 列表 / 三元组图 | LLM tool call: ADD/UPDATE/DELETE/NOOP | 向量 (Mem0) / 向量+一跳图遍历 (Mem0ᵍ) | LLM 决定 UPDATE 还是 ADD | 长对话 |
| **SAGE** | 异构图 | 训练过的 writer (反思反馈) | 训练过的 GFM reader | — | 多跳 QA |
| **EXG** | 经验图 (golden/warning/anchor + similar/fix/contain 边) | 框架自动写 (任务后) | 向量 seed + similar 一跳 + fix 一跳 (固定) | — | 代码 / 多跳 QA |
| **Trace2Skill** | 单一 skill 文档 (Markdown) | 多 trace LLM 并行 patch + conflict-free consolidation | 反检索（skill 直接拼进 prompt） | patch 阶段消除冲突 | 技能蒸馏 |
| **Zep / Graphiti** | 双时序知识图 (event + entity) | agent 触发 + 后端 pipeline 抽取/去重/失效 | 向量 + BFS + reranker (后端 pipeline) | 时序无效化 (invalidate, 不删) | 长对话 |
| **MemPO** | 参数 (LoRA / RL) | trajectory + memory 双 advantage RL | 无检索 (隐式存权重) | — | 长程 agent 任务 |
| **MMPO** | 参数 | turn-level Belief-Entropy credit | 无检索 | — | 长程对话 |
| **Skill-SD** | 参数 + skill 标签 | 自蒸馏 (skill-conditioned SFT) | 反检索 (skill 名直接条件) | — | 多轮 agent |
| **LongMemEval-V2** | benchmark：25 M–115 M token haystack | — (评估基准) | (基线方法用 coding-agent 翻文件) | — | web agent 轨迹 |
| **TMMA** (Workshop) | NL 列表 | 预防式写入 (写之前先检测矛盾) | 向量 | 写时拒绝/合并 | 对话 |
| **MemConflict** | benchmark | — | — | 评估 dynamic / static / conditional 三类冲突 | 对话 |
| **STALE** (arXiv 2605.06527) | benchmark | — | — | 评估 implicit 冲突（前沿模型仅 55.2%） | 对话 |

**三个空白象限**：

1. **图 + 运行时 agent 自主检索**：无人做（A-MEM 有图无 agent 检索；MemGPT 有 agent 检索无图）。
2. **基于客观信号的冲突判定**：无人做（现有全靠 LLM 主观或物理规则）。
3. **工程/任务经验记忆的冲突处理**：MemConflict 等只评估对话场景，MLE 经验的双真冲突（两次执行结果都真实但矛盾）无人研究。

---

## 4. 项目代码现状与差距分析

本项目（MLEvolve + paper-skills）已具备实现上述空白方向的基础设施。

### 4.1 已有基础

| 组件 | 代码位置 | 现状 | 与记忆研究的关系 |
|------|---------|------|----------------|
| 搜索节点 | `engine/search_node.py` | `SearchNode` 含 code/plan/metric/stage/parent/children/branch_id | 天然是实验因果树，但未图结构化 |
| 全局记忆 | `agents/memory/global_memory.py` | 平面 record 库，按 label ±1 区分成败 | 平面向量检索，无图、无 agent 检索 |
| 检索器 | `agents/memory/retriever.py` | BM25 + FAISS + RRF 融合 | 固定管道，非 agent |
| 记忆注入 | `improve_agent.py:305` → `planner_with_memory.py` | 两阶段：自由 plan → 检索相似 record 精化 JSON | 检索即注入，无采纳闭环 |
| 文献冷启动 | `engine/coldstart/methodology_agent.py` | LLM 匹配 paperinsight 类别 → 读 HIGH refs | 论文知识是静态目录，非图 |
| 经验知识库 | `paper-skills/experience_kb/` | 手工 15 条 insight（small-data-transformer-finetuning） | 手工版 Trace2Skill，未自动化 |

### 4.2 核心差距

1. **记忆是平面 record，未图结构化**：SearchNode 间的 parent-children / improves_over / fails_due_to 关系存在但未被显式建模为图边。
2. **检索是固定向量，非 agent 自主**：`planner_with_memory.py` 每步固定检索相似 record，agent 无权决定调不调、调几次、沿哪条边多跳。
3. **无冲突处理机制**：相似的成功模式重复存储，矛盾的执行结果（同一方法不同 metric）并列存在，无判定机制。
4. **无采纳闭环**：检索到的记忆是否被代码采纳、采纳后是否有效，完全无追踪。

### 4.3 MLE 场景的独特优势

MLE 经验记忆比对话记忆有四个本质差异，每一个都"逼"出现有系统没有的新机制：

| 差异 | 对话场景 | MLE 场景 | 逼出的机制 |
|------|---------|---------|-----------|
| **因果关系** | 语义相似（similar_to 够用） | 执行派生（B 改进 A，metric 变化） | 异构因果边 |
| **可验证性** | 无法验证对错 | code 可重跑、metric 是 ground truth | 客观质量信号 |
| **冲突性质** | 一真一假 | **双真**（两次执行都真实但结果矛盾） | 基于条件的判定 |
| **内容类型** | 文本事实 | 可执行代码 + metric | verifiable code memory |

**第四点尤其重要**：对话场景的 static conflict 是"真值 vs 错误矛盾"，LLM 多少能靠语义识别；但 MLE 的 static conflict 是"两次真实执行结果矛盾"，LLM 没有先验知道该信谁——**只有 metric 和执行条件能判定**。这是本研究方向不可替代的核心优势。

---

## 5. 研究方向

> **方向声明（核心范式）**：本研究**不是从 SearchNode 直接建图**，而是**先用 Trace2Skill 范式蒸馏出 skill，再把蒸馏产物组织成图**。数据流：`trace → skill → graph → agent retrieval`。图节点是 Skill 蒸馏产物，不是 SearchNode 本身；蒸馏是周期性的，与 mlevolve 在线搜索解耦。

基于上述空白分析和项目基础，提出**三层互补**的研究贡献，构成"蒸馏→组织→检索→演化"的完整流水线。

### 5.1 主贡献 1：MLE 场景下的改进 Trace2Skill 蒸馏管线

#### 与 Trace2Skill 原版的关键差异（你的 novelty 边界）

| 维度 | Trace2Skill 原版 | 本研究改进 |
|------|------------------|-----------|
| **场景** | spreadsheet / VisionQA / math 等通用任务 | **MLE 工程经验**（含 metric / code / 失败 trace）|
| **冲突处理** | Stage 3 消除合并 + prevalent bias 丢弃低频 | **条件化保留 + metric 客观判定**（不丢失低频正确经验）|
| **patch 验证** | LLM 主观判断 patch 价值 | **结合 metric delta 客观验证**（哪个 patch 真的有效）|
| **产出形态** | 单一 Markdown 文档（线性）| **结构化 patch + 适用条件标注**，为 5.2 节的图构建做准备 |

#### 流程

```
Stage 1 - Trace Collection: 从 mlevolve/runs/ 抽取 SearchNode root-to-leaf 路径
Stage 2 - Parallel Patch Proposal: success/error analyst 并行提 patch
                                  （MLE 扩展：每个 patch 带 metric_delta + applicable_conditions）
Stage 3 - Conflict-Aware Consolidation: 不消除冲突，按条件聚类保留
                                       （高频同条件 → 主体；低频但 metric 优 → 反常有效；不同条件 → 条件分支）
```

#### Novelty 验证锚点
- 你的 `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md` 是手工 ground truth（15 条）。
- 自动蒸馏的产出对比手工版本：召回率 ≥ 10/15、新发现 1-2 条、条件标注 100%。

### 5.2 主贡献 2：Skill Graph——蒸馏产物的图结构组织

#### 核心思想
把 5.1 蒸馏出的 skill 文档**进一步结构化为图**，让 skill 之间的关系（适用条件、失败规避、冲突、演化）显式可遍历，避免线性 Markdown 的隐含联系丢失。

#### 图结构设计

**节点类型（6 类）**：
- `Skill`：完整技能文档（顶层，如 `small-data-transformer-finetuning`）
- `SOP`：技能中的具体流程规则（如"先 stratified split"、"用 partial unfreezing"）
- `Condition`：适用条件（如"小数据集"、"transformer backbone"、"GPU 时间有限"）
- `FailureMode`：常见失败（如"冻结 backbone 性能差"、"全参数微调天花板"）
- `Evidence`：证据节点（含 metric delta、source trace_id）
- `Implementation`：可复用代码模板/配置/prompt 片段

**边类型（10 类）**：
- 内部组织：`contains` (Skill→SOP/FailureMode), `has_implementation` (SOP→Implementation)
- 条件适用：`applies_when` (SOP→Condition), `requires` (SOP→SOP，前置依赖)
- 失败规避：`prevents` (SOP→FailureMode), `causes` (FailureMode→SOP，反向归因)
- 证据支撑：`supported_by` (SOP→Evidence), `metric_delta` (Evidence 数值边)
- 冲突演化：`conflicts_with` (SOP↔SOP), `refines` (新版→旧版), `generalizes_from` (Skill→Trace)

#### 与现有图记忆的对比

| 维度 | A-MEM | EXG | SAGE | **本研究 Skill Graph** |
|------|-------|-----|------|----------------------|
| 节点是什么 | Zettelkasten note（对话事实）| case + task anchor（轨迹）| entity + memory fragment | **Skill 蒸馏产物**（多粒度）|
| 节点产生时机 | 流式（agent 写入）| 流式（每次交互）| 流式（writer 训练）| **周期性蒸馏后批量建**|
| 边的语义来源 | 语义相似 | 同任务 fixed_by + 相似 | 训练学到的结构角色 | **方法学关系**（适用条件/失败规避/冲突）|
| 与蒸馏的关系 | 无 | 无 | 无 | **图是蒸馏的进化形态**|

#### Novelty 边界
- vs 把 Trace2Skill 产物当 Markdown：**图能表达跨 SOP 的隐含联系**（如某个 SOP 适用条件 = 另一个 SOP 的失败模式）
- vs A-MEM/EXG/SAGE：**图节点是高层抽象（蒸馏出的 SOP），不是低层 trace**——粒度差一个层级

### 5.3 主贡献 3：Agent 自主多跳检索 Skill Graph

#### 核心思想
mlevolve 的 draft / improve / debug agent 在搜索过程中，通过 tool call **主动查询 Skill Graph**，而不是被系统固定注入。

#### 与 mlevolve 现有记忆的关系（双层架构）

```
Layer A（保留）: SearchNode 树 + GlobalMemoryLayer
                 → fetch_child_memory() 提供局部上下文（兄弟尝试）
                 → 当前 run 内的细粒度经验

Layer B（新增）: Skill Graph
                 → search_skill_graph(...) tool 提供高层方法学
                 → 跨 run / 跨任务的可迁移知识
```

#### Tool 设计

| Tool | 用途 | 检索路径 |
|------|------|---------|
| `get_sop_for_condition(task_desc, conditions)` | improve agent 找适用 SOP | Condition → APPLIES_WHEN(rev) → SOP → SUPPORTED_BY → Evidence |
| `trace_failure_to_fix(error)` | debug agent 失败追溯 | FailureMode → PREVENTS(rev) → SOP → HAS_IMPLEMENTATION → code |
| `query_with_conflict_awareness(topic)` | 看到所有相关版本（含冲突）| SOP → CONFLICTS_WITH → SOP，返回 (winner, loser, resolution) |
| `find_chain(start, edge_path, k)` | agent 自定义复杂查询路径 | 任意多跳遍历 |

#### Novelty 边界
- vs A-MEM 的固定向量检索：**agent 决定何时查、查什么 tool、几跳**
- vs EXG 的固定三通道模板：**检索路径由 agent 运行时构造，不是硬编码**
- vs MemGPT 的 archival_memory_search：**MemGPT 查的是平面块，本研究查的是结构化 Skill Graph**

### 5.4 副贡献：基于执行信号的冲突判定（贯穿 5.1 + 5.2）

#### Motivation
MemConflict 实测数据：static (0.44) / dynamic (0.50) 表现差，SEH@K vs AA gap 0.10-0.16 → **检索不是瓶颈，判定才是**。

#### 应用位置
1. **5.1 蒸馏阶段**：`ConflictAwareConsolidator` 不丢弃低频 patch，按条件聚类保留。
2. **5.2 图构建阶段**：`builder._link_conflicting_sops` 自动检测同条件矛盾 SOP，加 `conflicts_with` 边。
3. **5.4 演化阶段**：新蒸馏产物合并到现有图时，用 metric 客观判定主导版本。

#### 核心逻辑
```
矛盾候选对 (A, B)
    ↓
抽取条件（task_type / data_size / model_size / 失败模式）
    ↓
条件相同？
    ├─ 否 → 加 REFINES 边（条件分支，都保留）
    └─ 是 → metric 比较
        ├─ 显著优 → CONFLICTS_WITH + winner 标注
        └─ 接近 → 保留两者 + 加置信度警告
```

### 5.5 三层贡献的流水线关系

```
[Layer 1 蒸馏] mlevolve trace → Trace2Skill 改进版 → skill 文档
       ↓ 周期性触发，离线
[Layer 2 组织] skill 文档 → SkillGraphBuilder → Skill Graph
       ↓ 一次构建后持久化
[Layer 3 检索] Skill Graph + Layer A 局部记忆 → agent 自主多跳查询
       ↓ 每次 mlevolve 迭代
新 trace 产生 → 触发 Layer 4 演化（冲突处理 + 写回）
```

**实验预测**（与 v1 路线图一致）：
- 只做 5.1（蒸馏）+ 用 Markdown 注入 prompt（Trace2Skill 反检索式）→ 性能基线
- 加 5.2 + 5.3（图 + agent 检索）→ 检索精度（SEH@K）显著提升
- 加 5.4（冲突判定）→ 在矛盾场景下 AA 显著提升

每层贡献都可独立验证，避免"全有或全无"风险。

---

## 6. 实验框架

### 6.1 主实验（回答"好不好"）

5 条件 × 3 任务 × 3 seed，对比：

| 条件 | 结构 | 检索 | 冲突处理 |
|------|------|------|---------|
| (a) No-Mem | 无 | 无 | 无 |
| (b) Flat-Vec（mlevolve现状） | 平面 | 固定向量 | 无 |
| (c) Graph-Vec（EXG式） | 图 | 固定向量+图遍历 | 无 |
| (d) Flat-Agent（MemGPT式） | 平面 | agent检索 | 无 |
| (e) Graph-Agent（本研究完整方法） | 因果图 | agent多跳检索 | metric判定 |

指标：最终 metric、收敛速度、重复错误率。

### 6.2 消融（回答"图还是 agent 的功劳"）

在完整方法上做减法：w/o 因果边、w/o 多跳、w/o agent 自主性、w/o metric 判定。

### 6.3 冲突专项（回答"判定有没有用"）

借鉴 MemConflict 的 SEH@K vs AA 双层评估：
- 构造 static/dynamic 冲突场景（同方法不同条件/不同结果）
- 对比：LLM 主观判定 vs metric 客观判定
- 预期：metric 判定显著提升 AA，尤其是双真冲突

### 6.4 效率分析（回答"慢不慢"）

报告检索延迟（p50/p95）、token 成本、检索调用次数。论证"agent 检索虽单次贵，但减少试错步骤，端到端更快"。

### 6.5 跨任务泛化（回答"过拟合吗"）

至少 2-3 个 MLE-bench 任务（NLP / tabular / CV），验证跨任务迁移。

---

## 7. 投稿目标与可行性

### 7.1 现实目标

参考 A-MEM（NeurIPS 2025 poster）的录用水平，本研究方向的现实目标是 **NeurIPS / ICML / ICLR poster**（录用率约 25-30%）。冲刺 oral/spotlight 需要极强实验，作为上限目标。

### 7.2 可行性

- **代码基础**：MLEvolve 框架完整（搜索树、记忆层、冷启动），只需增量改造。
- **实验数据**：Spooky 任务已有 Run1-8 历史轨迹（含丰富成功/失败案例），可做离线验证。
- **算力估算**：主实验 + 消融约 700-900 GPU-h，核心的单任务验证只需约 16 GPU-h（1 周内可见信号）。

### 7.3 主要风险

1. **Novelty 被质疑增量**：需用消融证明"图 + agent 检索"的组合效应 > 各部分之和。
2. **Agent 检索延迟**：需报告效率指标，论证总成本更优。
3. **条件抽取质量**：依赖 LLM 归因，需用 metric 反向校准。

---

## 8. 结论

当前 agent 记忆研究在四个维度上高度碎片化。本综述通过横向拆解代表性工作，定位出三个空白象限：(1) 图结构 + 运行时 agent 自主检索；(2) 基于客观信号的冲突判定；(3) MLE 工程经验的双真冲突处理。

基于项目已有的 MLEvolve 搜索框架和 MLE 场景独有的客观执行信号，提出"图结构经验记忆 + agent 自主多跳检索"（主贡献）与"基于执行信号的 static/dynamic 冲突判定"（副贡献）两个互补方向。两者构成"召回→判定→采纳"流水线，每个贡献都有明确的对比对象（A-MEM/MemGPT/EXG/Trace2Skill/Mem0）和反方论证。

下一步的关键是**用最小实现验证核心 claim**（图 + agent 检索 > 固定向量），在 Spooky 任务上跑出对比结果后，再根据结果强弱决定实验扩展范围。

---

## 参考文献（按本综述出现顺序）

### 核心对标论文

1. **A-MEM**: Agentic Memory for LLM Agents. arXiv:2502.12110. NeurIPS 2025 (poster).
   - 图(Zettelkasten) + agent自主写入 + 纯向量检索。本研究 baseline #1。
2. **MemGPT**: Towards LLMs as Operating Systems. arXiv:2310.08560.
   - 分层文本块 + agent tool call 检索。agent 检索范式来源。
3. **SAGE**: Self-Evolving Agentic Graph-Memory Engine. arXiv:2605.12061.
   - 异构图 + 训练式 GFM 检索 + writer-reader 闭环。图结构上界参照。
4. **EXG**: Self-Evolving Agents with Experience Graphs. arXiv:2605.17721.
   - 经验图 + 固定向量+图遍历检索 + failure-aware rerank。最直接竞品。
5. **Trace2Skill**: Distill Trajectory-Local Lessons into Transferable Agent Skills. arXiv:2603.25158.
   - 技能蒸馏 + 反检索 + 冲突消除合并。蒸馏路线代表。
6. **Mem0**: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
   - ADD/UPDATE/DELETE tool call + 图(Mem0g) + 反直觉证据（图非万能）。

### 冲突处理与评估

7. **MemConflict**: Evaluating Long-Term Memory Systems Under Memory Conflicts. arXiv:2605.20926.
   - 三类冲突(dynamic/static/conditional) benchmark。static 0.44 / dynamic 0.50 的硬证据。
8. **STALE**: Can LLM Agents Know When Their Memories Are No Longer Valid? arXiv:2605.06527.
   - Implicit conflict 评估。Adoption gap 的独立佐证。
9. **TMMA**: Truth-Maintained Memory Agent. NeurIPS 2025 Workshop.
   - 写入时预防式质量控制。
10. **ConflictBank**: NeurIPS 2024. LLM 知识冲突 benchmark。

### 其他记忆工作

11. **MemPO**: Self-Memory Policy Optimization. arXiv:2603.00680.
    - 参数化记忆，反对 embedding 检索。
12. **Skill-SD**: Skill-Conditioned Self-Distillation. arXiv:2604.10674.
    - 技能蒸馏 + 参数内化，反检索。
13. **LongMemEval-V2**: Evaluating Long-Term Agent Memory. arXiv:2605.12493.
    - AgentRunbook-C（coding agent 检索轨迹文件）。唯一真 agent 检索经验。
14. **Zep/Graphiti**: Temporal Knowledge Graph. arXiv:2501.13956.
    - 时序 KG + 后端固定 pipeline 检索。
15. **Graph-based Agent Memory** (Survey). arXiv:2602.05665.
    - 记忆分类法 + 生命周期四阶段理论框架。
16. **InternAgent-1.5**: Unified Agentic Framework for Long-Horizon Scientific Discovery. arXiv:2602.08990.
    - SPM/TEM/SKM 分层记忆 + Generation-Verification-Evolution。
17. **Meta-Cognitive Memory (MMPO)**: arXiv:2605.30159. Belief Entropy 优化摘要质量。
18. **MemoryArena**: arXiv:2602.16313. 多会话记忆 benchmark。
19. **Evidence Tracing Survey**: arXiv:2606.04990. memory lineage 概念。

### 项目内部参考

- `mlevolve/agents/memory/global_memory.py` / `retriever.py`：当前平面记忆与混合检索。
- `mlevolve/engine/search_node.py` / `agent_search.py`：搜索树与多阶段调度。
- `mlevolve/agents/planner/planner_with_memory.py`：记忆注入现状。
- `mlevolve/engine/coldstart/methodology_agent.py`：文献冷启动。
- `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md`：手工版 Trace2Skill 产物（15 条 insight）。
