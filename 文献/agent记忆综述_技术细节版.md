# Agent 长期记忆系统综述：分环节技术实现与研究方向

> **定位**：面向博士组会汇报的技术细节型文献综述。不停留在"谁做了什么"，而是逐一拆解代表性工作在 **记忆结构 / 写入 / 检索 / 冲突处理** 四个环节上**具体怎么实现**（节点 schema、建边规则、相似度公式、检索跳数、reward 设计、冲突仲裁逻辑），再据此定位文献空白、推导本人下一步研究方向（基于 MLEvolve 的 Skill Graph）。
>
> **组织方式**：主体按四环节 + 评估协议 **横向拆解**（便于对比、找空白）；文末附每个系统的**纵向实现速查卡**（便于理解单个工作的完整设计）。
>
> **覆盖范围（全面但分主次）**：
> - **深讲**（与本人方向直接相关）：图结构记忆（A-MEM / EXG / SAGE / Mem0ᵍ / Zep）、技能蒸馏（Trace2Skill / Skill-SD）、冲突评估（MemConflict / STALE）。
> - **略讲**：参数化记忆（MemPO / MMPO）、长程 benchmark（LongMemEval-V2 / MemoryArena）、分层记忆系统（InternAgent-1.5）、两篇 survey 的分类法。
>
> **准确性说明**：技术细节来自对本地 PDF 与 MemConflict 代码仓库的精读。公式编号、超参、实测数字以各论文"原文报告值"为准，正式引用前建议回原文二次核对（已在每节标注来源）。

---

## 0. 一页纸速览：四环节 × 代表系统

| 系统 | ① 结构 | ② 写入 | ③ 检索 | ④ 冲突处理 |
|------|--------|--------|--------|-----------|
| **A-MEM** | Zettelkasten 笔记图（同质，similar_to 边） | LLM 自主构造 note + 自动建链 + 演化更新 | **纯向量 top-k**（§3.4 三公式，无 LLM） | 演化式 link-update（新笔记触发旧笔记重写） |
| **MemGPT** | 分层文本块（main/recall/archival，非图） | agent function call（append/replace） | **agent function call**（archival_memory_search） | 覆盖式（insert/replace） |
| **EXG** | 经验图（异构：case + task anchor；similar/contain/fixed_by 边） | 固定管道（规则建边） | 固定三通道并行 + 1 跳 + rerank | 无显式（覆盖式插入） |
| **SAGE** | 异构图（entity + 关系三元组，Neo4j） | **训练式 writer**（GRPO 策略网络） | **训练式 GFM reader**（软寻址 + 结构门控传播） | LLM Conflict Detector + 时序失效 |
| **Mem0 / Mem0ᵍ** | NL 列表 / entity 三元组图（Neo4j） | LLM tool call：ADD/UPDATE/DELETE/NOOP | 向量（Mem0）/ 向量 + 全邻域遍历（Mem0ᵍ） | LLM 判定 UPDATE vs DELETE |
| **Zep / Graphiti** | 双时序知识图（entity + event，valid_at/invalid_at） | agent 触发 + 后端 pipeline 抽取/去重/失效 | 向量 + BFS + reranker（后端 pipeline） | **时序失效**（invalidate 不删，客观仅限时间维度） |
| **Trace2Skill** | 单一 Markdown 技能文档（**无图**） | 多 trace 并行 patch + 层级合并消除冲突 | **反检索**（skill 直接拼进 prompt） | patch 合并阶段消除 + prevalent bias 丢低频 |
| **Skill-SD** | 自然语言技能 + 内化进权重 | 辅助 LLM 总结 + 自蒸馏 SFT（reverse-KL） | 训练内 UCB buffer；**inference 无检索** | — |
| **MemPO / MMPO** | 参数（模型内 `<mem>` action） | RL（双 advantage / Belief-Entropy credit） | **无检索**（隐式存权重） | — |
| **InternAgent-1.5** | 三层（SPM 策略 / TEM 任务情节 / SKM 语义图） | 分层 consolidate + 写回 | 分层检索（KG 遍历 + episode 召回） | — |

**一句话结论**：没有任何系统同时做到 **图结构 + 运行时 agent 自主检索 + 基于客观信号的冲突判定**。"节点是 procedural skill（程序性技能）"的图记忆是文献空白。

---

## 1. 引言：四环节分析框架

### 1.1 为什么按"环节"拆

记忆系统的能力可分解为一条流水线（与 *Graph-based Agent Memory Survey*, arXiv:2602.05665 的生命周期四阶段 Extraction → Storage → Retrieval → Evolution 一致）：

```
原始 trace ──①结构化──> 记忆单元 ──②写入──> 组织好的存储
                                              │
              查询 ──────③检索──────> 相关子集/子图
                                              │
              矛盾出现 ──────④冲突处理──────> 仲裁后的记忆
```

按系统纵向讲会重复且难比较；**按环节横向拆**才能暴露"谁在哪一环做得深、哪一环是集体短板"。本综述据此把每个系统在四环节上的实现拆开对照。

### 1.2 问题本质（MemConflict 的定义）

> **Memory validity is a "query-conditioned fitness-for-use" problem.**

记忆有效性 = "在当前查询下能否提供正确、有效、适用的信息"。这把记忆从**存储问题**抬成**推理问题**：不是"存了就行"，而是"找到若干相关甚至矛盾的经验后，怎么组织、怎么判定、怎么多跳追溯"。普通 RAG 只解决"找得到"，本综述关心的是其后的组织、判定、追溯。

---

## 2. 环节一：记忆结构（怎么组织）

记忆的物理组织形式决定了能表达什么关系。三类范式：平面记录 / 分层块 / 图。

### 2.1 平面记录与分层块

- **平面向量库（Flat）**：每条记忆是独立记录，靠向量相似度检索（早期 RAG、ReasoningBank、本项目 mlevolve 的 `GlobalMemoryLayer`）。无法表达"记忆 A 是 B 的改进版"这类关系。
- **分层文本块（Tiered）**：**MemGPT / Letta**（arXiv:2310.08560）借鉴 OS 内存层级，分三层——`core/main memory`（始终在上下文内的 persona/human 块）→ `recall memory`（对话历史）→ `archival memory`（外部向量库）。解决了上下文窗口有限的问题，agent 可在层间搬运信息，但**层内仍是文本块，不是图**，多跳推理弱。

### 2.2 图结构：同质 vs 异构

把记忆组织成节点 + 边，节点是记忆单元，边编码关系。**图的强弱取决于边类型的信息量**。

| 系统 | 节点（粒度 + schema） | 边类型 | 同质/异构 | 存储后端 |
|------|---------------------|--------|----------|---------|
| **A-MEM** | Zettelkasten 笔记 (K 关键词, V 内容, T 标签, C 上下文) | `similar_to`（语义链接） | 同质 | 向量库 + 链接表 |
| **EXG** | **Case 节点**：`c_τ^(k) = (τ, x_τ, y_τ^(k), r_τ^(k), σ_τ^(k))`（任务 id / 输入 / 第 k 次输出 / 成功标志 r∈{0,1} / 执行信号摘要）+ **Task Anchor 节点** | `contain`(anchor→case) / `similar`(无向带权) / `fixed_by`(case→case 纠错) | 异构 | 图结构 + FAISS |
| **SAGE** | **Entity 节点**（类型 + 嵌入 x_e + 时间戳 t_e）+ **关系三元组** (u, r, v) | 关系边（typed, directed）+ alias + 时序约束 | 异构 | Neo4j |
| **Mem0ᵍ** | Entity 节点（类型 + 嵌入 + 时间戳）+ 关系三元组 (v_s, r, v_d) | explicit / implicit / temporal 关系 | 异构 | Neo4j |
| **Zep** | Entity + Event 双类型（带 `valid_at` / `invalid_at` 时间戳） | 时序关系边 | 异构 + 时序 | 时序 KG |
| **本研究 Skill Graph** | **Skill / SOP / Condition / FailureMode / Evidence / Implementation**（6 类，方法学多粒度） | applies_when / prevents / refines / conflicts_with 等（方法学关系） | 异构 | NetworkX |

**关键观察**：
- A-MEM 的 `similar_to` 是**最弱**的边（纯语义相似）；EXG 的 `fixed_by`（错误→修复）已有**因果意味**；SAGE 的结构角色最强但依赖训练。
- 所有图系统的节点都是**事实 / 事件 / 实体 / case（声明性或情节性知识）**。EXG 的 case 是"轨迹快照"（episodic），不是可复用的程序性技能。**没有一个系统的节点是蒸馏出的 procedural skill（SOP 级方法学）**——这是本研究的结构层 novelty。

### 2.3 *Graph Memory Survey* 的认知分类法（可作框架支撑）

该 survey 把记忆按认知结构分五层：**Semantic**（一般化知识）/ **Procedural**（怎么做）/ **Associative**（潜在关联）/ **Episodic**（发生了什么）/ **Sentiment**（情感）。又按功能分 **Knowledge Memory**（静态、被动、预加载）vs **Experience Memory**（动态、主动、来自执行轨迹）。

> **对本研究的支撑**：现有图记忆几乎全落在 Semantic/Episodic + Knowledge 象限；**Procedural + Experience 的图**几乎无人做。MLE 的执行轨迹正是 Experience Memory，蒸馏出的 SOP 正是 Procedural Memory。

---

## 3. 环节二：写入（谁决定记什么、怎么连、何时更新）

三种写入策略：固定管道 / agent 自主 / 训练式。

### 3.1 固定管道写入

系统按预设规则提取记忆，agent 不参与决策。

- **EXG**（完全确定性规则）：每完成一次任务尝试自动触发。流程 = 轨迹 → case 抽象 → 图更新。建边规则：①新 case 连 `contain` 到 task anchor；②若当前成功且前序失败，加 `fixed_by` 边；③相似边由公式 (9) 计算后建：
  $$ s(c_i, c_j) = \alpha\langle e_p(c_i), e_p(c_j)\rangle + (1-\alpha)\,h(c_i)h(c_j)\langle e_f(c_i), e_f(c_j)\rangle,\quad \alpha=0.8 $$
  其中 $e_p$ 是 prompt 内容嵌入（MiniLM），$e_f$ 是失败文本嵌入，$h\in\{0,1\}$ 指示是否含失败信息。去重在 Algorithm 1 的 Deduplicate 步。**无 LLM 决策**。
- **本项目 mlevolve 现状**：每个 `SearchNode` 解析完自动入 `GlobalMemoryLayer`，按 `_determine_label` 计算 label∈{-1,0,1}（成功/中立/失败）。固定、快，但写入质量不可控。

### 3.2 Agent 自主写入

LLM 通过 tool call 决定记什么、怎么连、何时更新。

- **MemGPT**：显式 function call —— `core_memory_append` / `core_memory_replace`（写 core）、`archival_memory_insert`（写 archival）。agent 自行决定何时调用。
- **Mem0 / Mem0ᵍ**：提取阶段 LLM 抽候选记忆 $\Omega=\{w_1,...,w_n\}$；更新阶段先向量检索 top-s=10 相似记忆，再由 LLM 输出 **tool call：ADD / UPDATE / DELETE / NOOP**。Mem0ᵍ 额外用 Entity Extractor + Relation Generator 抽三元组，按实体嵌入相似度 ≥ 阈值 Γ 合并去重。
- **A-MEM**：LLM 在"记忆管理阶段"自主构造 note（生成 keywords/tags/links），并判断是否建链、是否触发旧 note 演化更新。**A-MEM 在 §2.2 明确自白**：它的 agency 在"存储和演化"，而非"检索"——这句话划清了写入 agent 与检索 agent 的边界，对本研究定位至关重要。

### 3.3 训练式写入

用 RL 训练一个 writer 策略，按下游反馈优化写入。

- **SAGE**（最完整的训练式 writer）：状态 $s_t=(q, D, G_{t-1}, D^{proc}_{t-1})$，策略网络 $\pi_\theta$ 输出动作 $a_t$ = 实体-关系三元组 + source anchor，图更新 $G_{t+1}=G_t \oplus a_t$。用 **GRPO** 训练，多目标奖励：
  $$ r_{task}=\frac{\alpha\, r_{rec}+\beta\, r_{pre}+\gamma\, r_{ded}}{\alpha+\beta+\gamma} $$
  其中 $r_{rec}$=证据覆盖率、$r_{pre}$=精度（惩罚无关扩展）、$r_{ded}$=充分性（Judge 是否能据此答对）。另加防重复惩罚 $\rho_{rep}(G)$。**Writer-Reader 闭环交替训练**：先固定 reader 训 writer，再用新 writer 生成图训 reader（理论上有界图漂移保证 document score 变化有界，Prop 1.iii）。
- **Trace2Skill**（蒸馏式写入，本研究起点，详见 §3.4）。

### 3.4 重点：Trace2Skill 的蒸馏写入管线（本研究改进对象）

来源：arXiv:2603.25158v4。这是"从大量 trace 自动蒸馏出可迁移 skill"的代表，本研究在其基础上改造。三阶段：

**Stage 1 — 轨迹生成**：ReAct agent $\pi_\theta$ 在冻结初始技能 $S_0$ 下并行跑任务集，每条轨迹 $\tau_i=\{q_i,(r_k,a_k,o_k)_{k=1}^{T_i}, y_i\}$（推理/动作/观察 + 正确性标签 $y_i\in\{0,1\}$），按标签分 $T^-$（失败）/ $T^+$（成功）。

**Stage 2 — 并行多 analyst 提 patch**（不对称设计）：
- **成功分析器 $A^+$**（单次前向）：输入 $S_0$ 副本 + 一条成功轨迹 → 识别成功行为 → 提 patch $p_i$。
- **错误分析器 $A^-$**（多轮 ReAct 环）：可读输入输出文件、对比 agent 答案 vs 真值、迭代逼近根因；终止于"成功修复 + 因果解释"或耗尽预算（未验证则丢弃）。
- **独立性保证**：所有 analyst 在冻结 $S_0$ 上跑，互不可见 → 保留多样性。

**Stage 3 — 无冲突合并（层级合并算子 M）**：
$$ L=\lceil \log_{B_{merge}} |P|\rceil,\quad p^{(\ell+1)}=M(\pi_\theta, S_0, \{p_1^{(\ell)},...,p_{B_{merge}}^{(\ell)}\}) $$
M 做三件事：①去重；②**冲突消除**（三道编程栏杆：引用不存在文件→拒绝、同文件同行范围冲突→标记、格式校验）；③**模式挖掘（prevalent pattern bias）**——在多条独立 patch 中反复出现的修改更可能是系统性属性而保留，**低频修改判为 idiosyncratic 丢弃**。最终产出演化技能 $S^*=(M^*, R^*)$ = `SKILL.md`（Markdown 根文档）+ scripts/references/assets。

**注入方式 = 反检索**：inference 时技能预加载进 system prompt，无检索延迟、无 embedding 依赖。

**Trace2Skill 自承认的两个关键局限（§6，本研究切入点）**：
1. **丢弃低频正确经验**：prevalent bias 把低频但有效的 patch 当 idiosyncratic 过滤掉。
2. **仅物理冲突检测**：三道栏杆都是 syntactic/structural 检查（文件存在性、行号、格式），**未检测语义冲突**（两条 patch 对同一流程给出矛盾逻辑建议）。
3. （附）**无图**：skill 写成线性 Markdown，丢失 SOP 间隐含联系（如某 SOP 的适用条件 = 另一 SOP 的失败模式）。

> **本研究对 Trace2Skill 的改进**：① 用 metric_delta 客观验证 patch（不靠 prevalent 频次丢低频）；② 产出结构化 patch（带 applies_when / prevents / metric_delta / source_trace_ids）而非 Markdown，为建图准备；③ 把矛盾 patch 保留进图，由 `conflicts_with` / `refines` 边表达，而非合并阶段消除。

### 3.5 写入策略对比

| 策略 | 写入质量 | 自适应 | 训练成本 | 代表 |
|------|---------|--------|---------|------|
| 固定管道 | 低 | 无 | 无 | EXG, mlevolve 现状 |
| Agent 自主 | 中 | 高 | 无 | MemGPT, Mem0, A-MEM |
| 训练式 | 高 | 中 | 高 | SAGE, Trace2Skill（蒸馏式） |

> **对本项目的启示**：MLE 场景天然有 writer——搜索循环本身就是序列写入（每个 SearchNode 一步），不需额外训练 writer；可借鉴 A-MEM 的自主建边 + SAGE 的 reward 闭环思想，但蒸馏走 Trace2Skill 改进路线。

---

## 4. 环节三：检索（由谁触发、如何构造查询、几跳）

这是本研究最核心的维度。三类：固定向量 / agent 自主 tool / 训练式 reader。

### 4.1 固定向量检索（非 agent）

系统按固定管道触发，向量相似度返回 top-k。

- **A-MEM**：尽管叫 "Agentic Memory"，**检索是纯向量**——§3.4 三个公式（all-MiniLM 编码 + 余弦相似度 + top-10），**无 LLM 参与**。它在 §2.2 承认检索 agency 让给了 agentic RAG。
- **Mem0**：$Sim(q,m)=\cos(\mathrm{Emb}(q),\mathrm{Emb}(m))$，文本嵌入小模型，top-k (k∈{1,2})。
- **Mem0ᵍ**：双通道——①entity-centric（识别查询关键实体作锚点）；②语义三元组 $Sim_{triple}(q,(v_s,r,v_d))=\cos(\mathrm{Emb}(q),\mathrm{Emb}(v_s,r,v_d))$。再从锚点实体**遍历全部 incoming/outgoing 关系**构子图。无明确跳数限制但本质是一跳邻域扩展。

### 4.2 EXG：固定三通道 + 单跳 + rerank（最接近"图遍历"的固定管道）

来源：arXiv:2605.17721，Algorithm 1。新任务构造临时 case $c_q$，三通道并行生成候选：
1. **任务锚点通道**：$C_{task}=\{c \mid (a_\tau^{(q)} \to c)\in E_{contain}\}$（公式 4）。
2. **语义种子 + 单跳桥接**：FAISS 取 top-$K_s$=10 种子，沿 `similar` 边单跳扩展 $C_{sim}=\{c' \mid \exists c\in S,(c-c')\in E_{sim}\}$（公式 6），fanout $F_{sim}=F_{bridge}=5$。
3. **纠错迹**：$C_{fix}=\{c' \mid \exists c\in C_{sim},(c\to c')\in E_{fix}\}$（公式 7）。

最终池 $C=\mathrm{Cap}(\mathrm{Dedup}(C_{task}\cup C_{sim}\cup C_{fix}))$，全局上界 $K_c=30$。**Rerank** 用单跳相关性传播：
$$ \rho(c)=\max\{\rho^0(c),\ \max_{u\in S}[\rho^0(u)+w(u,c)]\}\quad (\text{公式 }10) $$
取 top-$H$=5，组装成 golden/warning/fixed-by 三类 hint 注入 system prompt。

**关键局限（Q3 的天花板）**：路径长度永远 ≤ 2（similar 1 步 + fix 1 步），同种边不允许连走 ≥ 2 步，检索阶段零 LLM 调用——**信息不够也不能再走**。这正是"agent 自主多跳"要突破的点。

### 4.3 Agent 自主检索（运行时 tool call）

LLM 在运行时通过 tool call 自主决定何时检索、检索什么、几跳。

- **MemGPT**：`archival_memory_search` tool，agent 自己决定调不调、查什么。但**记忆是分层块不是图**。
- **LongMemEval-V2 的 AgentRunbook-C**：coding agent 在沙箱里把轨迹存成文件，运行时写代码检索/拼接证据（report 72.5%，比 prompt-based QA 高一档）。这是文献里**唯一接近"真 agent 检索"的经验**，但**记忆是文件不是图**。

> **空白**：图 + 运行时 agent 自主检索的组合无人做（A-MEM 有图无 agent 检索；MemGPT/AgentRunbook 有 agent 检索无图）。

### 4.4 训练式 reader（SAGE，图检索上界参照）

来源：arXiv:2605.12061。Reader 是为多图预训练的 **Graph Foundation Model (GFM)**，离线 contrastive + supervised 训练，inference 冻结前向（非运行时 agent）。两个认知启发机制：

- **软寻址**（公式 1，模拟注意分配）：多维刺激强度
  $$ s_e(q)=\lambda_1\mathrm{Exact}+\lambda_2\mathrm{Alias}+\lambda_3\max_m\cos(\mathrm{Emb}(desc(e)),\mathrm{Emb}(\tilde q_m))+\lambda_4\mathrm{Type}+\lambda_5\mathrm{Cons}+\lambda_6\sum\mathrm{EL} $$
  再 softmax 成注意分布 $p^0(e|q)$。
- **结构条件传播**（公式 2-4，模拟突触门控）：节点结构特征 $\phi(v)=[\log(1+d_v),c_v,\kappa_v,\bar d_{N(v)}]$、边对特征 $\psi(u,v)$，经向量门控 $g_{uv}^{(l)}$ 做 **hub 抑制 / bridge 保留 / 重复 habituation**，L 层 GNN 完整多跳传播。最后实体相关性投影到文档级。

**效果（report）**：NQ 零样本 Recall@2/5 = 82.5/91.6（vs HippoRAG2 45.6/78.0）；检索延迟 0.03s（vs GraphRAG 2.76s）。**优势**：能识别图结构角色（bridge/hub）。**局限**：推理时是固定前向，无法按当前失败的具体特征动态调整检索策略。

### 4.5 检索机制的空白象限

```
                        记忆结构
              平面 / 分层            图结构
          ┌──────────────────┬────────────────────┐
   固定    │ mlevolve 现状     │ A-MEM(纯向量)       │
   检索    │ Mem0, RAG        │ EXG(固定三通道+1跳)  │
          ├──────────────────┼────────────────────┤
  agent   │ MemGPT           │   ★ 空白 ★          │
  自主     │ AgentRunbook-C   │ (图 + agent 多跳)    │
          └──────────────────┴────────────────────┘
                                 ↑ SAGE 在此象限"上方"
                              （训练式 reader，非运行时 agent）
```

---

## 5. 环节四：冲突处理（矛盾记忆怎么办）

四种路线：写入时预防 / 更新时覆盖 / 消除合并 / 检索时按条件保留。

### 5.1 四种路线

| 路线 | 思想 | 代表 | 局限 |
|------|------|------|------|
| **A. 写入预防** | 写之前先检测矛盾，阻止错误进入 | TMMA（NeurIPS'25 Workshop） | 会把"条件化正确的矛盾"也挡掉 |
| **B. 更新覆盖** | 新来时 LLM 决定 UPDATE/DELETE | Mem0, MemGPT, Letta | 强制二选一，丢失条件 nuance |
| **C. 消除合并** | 批量合并成一条无冲突记忆 | Trace2Skill（层级合并 + prevalent bias） | 丢弃低频正确；只做物理冲突检测，漏语义冲突 |
| **D. 时序失效** | 不删，标 `invalid_at` | Zep/Graphiti | **客观但仅限时间维度** |
| **E. 检索时按条件选** | 都保留，检索时按条件选 | **（无人做）** | 需条件建模，现有系统不建模 |

- **SAGE / Mem0ᵍ**：LLM-based Conflict Detector 判定新旧关系是否矛盾 → ADD/UPDATE/DELETE/NOOP 或标过时（时序失效）。本质是 LLM 主观判定。
- **Zep**：双时序，`valid_at`/`invalid_at` 客观仲裁——但只覆盖"时间维度"（谁更新），不解决"两个同时有效但矛盾"。

### 5.2 冲突评估的硬证据（MemConflict / STALE）

这是"检索不是瓶颈、判定才是"的实证支柱。

**MemConflict**（arXiv:2605.20926，本地有代码）三类冲突：

| 类型 | 有效性维度 | 例 | 正确处理 |
|------|-----------|-----|---------|
| **Dynamic** | 时间有效性 | 用户搬家了 | 覆盖旧值 |
| **Static** | 事实正确性 | 出生地被误说错 | 坚持旧值 |
| **Conditional** | 上下文适用性 | 早咖啡/晚牛奶 | 按条件选 |

**指标体系**（黑盒 + 白盒 + 诊断，代码 `Evaluation/` 实现）：
- **AA**（黑盒）：最终答案是否匹配 gold。
- **SEH@K**（白盒）：top-K 检索是否含 gold memory item。
- **SRS**：gold item 的排名分。
- **UOCS**（dynamic 诊断）：是否识别更新顺序。
- **CRS**（static 诊断）：是否识别矛盾候选。
- **EUG**（可靠性诊断）：检索到 gold 却没用于答案的比例。

**三个关键发现**：
1. **static (≈0.44) / dynamic (≈0.50) 是重灾区**，conditional 反而好（最高 0.84）——因为 conditional 只要保留 (条件,值) 对，而 static/dynamic 需判定"该不该更新"，靠 LLM 主观极易错。
2. **SEH@K 比 AA 高 0.10–0.16**——记忆找到了，冲突时用错了。**检索不是瓶颈，判定才是**。
3. Mem0 在 dynamic 上低至 0.12，因 ADD/UPDATE 靠 LLM 判断误判率高。

**数据构造**（代码 `Code/` + `Prompt/`）：Step1 profile（fixed facts + timeline 初值）→ Step2 timeline 状态转移 → Step3 三类冲突 + 同 field 不同 person 的 distractor → Step4 包装多轮对话 + 自然化 rewrite（Prompt4_4 去掉 "Point_A/B" 等泄露标记）。

**STALE**（arXiv:2605.06527）补充**隐式冲突**：新观察使旧信念失效但无显式否定（Type I 共指冲突 / Type II 级联失效）。三维 probe：SR（识别过时）/ PR（拒绝含过时前提的 query）/ IPA（主动在下游应用新状态）。关键发现：**Recognition ≠ Application**——Type I-SR 可达 76% 但 IPA 仅 39%；前沿模型整体仅 55.2%。诊断出同一个 **"current-state adjudication gap"**：新证据检索率 77.5%，但最终答对仅 23%。

> **对本研究的启示**：static/dynamic 的瓶颈是"判定信谁"，图/agent 检索（提升召回）帮助有限——真正需要的是**客观判定信号**。MLE 场景恰好有（metric + 执行条件），且 MLE 的 static 冲突是"两次真实执行结果矛盾"（**双真冲突**），对话场景不存在，LLM 没有先验知道信谁——只有 metric 能判。

---

## 6. 评估协议横向对比（怎么衡量）

| Benchmark | 评估能力 | 黑盒指标 | 白盒/诊断指标 | 关键发现 |
|-----------|---------|---------|--------------|---------|
| **MemConflict** | 三类冲突下的判定 | AA | SEH@K, SRS, UOCS, CRS, EUG | 判定是瓶颈（SEH@K > AA） |
| **STALE** | 隐式冲突的状态更新 | AA（SR/PR/IPA 三维） | attention 分析 | Recognition ≠ Application；前沿 55.2% |
| **LongMemEval-V2** | web agent 长期经验（5 维：静态/动态/工作流/陷阱/前提） | AA | latency-精度 Pareto | 25M–115M token haystack；只有 coding-agent 翻文件式 (72.5%) 能 scale |
| **MemoryArena** | 多会话相互依赖任务 | SR, PS, sPS, SR@k | 依赖深度衰减曲线 | 所有方法随依赖深度明显衰减；外部记忆未必 > 长上下文 |

**黑盒 vs 白盒的价值**：单看 AA 无法区分"检索失败"还是"利用失败"。MemConflict 的设计最完整（黑盒 AA + 白盒 SEH/SRS + 诊断 UOCS/CRS/EUG），能把失败分解为检索 vs 利用——这正是支撑"判定才是瓶颈"的方法论。

> **对本研究实验设计的借鉴**：① 借 SEH@K vs AA 的双层评估，做冲突专项实验（对比 LLM 主观判定 vs metric 客观判定）；② 借 MemoryArena 的 SR@k 衰减曲线看跨依赖深度的鲁棒性。

---

## 7. 略讲分支：参数化记忆与分层系统

### 7.1 参数化记忆（记忆内化进权重，反检索）

- **MemPO**（arXiv:2603.00680）：记忆是模型自己生成的 `<mem>` 块，inference 时只用前一步 memory 作 context（丢弃更早历史）。训练用**双层 advantage**：轨迹级 $A^T$（GRPO）+ **记忆级 $A^M$**——以"给定该 memory 时正确答案的条件概率减去不含 memory 的 bias" $R^M=P[a_{ans}|\tau(s_t^{mem})]-\epsilon$ 衡量 memory 质量。长 horizon 收益更大（10-objective F1 +13.48pp，token -67.58%）。
- **MMPO**（arXiv:2605.30159）：把长程任务建成 POMDP，summary 诱导信念 $b_t^M(s)=P(s_t|m_t)$，目标 $\min H(s_t|m_t)$。用 **Belief Entropy** $H^{BE}(m_t)=H(y|m_t,q)$（对 anchor question "当前进度 + 还缺什么信息" 的回答熵）作代理信号，token 级熵估计，sub-trajectory dense reward $R_k=\alpha\sigma(-H^{BE})+r_{final}$。发现 entropy 下降与 accuracy 强相关 (r=-0.68)，可零训练做 Best-of-N 选择。
- **Skill-SD**（arXiv:2604.10674）：技能蒸馏进权重。学生（plain prompt）+ 教师（skill-augmented，参数从学生 checkpoint 动态 sync）；loss = GRPO + λ·自蒸馏（重加权 reverse-KL，λ=0.001）。training 内用 UCB buffer 选 skill，**inference 无检索**。关键：off-policy rollout 会中期崩溃，动态同步必要。

> **对本研究的意义**：参数化路线反对 embedding 检索（记忆进权重）。本研究走"外部图 + 检索"路线，但参数化是有力的对照/baseline，且 Skill-SD/Trace2Skill 都说明"skill 是有效的记忆载体"——支撑本研究"节点是 skill"的合理性。

### 7.2 分层记忆系统（InternAgent-1.5）

来源：arXiv:2602.08990。Generation–Verification–Evolution 循环 + 三层记忆：
- **SPM (Strategy-Procedural Memory)**：高层科研策略 / workflow pattern（指导 Generation 的 hypothesis formulation）。
- **TEM (Task-Episodic Memory)**：按 task 编排的时序 episode（输入参数 / 中间结果 / 失败模式），驱动 Evolution 的下一轮 refine。
- **SKM (Semantic-Knowledge Memory)**：跨学科异构知识图（typed edges：cites / by_product / prerequisite）。

> 这是"分层 + 图"的工程化样板，SPM 的"procedural"命名与本研究方向呼应，但它的 procedural 是高层策略文本，非可遍历的 SOP 图。

---

## 8. 系统实现速查卡（纵向）

> 每张卡 = 一个系统从结构到冲突的完整设计，用大白话讲清"它实际在做什么"。来源已回 PDF 核对的标 ✅原文；本地无 PDF 的标 ⚠️据原论文。

### 8.1 EXG（arXiv:2605.17721）✅原文

**一句话**：把 agent 每次解题的"完整尝试"压缩成一个 case 节点存进图，下次遇到相似任务时，沿"相似边"和"纠错边"把历史经验（含成功正例和失败教训）捞出来塞进 prompt。

- **结构**：异构经验图。两类节点——**case 节点** $c_\tau^{(k)}=(\tau, x_\tau, y_\tau^{(k)}, r_\tau^{(k)}, \sigma_\tau^{(k)})$（任务 id、输入、第 k 次输出、成功标志 $r\in\{0,1\}$、执行信号摘要如错误消息），成功的叫 golden case、失败的叫 warning case；**task anchor 节点**每任务一个，把同任务的 case 归到一起。三种边：`contain`（anchor→case，归属）、`similar_to`（无向带权，语义相似）、`fixed_by`（case→case，"这次修复了上次的错"）。
- **写入**：**固定规则、无 LLM 决策**。每完成一次尝试就抽成 case 入图；相似边由加权公式 (9) 自动建——$s(c_i,c_j)=\alpha\langle e_p,e_p\rangle+(1-\alpha)h\cdot h\langle e_f,e_f\rangle$，$\alpha=0.8$，意思是"主要看 prompt 内容相似度，再叠加失败文本相似度"。
- **检索**：固定三通道并行捞候选——①任务锚点通道（同任务的所有 case）；②语义通道（FAISS 取 top-$K_s$=10 相似种子，再沿 `similar_to` **单跳**扩展，扇出 $F_{sim}=5$）；③纠错通道（从上面结果沿 `fixed_by` 走一跳找修复案例）。合并去重后封顶 $K_c=30$，用相关性传播 (公式 10) rerank，最后输出 $H=5$ 条 hint（golden/warning/fixed-by 三类）注入 system prompt。**关键限制**：路径长度永远 ≤2 跳，同种边不能连走，检索全程零 LLM 调用——信息不够也不能再深挖。
- **冲突**：无显式处理，新 case 直接插入、旧 case 保留。
- **实验**：HumanEval / EvalPlus / MuSiQue / HotpotQA 等，Qwen3 系列。report pass@1 相对 Reflexion 提升 >150%，同时 **LLM 调用减少 45.7%、延迟降 30.5%**（因为检索是纯图遍历不烧 LLM）。
- **与本研究差异**：它节点是 **case（一次尝试的快照，episodic）**，我是蒸馏出的 **SOP（方法规则，procedural）**，抽象层级高一级；它检索固定 ≤2 跳、不能 agent 自主多跳，我让 agent 运行时决定走几跳、沿哪条边。

### 8.2 SAGE（arXiv:2605.12061）✅原文

**一句话**：把对话/文档抽成"实体关系图"，但写入和检索都不靠固定规则，而是**训练出两个模型**——一个 writer 学会"该往图里写什么"，一个 reader（图基础模型）学会"该从图里捞哪个子图"，靠 writer-reader 互相给奖励来共同进化。

- **结构**：异构实体图，Neo4j。节点是 entity（带类型、嵌入 $x_e$、时间戳 $t_e$），边是关系三元组 $(u,r,v)$，还建模 alias（别名，同一实体不同叫法）和时序约束。
- **写入（训练式 writer）**：writer 是一个策略网络 $\pi_\theta$，状态含当前查询和已有图，动作是"抽哪些实体关系三元组写进图"，用 **GRPO** 训练。奖励是多目标加权——$r_{rec}$（写的图能否覆盖答题需要的证据）、$r_{pre}$（精度，别写一堆无关的）、$r_{ded}$（充分性，据此能否答对），外加一个防重复惩罚 $\rho_{rep}$。
- **检索（训练式 GFM reader）**：reader 是为多图预训练的图基础模型，**推理时一次前向**。两个认知启发机制——**软寻址**（公式1：综合精确匹配、别名匹配、多个语义探针的最大余弦相似度等给每个实体打激活分，再 softmax 成注意力分布）+ **结构门控传播**（公式2-4：L 层 GNN，根据节点是 hub 还是 bridge 动态调整信息流，做到"抑制枢纽噪声、保留桥接节点"）。
- **冲突**：LLM Conflict Detector 判新旧关系是否矛盾 + 时序失效标记。
- **实验**：NQ 零样本 Recall@2/5 = **82.5/91.6**（远超 HippoRAG2 的 45.6/78.0）；检索延迟 **0.03s**（vs GraphRAG 2.76s）。
- **与本研究差异**：它检索是**训练好的模型固定前向**，推理时不能临时改路径；我是 agent 每跳看结果再决定下一步。且它节点是 entity（事实），我是 SOP（方法）。

### 8.3 Mem0 / Mem0ᵍ（arXiv:2504.19413）✅原文

**一句话**：面向生产环境的对话记忆。基础版 Mem0 把记忆存成自然语言句子用向量检索；图版 Mem0ᵍ 升级成"实体-关系图"。它最有价值的结论反而是**反面教材**——加了图收益很小却慢很多。

- **结构**：Mem0 是 NL 句子列表（向量库）；Mem0ᵍ 是**有向标记图** $G=(V,E,L)$——$V$ 实体节点（每个含类型、嵌入 $e_v$、时间戳 $t_v$ 三组件）、$E$ 关系边、$L$ 给节点赋语义类型（如 "Alice→Person"）；关系存成三元组 $(v_s,r,v_d)$。Neo4j 后端。
- **写入**：提取阶段两个 LLM 模块——entity extractor 抽实体、relationship generator 判实体对之间有没有关系并打标签。更新阶段先向量检索 top-10 相似记忆，再让 LLM 输出 **tool call：ADD/UPDATE/DELETE/NOOP**（新增/更新/删除/不动）。Mem0ᵍ 实体嵌入相似度 ≥ 阈值 $\Gamma$ 就合并去重。
- **检索**：Mem0 纯向量 top-k（k∈{1,2}）；Mem0ᵍ 双通道——先 entity-centric 找查询里的关键实体作锚点，再沿关系三元组算细粒度相似度，从锚点遍历**全部 incoming/outgoing 关系**构子图。
- **冲突**：LLM 判定该 UPDATE 还是 DELETE（主观）。
- **实验**：LOCOMO 长期对话基准。**关键反直觉证据**：Mem0ᵍ 总分只比基础版高约 **2%**，但搜索延迟约为 Mem0 的 **3 倍**（0.476s vs 0.148s）、token 消耗**翻倍**（3616 vs 1764）。即"图不总是更划算"。
- **与本研究差异**：它图的收益来自"实体间关系推理"（有限），我图的收益来自"发现语义检索看不见的方法学联系"；它冲突靠 LLM 主观覆盖，我靠 metric 客观判定 + 条件保留。**这张卡是你必须主动回应的反方证据**（见 §三速答卡 4）。

### 8.4 A-MEM（arXiv:2502.12110, NeurIPS'25 poster）⚠️据原论文

**一句话**：把每条记忆写成一张"卡片盒笔记"（Zettelkasten 风格），让 LLM 自主给笔记生成关键词/标签/上下文并自动建链——但**检索其实是纯向量的**，名字里的 "Agentic" 只体现在写入端。

- **结构**：Zettelkasten 笔记**同质图**。每张 note 含 K（关键词）、V（内容）、T（标签）、C（上下文描述），外加时间戳和 links。边只有 `similar_to` 一种（语义链接）。
- **写入（agent 自主）**：LLM 在记忆构造阶段自主生成 note 的结构化属性、判断与哪些旧 note 建链，并触发旧 note 的演化更新（重写）。
- **检索（纯向量、非 agent）**：§3.4 三个公式——all-MiniLM 编码 + 余弦相似度 + top-10，**全程无 LLM**。论文 §2.2 自白：agency 在"存储与演化"，检索让给了 agentic RAG。
- **冲突**：演化式 link-update（新笔记触发相关旧笔记重写），无专门冲突仲裁。
- **定位**：本研究 **baseline #1**——它是"图 + agent 写入"但检索仍是向量，正好对照出"图 + agent 检索"的空白。

### 8.5 MemGPT / Letta（arXiv:2310.08560）⚠️据原论文

**一句话**：把 LLM 当操作系统，给它一套"内存管理工具"，让 agent 自己决定把什么搬进上下文、什么存到外部库——agent 检索范式的源头，但记忆是**分层文本块、不是图**。

- **结构**：仿 OS 内存层级的三层文本块——**main/core memory**（始终在上下文里的 persona/human 块）、**recall memory**（对话历史）、**archival memory**（外部向量库）。非图。
- **写入 + 检索（均 agent function call）**：agent 通过显式函数调用操作记忆——`core_memory_append/replace`（改 core）、`archival_memory_insert`（写库）、`archival_memory_search`（查库）。**关键**：检索是 agent 自己决定调不调、查什么，这正是"agent 自主检索"的原型。
- **冲突**：覆盖式（insert/replace/append）。
- **定位**：本研究的 **agent 检索范式来源**，但它查的是平面文本块，我查的是结构化 Skill Graph。

### 8.6 Zep / Graphiti（arXiv:2501.13956）⚠️据原论文

**一句话**：给知识图谱的每条边都打上"何时生效/何时失效"两个时间戳，靠时间先后来仲裁冲突——是唯一用客观信号（时间）处理冲突的系统，但只能管"谁更新"这一维。

- **结构**：双时序知识图。两类节点 entity + event，每条边带 **`valid_at` / `invalid_at`**（bi-temporal：既记事件发生时间，也记信息被获知/失效时间）。
- **写入**：agent 触发 + 后端 pipeline 自动做抽取、去重、失效标记。
- **检索**：后端固定 pipeline——向量召回 + 图上 BFS 遍历 + reranker 重排。
- **冲突（时序失效）**：新信息让旧关系过期时**不删除**，而是给旧边打 `invalid_at` 时间戳，保留历史可回溯。**客观，但只覆盖时间维度**——解决不了"两个事实同时有效但矛盾"。
- **与本研究差异**：它只能判 dynamic（时间先后）冲突；MLE 的双真冲突（两次执行都真但结果矛盾、无时间先后）它判不了，我用 metric + 执行条件判。

### 8.7 其余系统（详见前文对应小节）

- **Trace2Skill**（arXiv:2603.25158）：蒸馏式写入的代表，本研究改进对象——详见 §3.4（三阶段管线、层级合并算子 M、prevalent bias、两个局限）。
- **Skill-SD / MemPO / MMPO**：参数化/内化记忆（记忆进权重、反检索）——详见 §7.1（自蒸馏 reverse-KL、双 advantage、Belief Entropy）。
- **InternAgent-1.5**（arXiv:2602.08990）：三层记忆系统（SPM 策略 / TEM 任务情节 / SKM 语义图）+ Generation-Verification-Evolution 循环——详见 §7.2。

---

## 9. 文献空白与本人研究方向

> 本节整合 `research_review_graph_trace2skill.md`（方向综述）与 `engineering_roadmap.md`（v3 路线图）的研究设计，使本篇成为自洽的「别人怎么实现 → 本人怎么做」完整文档。

### 9.1 三个空白象限（综述结论）

1. **图结构 + 运行时 agent 自主检索**：无人做（A-MEM 有图无 agent 检索；MemGPT 有 agent 检索无图；SAGE 是训练式前向非运行时）。
2. **节点是 procedural skill 的图**：所有图记忆节点都是事实/事件/实体/case，**没有 SOP 级方法学的图**（对照 §2.3 的认知分类法，Procedural + Experience 象限空白）。
3. **基于客观信号的冲突判定 + MLE 双真冲突**：现有冲突处理全靠 LLM 主观或物理规则；MemConflict/STALE 证明 static/dynamic 判定是瓶颈（§5.2）；MLE 的"两次真实执行矛盾"（双真冲突）无人研究。

### 9.2 方向声明与数据流

> **核心范式**：本研究**不从 SearchNode 直接建图**，而是**先用改进版 Trace2Skill 蒸馏出 skill，再把蒸馏产物组织成图**。
>
> **数据流**：`mlevolve trace → 改进 Trace2Skill 蒸馏 → Skill Graph → agent 多跳检索 → 冲突感知决策`。
>
> **关键定位**：图节点是 **Skill 蒸馏产物**，不是 SearchNode 本身；蒸馏是**周期性、离线**的，与 mlevolve 在线搜索解耦。

MLE 经验记忆相比对话记忆的四个本质差异（§6 已述）——执行派生的因果、metric 是 ground truth、双真冲突、可执行代码——每一个都"逼"出现有系统没有的新机制，构成本方向不可替代的基础。

### 9.3 主贡献 1：MLE 场景下的改进 Trace2Skill 蒸馏管线

与原版（§3.4）的关键差异即 novelty 边界：

| 维度 | Trace2Skill 原版 | 本研究改进 |
|------|------------------|-----------|
| **场景** | spreadsheet / VisionQA / math 等通用任务 | **MLE 工程经验**（含 metric / code / 失败 trace） |
| **冲突处理** | Stage 3 消除合并 + prevalent bias 丢弃低频 | **条件化保留 + metric 客观判定**（不丢低频正确经验） |
| **patch 验证** | LLM 主观判断 patch 价值 | **结合 metric_delta 客观验证**（哪个 patch 真有效） |
| **产出形态** | 单一 Markdown（线性） | **结构化 patch + 适用条件标注**，为建图准备 |

**流程**：Stage 1 从 `mlevolve/runs/` 抽 SearchNode root-to-leaf 路径 → Stage 2 success/error analyst 并行提 patch（每个带 `metric_delta` + `applies_when` + `source_trace_ids`）→ Stage 3 **不消除冲突**，按条件聚类保留（高频同条件→主体；低频但 metric 优→反常有效；不同条件→条件分支）。

**验证锚点**（离线、不烧算力）：以 `experience_kb/small-data-transformer-finetuning/insight.md` 的手工 15 条为 ground truth，自动蒸馏需满足——召回 ≥10/15、新发现 1–2 条、条件标注 100%、保留 2–3 对矛盾 patch（留给建图）。

### 9.4 主贡献 2：Skill Graph——蒸馏产物的图结构组织

把蒸馏出的 skill **进一步结构化为图**，让 SOP 间关系（适用条件 / 失败规避 / 冲突 / 演化）显式可遍历，避免线性 Markdown 丢失隐含联系。

**节点 6 类**：`Skill`（完整技能集）/ `SOP`（单条规则，如"用 partial unfreezing"）/ `Condition`（适用上下文，如"小数据集"）/ `FailureMode`（失败模式，如"全参微调天花板"）/ `Evidence`（trace_id + metric_delta）/ `Implementation`（代码模板）。

**边 7 类（精简版，v3）**：`contains`、`has_implementation`、`applies_when`(SOP→Condition)、**`prevents`(SOP→FailureMode，⭐核心)**、`refines`(SOP→SOP 条件分支)、`supported_by`(SOP→Evidence)、`conflicts_with`(SOP↔SOP 同条件矛盾)。

**核心建图算法 `detect_and_link_sop_relations`**：对每对 SOP，比较其 `prevents` 邻居（共享 FailureMode）与 `applies_when` 邻居（共享 Condition），再用 LLM 判文本是否对立——同条件 + 对立 → `conflicts_with`；不同条件 + 对立 → `refines`；同条件 + 同 FailureMode + 不对立 → 互补（agent 自行发现）。这是"用已有边推导 SOP 间关系"的自反建模，是图的不可替代价值所在。

**与现有图记忆的对比**：

| 维度 | A-MEM | EXG | SAGE | **本研究 Skill Graph** |
|------|-------|-----|------|----------------------|
| 节点是什么 | 笔记（对话事实） | case（轨迹） | entity + fragment | **Skill 蒸馏产物（多粒度 SOP）** |
| 节点产生时机 | 流式写入 | 流式每次交互 | 流式 writer 训练 | **周期性蒸馏后批量建** |
| 边的语义来源 | 语义相似 | 同任务 fixed_by + 相似 | 训练学到结构角色 | **方法学关系（适用/规避/冲突）** |
| 知识类型 | declarative | episodic | conceptual | **procedural** |

### 9.5 主贡献 3：Agent 自主多跳检索 Skill Graph

mlevolve 的 improve agent 在搜索中通过 tool **主动查询** Skill Graph，而非系统固定注入。Tool 设计（v3 精简到 3 个）：

| Tool | 用途 | 检索路径 |
|------|------|---------|
| `search_skill_with_paths(query, path_strategy)` | 核心检索（direct / via_failure_mode / via_condition / find_conflicts） | SOP →PREVENTS→ FailureMode ←PREVENTS← 其他 SOP（发现语义外互补 SOP） |
| `analyze_conflict(sop_a, sop_b)` | 判定真冲突 / 互补 / 条件分支 | 沿 PREVENTS / APPLIES_WHEN 比较共享 FailureMode 与 Condition |
| `get_implementation(sop_id)` | 取可用代码模板 | SOP →HAS_IMPL→ Implementation |

**集成最小化**：只改 `improve_agent.py`（加 ReAct + tool 分支）+ `agent_search.py`（加载图）；**debug_agent 不改，留作对照组**——给出干净 ablation。

### 9.6 副贡献：基于执行信号的冲突判定

**Motivation**（§5.2 硬证据）：MemConflict static 0.44 / dynamic 0.50，SEH@K vs AA gap 0.10–0.16 → 检索不是瓶颈，判定才是。MLE 的双真冲突只有 metric + 执行条件能判。

**核心逻辑**：矛盾候选对 (A,B) → 抽取条件（task_type / data_size / model_size / 失败模式）→ 条件相同？否 → 加 `refines`（条件分支，都保留）；是 → metric 比较：显著优 → `conflicts_with` + winner 标注；接近 → 两者保留 + 置信度警告。应用于蒸馏（§9.3 Stage 3）、建图（§9.4 关系检测）、演化（写回）三处。

> **方向版本说明**：`research_review`（中间版）把冲突判定列为与图检索并列的副贡献；`engineering_roadmap` v3 进一步收紧，将其**降级为图原生的沿边推理**（用 `prevents`/`refines` 边表达，而非单独的 static/dynamic 分类）。两版的取舍是组会可讨论的决策点——前者实验更全、后者方向更聚焦。

### 9.7 双层记忆协调（关键工程设计）

| 维度 | Layer A · Trace 记忆（保留现状） | Layer B · Skill Graph（新增） |
|------|------------------------------|----------------------------|
| 回答 | "这分支**已试过**什么？结果如何？" | "这种情况下**方法学**该怎么做？" |
| 时间尺度 | 当前 run（小时级） | 跨 run（周/月级） |
| 粒度 | 具体 code + metric | 抽象 SOP + 条件 |
| 呈现 | **强制注入**（必看） | **按需 tool 查**（agent 自主） |

两层**绝不合并**：只信 B → 重复犯错（不知道本分支刚试败什么）；只信 A → 永远局部探索。当前 run 内 Layer B **只读不写**，保证图稳定，蒸馏只对"足够多 trace 形成的稳定模式"生效。

### 9.8 与各竞品的精确 novelty 边界

| 对比对象 | 它的做法 | 本研究差异 |
|---------|---------|-----------|
| **A-MEM** | 图 + agent 写 + 纯向量检索 | 节点是 SOP（非笔记）；agent 运行时多跳检索 |
| **EXG** | case 图 + 固定三通道 ≤2 跳 | 节点是蒸馏 SOP（高一抽象层级）；检索路径 agent 运行时构造 |
| **SAGE** | entity 图 + 训练式 GFM 前向 | 节点是 SOP；运行时 agent 决策非固定前向 |
| **Trace2Skill** | 蒸馏成线性 Markdown + 反检索 | 蒸馏成图 + 多跳检索；保留低频 + 语义冲突建模 |
| **MemGPT** | 分层块 + agent 检索 | 查的是结构化 Skill Graph 非平面块 |
| **Mem0ᵍ** | 三元组图 + LLM 覆盖冲突 | metric 客观判定 + 条件保留（refines 边） |

### 9.9 实验框架

- **主实验（好不好）**：5 条件 × 3 任务 × 3 seed——(a) No-Mem / (b) Flat-Vec（mlevolve 现状）/ (c) Graph-Vec（EXG 式：图 + 固定向量+图遍历）/ (d) Flat-Agent（MemGPT 式：平面 + agent 检索）/ (e) Graph-Agent（本研究完整方法）。指标：最终 metric、收敛速度、重复错误率。
- **消融（图还是 agent 的功劳）**：w/o 因果边 / w/o 多跳 / w/o agent 自主 / w/o metric 判定。
- **★ 专项实验（v3 差异化，证"图发现语义外联系"）**：**Path Diversity@K**（图检索 SOP 文本方差是否更大）、**Hidden Link Recall**（能否召回向量找不到的"语义远但相关"SOP）、**冲突分类正确率**（图证据是否把 LLM 判定从 ~60% 提到 ~80%）。
- **冲突专项（判定有没有用）**：借 MemConflict 的 SEH@K vs AA 双层，对比 LLM 主观判定 vs metric 客观判定，预期 metric 判定显著提升 AA（尤其双真冲突）。
- **效率（慢不慢）**：检索延迟 p50/p95、token 成本、检索调用次数，论证"agent 检索单次贵但减少试错步数、端到端更快"。
- **泛化（过拟合吗）**：2–3 个 MLE-bench 任务（NLP / tabular / CV）跨任务迁移。

**实验预测**（分层可验证，避免全有或全无）：只做蒸馏 + Markdown 注入 → 性能基线；加图 + agent 检索 → SEH@K 显著提升；加冲突判定 → 矛盾场景 AA 显著提升。

### 9.10 可行性与风险

- **代码基础**：MLEvolve 框架完整（搜索树、记忆层、冷启动），只需增量改造（在线代码仅改 `improve_agent.py`）。
- **数据基础**：Spooky 任务 60+ 个 run（含丰富成功/失败案例）+ 手工 15 条 ground truth，离线验证不依赖新算力。
- **算力估算**：主实验 + 消融约 700–900 GPU·h；核心单任务验证约 12–16 GPU·h（1 周内见信号）。
- **投稿目标**：对标 A-MEM（NeurIPS'25 poster）→ **NeurIPS / ICML / ICLR poster**；冲刺 oral/spotlight 需极强实验。
- **主要风险**：① novelty 被质疑增量 → 用消融证"图 + agent 检索"的组合效应 > 各部分之和；② agent 检索延迟 → 报告效率指标论证总成本更优；③ 条件抽取依赖 LLM 归因 → 用 metric 反向校准 + 多人标注。

---

## 10. 参考文献

**图记忆**
1. A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110. NeurIPS 2025 (poster).
2. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
3. SAGE: Self-Evolving Agentic Graph-Memory Engine. arXiv:2605.12061.
4. EXG: Self-Evolving Agents with Experience Graphs. arXiv:2605.17721.
5. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.
6. Zep/Graphiti: Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.

**技能蒸馏 / 参数化记忆**
7. Trace2Skill: Distilling Trajectory-Local Lessons into Transferable Agent Skills. arXiv:2603.25158.
8. Skill-SD: Skill-Conditioned Self-Distillation. arXiv:2604.10674.
9. MemPO: Self-Memory Policy Optimization. arXiv:2603.00680.
10. MMPO: Meta-Cognitive Memory Policy Optimization (Belief Entropy). arXiv:2605.30159.

**冲突评估 / benchmark**
11. MemConflict: Evaluating Long-Term Memory Systems Under Memory Conflicts. arXiv:2605.20926.（本地代码）
12. STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? arXiv:2605.06527.
13. LongMemEval-V2: Evaluating Long-Term Agent Memory. arXiv:2605.12493.
14. MemoryArena: Multi-Session Interdependent Task Memory Benchmark. arXiv:2602.16313.

**survey / 系统**
15. Graph-based Agent Memory: A Survey. arXiv:2602.05665.
16. Evidence Tracing and Execution Provenance: A Survey. arXiv:2606.04990.
17. InternAgent-1.5: Unified Agentic Framework for Long-Horizon Scientific Discovery. arXiv:2602.08990.

**项目内部参考**
- `mlevolve/agents/memory/global_memory.py` / `retriever.py`：平面记忆 + 混合检索现状。
- `mlevolve/engine/search_node.py` / `agent_search.py`：搜索树与多阶段调度。
- `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md`：手工 Trace2Skill 产物（15 条 ground-truth insight）。
