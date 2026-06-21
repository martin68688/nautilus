# MLEvolve 记忆系统改造路线图（v3）：Skill Graph + 多跳冲突发现

> **方向声明（v3 收紧版）**
>
> 核心 narrative：**把 Trace2Skill 的 skill 蒸馏产物组织成图，让 agent 通过多跳遍历发现"语义外的方法学联系"，特别是在冲突处理时识别"表面矛盾但实则互补"的 SOP。**
>
> 这条路线的双重 novelty（基于 5 个图记忆系统的精确核对得出）：
> 1. **节点性质新**：节点是 skill（procedural knowledge），而 A-MEM/EXG/SAGE/Mem0g/Zep 全是事实/事件/实体/case
> 2. **图的价值新**：用方法学关系边（applies_when / prevents / refines）发现向量检索做不到的间接联系，这是图记忆领域第一次为 procedural knowledge 服务
>
> 数据流：`mlevolve trace → Trace2Skill 蒸馏 → Skill Graph → agent 多跳检索 → 冲突感知决策`

---

## 0. 核心 Narrative（一页纸钉死方向）

### 用一个具体场景说清楚整个 idea

```
情景：mlevolve 当前正在做 Spooky Author Identification 任务的 improve 阶段。
当前 SearchNode 用了 partial unfreezing，metric 0.0725。
agent 想问："还有什么方法能进一步降低 log loss？"

向量检索能做的：
  query="reduce overfitting on small NLP dataset"
  → 返回语义最近的 3 条 SOP：
     "use stronger dropout"
     "use weight decay"
     "use early stopping"
  问题：这些建议都和 query 语义近，但**互相之间也很近**——agent 拿到的是"同一类思路的不同变体"。

Skill Graph + 多跳检索能做的：
  start = 当前 SOP "partial unfreezing"
  walk: SOP → PREVENTS → FailureMode "small-data overfitting" ← PREVENTS ← 其他 SOP
  返回：
     "label smoothing"            ← 和 "partial unfreezing" 文本不近
     "longer epochs with warmup"  ← 和 "partial unfreezing" 文本不近
     "stratified k-fold CV"       ← 和 "partial unfreezing" 文本不近
  agent 看到："这三个 SOP 都在解决同一个 failure mode，但和我现在用的方法是
              **不同的解决路径**——是组合用，不是替代。"

冲突处理场景下的价值：
  agent 检测到 SOP_A "用 label smoothing" 和 SOP_B "不要用 label smoothing"
  传统做法：LLM 主观判断 / metric 比较 / 强制覆盖
  Skill Graph 做法：
    沿 SOP_A.PREVENTS 找到 FailureMode "overconfidence"
    沿 SOP_B.PREVENTS 找到 FailureMode "label noise insensitivity"
    → 两个 SOP 解决的是不同问题 → 不是真冲突，是适用条件不同
    → 加 REFINES 边（条件分支保留），不是 CONFLICTS_WITH（覆盖）

这就是图存在的理由：用方法学关系发现"语义检索看不见的隐含联系"。
```

### 这个 narrative 的 contribution 三件套

1. **节点是 skill**（图记忆领域空白 — 见 §1 精确核对）
2. **多跳遍历方法学关系**（向量检索做不到 — 见上面具体场景）
3. **冲突处理的图原生方案**（不是覆盖/消除/合并，是**沿边推理**）

### 砍掉的内容（v2 → v3 的纪律）

| v2 有 | v3 状态 | 原因 |
|-------|--------|------|
| static/dynamic 冲突分类（MemConflict 概念）| ❌ 砍 | 对话场景概念，对 MLE 不贴切 |
| 基于 metric delta 的冲突判定 | ⚠️ 降级 | 作为 §4 演化阶段的辅助信号，不是核心 |
| Memory Adoption 闭环（最初综述 §4.5）| 🟡 推后 | 写入 future work，避免方向太大 |
| SearchNode 直接做图节点（v1 错误）| ❌ 已砍 | v2 已纠正 |

**v3 只做三件事**：蒸馏 → 建图 → agent 多跳检索（含冲突处理）。

---

## 1. 与现有图记忆系统的精确差异（节点性质对比表）

这是你 related work 的核心论据。**节点是 skill 这件事，文献空白**。

| 系统 | 节点是什么 | 内容例子 | 粒度层级 | 对应知识类型 |
|------|----------|---------|---------|------------|
| **A-MEM** (NeurIPS'25) | Zettelkasten 笔记 (K, V, T, C) | "User Alice mentioned being vegetarian" | 事件级 | declarative（事实）|
| **EXG** | case 节点（轨迹快照）| (task_input, prompt, output, signature, reward) | 轨迹级 | episodic（经验）|
| **SAGE** | entity + memory fragment | "DeBERTa" 实体 + 关系三元组 | 实体级 | conceptual（概念）|
| **Mem0g** | entity + relation triplet | (Alice, lives_in, San_Francisco) | 实体级 | declarative（事实）|
| **Zep/Graphiti** | entity + event（带时序）| 实体 + 时间戳事件 | 实体级 + 时序 | declarative + temporal |
| **本研究 Skill Graph** | **Skill / SOP / Condition / FailureMode / Evidence / Implementation** | "use partial unfreezing when small dataset" | **方法学级** | **procedural（程序性）**|

### 为什么没人做 procedural knowledge 的图？

| 原因 | 解释 |
|------|------|
| 主流场景是对话 | 对话记忆天然是 declarative knowledge，KG 够用 |
| Procedural knowledge 难抽取 | 需要任务执行场景 + 客观评估信号 |
| Trace2Skill 做了蒸馏但没建图 | 它把 skill 写成 Markdown，丢失了 SOP 间隐含联系 |
| **MLE 场景被忽视** | LLM 时代才有 mlevolve 这种系统化 trace 库 |

**你恰好处在三个条件交汇处**：MLE 场景（有客观信号）+ Trace2Skill 范式（蒸馏方法成熟）+ 图记忆领域（关系建模成熟）。这是 NeurIPS 2026 的窗口期。

---

## 2. 现状盘点（极简版）

```
P0 现状（基于代码事实）

mlevolve（在线搜索）        paper-skills（离线知识）
─────────────────────       ────────────────────────
SearchNode 树 ✅              experience_kb/ 手工 skill ✅
GlobalMemoryLayer (平面) ✅   methodology_kb/ 论文知识 ✅
BM25+FAISS 固定检索 ✅        (无图，无自动蒸馏)
                              
缺什么：
  ❌ Trace2Skill 自动蒸馏管线
  ❌ Skill Graph 结构
  ❌ Agent 多跳检索 tool
```

**关键资产**：你的 `experience_kb/small-data-transformer-finetuning/insight.md`（手工 15 条带 confidence 的 SOP）是 P1 蒸馏的 ground truth。

---

## 3. P1：Trace2Skill 蒸馏管线（2 周）

### 目标
从 mlevolve trace 自动蒸馏出 skill **patches 列表**（注意：不是直接产 Markdown，而是产**结构化 patch**，因为 P2 要把它们建图）。

### 与 v2 的差异
v2 想产出 Markdown 文档。**v3 直接产出图就绪的结构化 patch**——一步到位，不绕路。

### 新增文件

#### `paper-skills/distillation/patch_proposer.py`

```python
"""
Trace2Skill Stage 2 改进版：产出图就绪的结构化 patch。
patch 字段直接对应 P2 的 Skill Graph 节点 schema。
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StructuredSkillPatch:
    """直接为图节点准备的 patch（不是 Markdown）。"""
    patch_id: str

    # 主要节点候选
    sop_text: str                          # → 未来的 SOP 节点
    applies_when: List[str]                # → Condition 节点（文本短语，P2 拆成节点）
    prevents: List[str]                    # → FailureMode 节点
    requires: List[str]                    # → 前置 SOP（依赖）
    implementation_snippet: Optional[str]  # → Implementation 节点

    # 客观证据（关键差异：MLE 独有）
    metric_delta: Optional[float]          # 来自 trace 的指标变化
    source_trace_ids: List[str]            # 反向归因
    confidence: str                        # HIGH / MEDIUM / LOW（基于 metric_delta + 出现频次）

    # 蒸馏元数据
    success_or_failure: str                # 来自成功 trace 还是失败 trace
    analyst_id: str                        # 哪个 analyst 提的（多 analyst 并行）
```

#### `paper-skills/distillation/run_distill.py`

```python
"""
完整蒸馏 pipeline：
  Stage 1: TraceCollector 从 mlevolve/runs/ 抽 root-to-leaf trace
  Stage 2: 并行多 analyst 提 StructuredSkillPatch
  Stage 3: 简单去重（不做激进合并，把矛盾保留给 P2 的图建模处理）

关键差异 vs Trace2Skill 原版：
  - 原版 Stage 3 用 prevalent bias 丢弃低频 → 我们保留所有 patch
  - 矛盾的 patch 都进 P2，由图边（CONFLICTS_WITH / REFINES）表达
  - 这样不丢信息，而且让"冲突"成为图的一等公民
"""
def run_distillation(runs_dir, output_dir, frozen_skill=None):
    traces = TraceCollector().collect(runs_dir)
    patches = ParallelPatchProposer(n_analysts=4).propose_all(traces, frozen_skill)
    deduplicated = simple_dedup(patches)  # 仅文本完全相同的去重
    save_patches(deduplicated, output_dir)
    return deduplicated
```

### P1 验证 Milestone

| 验证项 | 通过标准 |
|-------|---------|
| 自动 patch 数 | ≥ 30 条（Spooky 8 个 run）|
| 召回手工 ground truth | ≥ 10/15 条手工 insight 在自动 patch 中找到 |
| 每个 patch 字段完整 | 100% 有 metric_delta + applies_when + source_trace_ids |
| 矛盾 patch 保留 | 至少检测出 2-3 对矛盾 patch（不丢，留给 P2）|

**P1 的产出是 JSON 文件，不是 Markdown**。这是和 v2 的关键差异——直接为 P2 的图建模做准备。

---

## 4. P2：Skill Graph 构建（2 周）

### 目标
从 P1 的 patch 列表构建图。**这一阶段是你"用图发现语义外联系"的物质基础**。

### 节点 Schema（v3 收紧版）

我把 v2 的"三选一"决定了——**采用 6 类节点**（你之前的选项 B），但简化字段：

```python
class NodeType:
    SKILL = "skill"             # 一个完整技能集合（如 small-data-transformer-finetuning）
    SOP = "sop"                 # 单条规则（如 "use partial unfreezing"）
    CONDITION = "condition"     # 适用上下文（如 "small dataset"）
    FAILURE_MODE = "failure_mode"  # 失败模式（如 "overconfidence"）
    EVIDENCE = "evidence"       # 客观证据（trace_id + metric_delta）
    IMPLEMENTATION = "implementation"  # 代码模板
```

### 边 Schema（v3 精简到 7 类）

只保留与"多跳推理"和"冲突处理"直接相关的边类型：

```python
class EdgeType:
    # 内部组织（结构边）
    CONTAINS = "contains"            # Skill → SOP / FailureMode
    HAS_IMPL = "has_implementation"  # SOP → Implementation

    # 方法学关系（核心：图的不可替代价值）
    APPLIES_WHEN = "applies_when"    # SOP → Condition
    PREVENTS = "prevents"            # SOP → FailureMode    ⭐ 核心
    REFINES = "refines"              # SOP → SOP（条件分支细化）

    # 证据
    SUPPORTED_BY = "supported_by"    # SOP → Evidence

    # 冲突
    CONFLICTS_WITH = "conflicts_with"  # SOP ↔ SOP（同条件矛盾）
```

**砍掉的边类型**（vs v2）：
- `requires`（依赖）：太复杂，先用 CONTAINS 替代
- `causes`（反向）：用 PREVENTS 反向遍历就行，不需要专门的边
- `metric_delta`（数值边）：作为 Evidence 节点的属性，不是独立边
- `generalizes_from`（蒸馏来源）：放在 Evidence.source_trace_ids 字段

**为什么砍**：边越少，agent 检索 tool 越容易设计，实验越好做。MVP 优先。

### 关键算法：从 patch 到图（自动建边逻辑）

```python
def build_graph_from_patches(patches: List[StructuredSkillPatch]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Step 1: 创建 SOP 节点（一个 patch 一个 SOP）
    for p in patches:
        sop_id = f"sop_{p.patch_id}"
        G.add_node(sop_id, type=NodeType.SOP, text=p.sop_text,
                   confidence=p.confidence)

        # Step 2: 创建/链接 Condition 节点（去重）
        for cond_text in p.applies_when:
            cond_id = canonicalize(cond_text)  # "small dataset" 和 "few samples" 合并
            if cond_id not in G:
                G.add_node(cond_id, type=NodeType.CONDITION, text=cond_text)
            G.add_edge(sop_id, cond_id, kind=EdgeType.APPLIES_WHEN)

        # Step 3: 创建/链接 FailureMode 节点（去重，⭐ 关键）
        for fm_text in p.prevents:
            fm_id = canonicalize(fm_text)
            if fm_id not in G:
                G.add_node(fm_id, type=NodeType.FAILURE_MODE, text=fm_text)
            G.add_edge(sop_id, fm_id, kind=EdgeType.PREVENTS)

        # Step 4: Evidence 节点
        ev_id = f"ev_{p.patch_id}"
        G.add_node(ev_id, type=NodeType.EVIDENCE,
                   metric_delta=p.metric_delta,
                   trace_ids=p.source_trace_ids)
        G.add_edge(sop_id, ev_id, kind=EdgeType.SUPPORTED_BY)

        # Step 5: Implementation 节点（如有）
        if p.implementation_snippet:
            impl_id = f"impl_{p.patch_id}"
            G.add_node(impl_id, type=NodeType.IMPLEMENTATION,
                       code=p.implementation_snippet)
            G.add_edge(sop_id, impl_id, kind=EdgeType.HAS_IMPL)

    # Step 6: 检测 SOP 间冲突和细化关系（⭐ 关键的图建模步骤）
    detect_and_link_sop_relations(G, patches)

    return G

def detect_and_link_sop_relations(G, patches):
    """
    核心逻辑：通过共享 FailureMode + Condition 检测 SOP 关系
    这是图的"自反"建边——基于已有的 PREVENTS / APPLIES_WHEN 边推理出 SOP 间关系
    """
    sops = [n for n in G.nodes if G.nodes[n]["type"] == NodeType.SOP]
    for sop_a, sop_b in itertools.combinations(sops, 2):
        # 共享 failure mode（PREVENTS 邻居重叠）
        prevents_a = set(G.successors_with_edge(sop_a, EdgeType.PREVENTS))
        prevents_b = set(G.successors_with_edge(sop_b, EdgeType.PREVENTS))
        shared_fm = prevents_a & prevents_b

        # 共享 condition
        cond_a = set(G.successors_with_edge(sop_a, EdgeType.APPLIES_WHEN))
        cond_b = set(G.successors_with_edge(sop_b, EdgeType.APPLIES_WHEN))
        shared_cond = cond_a & cond_b

        # 文本是否语义对立（用 LLM 判定）
        is_opposite = llm_judge_opposite(G.nodes[sop_a]["text"], G.nodes[sop_b]["text"])

        # 决策树
        if shared_cond and is_opposite:
            # 同条件 + 对立 → 真冲突
            G.add_edge(sop_a, sop_b, kind=EdgeType.CONFLICTS_WITH)
        elif shared_cond and shared_fm and not is_opposite:
            # 同条件 + 同 failure mode + 不对立 → 互补 SOP（不需要边，agent 自己会发现）
            pass
        elif not shared_cond and is_opposite:
            # 不同条件 + 对立 → REFINES（条件分支）
            G.add_edge(sop_a, sop_b, kind=EdgeType.REFINES)
```

⭐ **这个 `detect_and_link_sop_relations` 函数是 v3 路线图的核心创新点**——它从 patch 自动推导 SOP 间关系，是图的"自反建模"能力。

### P2 验证 Milestone

| 验证项 | 通过标准 |
|-------|---------|
| 节点数 | 60-100（30 patch × 平均 2-3 个 Condition/FailureMode）|
| FailureMode 节点共享 | 至少 1 个 FailureMode 被 ≥ 3 个 SOP 通过 PREVENTS 边连接（这是多跳检索的物质基础）|
| CONFLICTS_WITH 边检出 | 你 insight.md 里"简单 Linear 头 vs 复杂注意力头"应被识别 |
| REFINES 边检出 | "全参数微调" 和 "partial unfreezing" 在不同条件下应被识别为 REFINES |
| 可视化 | 导出 GraphML，用 Gephi 打开能看出"hub 型 FailureMode 节点"|

---

## 5. P3：Agent 多跳检索 + 冲突处理（核心实验，2 周）

### 重大设计变更（v3 vs v2）

v2 把"agent 检索"和"冲突处理"分成两个独立阶段（P3 + P4）。
**v3 把它们合并成一个**——因为冲突处理就是 agent 多跳检索的特化 use case。

### Tool 设计（v3 简化版）

只设计 3 个 tool（v2 是 4 个）：

```python
class SkillGraphToolkit:

    # ⭐ Tool 1：核心检索 tool（覆盖 80% 场景）
    def search_skill_with_paths(
        self,
        query: str,
        path_strategy: str = "auto",
        # auto / direct / via_failure_mode / via_condition / find_conflicts
    ) -> str:
        """
        agent 调用形式：
          search_skill_with_paths("how to handle small NLP dataset",
                                  path_strategy="auto")

        path_strategy 含义：
          - direct: 仅向量相似（baseline）
          - via_failure_mode: SOP → PREVENTS → FailureMode ← PREVENTS ← 其他SOP
                              （⭐ 核心：发现"语义外但解决同一问题"的互补SOP）
          - via_condition: SOP → APPLIES_WHEN → Condition ← APPLIES_WHEN ← 其他SOP
                          （发现"同条件下"的所有相关SOP）
          - find_conflicts: SOP → CONFLICTS_WITH / REFINES → 其他SOP
                          （冲突处理专用）
          - auto: agent 自动选（也可以让 LLM 决定）
        """
        ...

    # Tool 2：冲突分析 tool（特化）
    def analyze_conflict(self, sop_a_id: str, sop_b_id: str) -> dict:
        """
        给定两个看起来矛盾的 SOP，分析它们是真冲突还是隐含互补。

        返回 (基于图遍历的分析结果)：
        {
          "type": "real_conflict" | "complementary" | "conditional_branch",
          "shared_failure_modes": [...],   # 沿 PREVENTS 共享的 FM
          "shared_conditions": [...],       # 沿 APPLIES_WHEN 共享的 Condition
          "metric_evidence": {...},         # Evidence 节点对比
          "recommendation": "..."           # 给 agent 的建议
        }

        例子：
          analyze_conflict("use_label_smoothing", "use_longer_epochs")
          → {
              "type": "complementary",
              "shared_failure_modes": ["small-data overfitting"],
              "shared_conditions": ["small dataset"],
              "recommendation": "These two SOPs prevent the same failure mode
                                via different mechanisms. Combine them, don't choose."
            }
        """
        ...

    # Tool 3：实现获取 tool
    def get_implementation(self, sop_id: str) -> str:
        """
        从 SOP 沿 HAS_IMPL 边返回代码模板。
        (简单 tool，主要是为了方便 agent 直接拿到可用代码)
        """
        ...
```

### 集成到 mlevolve（最小改动）

只改 **2 个文件**（v2 改了 4 个）：

#### `agents/improve_agent.py`

在 `_diff_improve` 顶部加分支：

```python
def _diff_improve(agent, prompt_base, data_preview, parent_node):
    if not agent.acfg.use_skill_graph:
        return _diff_improve_legacy(...)  # 现状保留

    # 新路径
    toolkit = agent.skill_graph_toolkit
    final_plan = run_react(
        agent, parent_node,
        local_memory=parent_node.fetch_child_memory(),  # Layer A 保留
        skill_graph_tools=[
            toolkit.search_skill_with_paths,
            toolkit.analyze_conflict,
            toolkit.get_implementation,
        ],
        max_iter=4,  # 最多4次工具调用
    )
    return diff_generate_and_apply(agent, final_plan, parent_node.code, ...)
```

#### `engine/agent_search.py`

```python
class AgentSearch:
    def __init__(self, ...):
        # 加载 Skill Graph
        if self.acfg.use_skill_graph:
            graph = load_skill_graph(self.cfg.exp_id)
            self.skill_graph_toolkit = SkillGraphToolkit(graph)
```

**debug_agent 不在 v3 改造**——保留它用现有的固定向量检索作为对照组。这给你一个干净的 ablation：相同任务下，只 improve 阶段用 Skill Graph，看效果。

---

## 6. ⭐ 核心实验：语义外联系发现（v3 关键差异化）

这是 v3 比 v2 多的一个**专项实验**，专门证明你那个 motivation（"图能发现语义外的联系"）。

### 实验设计

**指标 1：Path Diversity@K**（自创指标，但概念清晰）

```
对每个测试 query：
  - 用纯向量检索取 top-K SOP
  - 用 Skill Graph 多跳取 top-K SOP（path_strategy=via_failure_mode）
  - 计算两组 SOP 的"文本相似度方差"
    - 向量检索：方差应较小（都是语义近的）
    - 图检索：方差应较大（包含语义远但因果相关的）

衡量：图检索是否真的能找到"语义外"的 SOP
```

**指标 2：Hidden Link Recall**

```
人工标注 20 个测试 case：每个 case 给一个 anchor SOP，
人工列出 5-10 个"专家会同时考虑"的相关 SOP（含语义近的和语义远的）。

对比：
  - 向量检索召回率（找到几个？）
  - 图多跳召回率（找到几个？）

预期：图多跳能找到的"语义远但相关"的 SOP，向量检索召回率应低于 30%；
      图多跳应能召回 60% 以上。
```

**指标 3：冲突场景的正确分类率**

```
人工构造 20 对 SOP（含真冲突 / 互补 / 条件分支），
让 analyze_conflict tool 自动分类，对比人工标注。

对比：
  - LLM 直接判断（baseline）
  - 图多跳推理 + LLM 判断（你的方法）

预期：图能提供共享 FailureMode / Condition 等结构化证据，
      让 LLM 判断准确率从 ~60% 提升到 ~80%。
```

**这三个指标合在一起，是你"图不可替代"的硬证据**。比单纯比 mlevolve 任务最终 metric 更直接、更说服力。

### 主任务实验（mlevolve 端到端）

继续保留 v2 的主实验设计，但简化对照组到 3 个：

| 条件 | 配置 | 预期 |
|------|------|------|
| Baseline-A | 关 global memory | 下界 |
| Baseline-B | 现状（平面 + 固定向量）| 当前基线 |
| **Treatment** | **Skill Graph + agent 多跳检索** | **核心 claim** |

**约 12 GPU-h**（比 v2 的 16 少）。决定方向是否成立。

---

## 7. P4：Skill Graph 演化（write-back，可选 1 周）

### 目标
新 trace 来后，把新 patch 合并进现有图。

### 简化版（v3 把 v2 的复杂 Evolver 简化）

```python
def evolve_graph(graph, new_patches):
    """简单版：依赖 P2 的 detect_and_link_sop_relations 处理冲突。"""
    for patch in new_patches:
        # 找匹配的现有 SOP
        match = find_similar_sop(graph, patch)
        if match is None:
            add_sop_to_graph(graph, patch)
        else:
            # 增强 evidence（不删除旧的）
            add_evidence_to_existing(graph, match, patch)

    # 重跑关系检测
    detect_and_link_sop_relations(graph, all_patches_so_far)
```

**v2 vs v3 的差异**：
- v2 在演化阶段做 metric 客观判定（static/dynamic 分类）
- v3 把判定全部交给 P2 的 `detect_and_link_sop_relations`——它本来就支持冲突识别。**演化只做"加节点"和"重跑边检测"**。

P4 是 nice-to-have，不影响核心论文。**时间紧可以砍**。

---

## 8. 设计决策（已定）

v3 替你决定了大部分（避免你纠结）：

| 决策 | v3 方案 | 理由 |
|------|---------|------|
| 节点粒度 | 6 类（B 选项）| 平衡表达力和实现复杂度 |
| 图后端 | NetworkX | 万级节点足够，重点不在性能 |
| 蒸馏触发 | 人工触发（脚本）| 论文不需要 online 演化 |
| 边类型 | 7 类（精简）| 越少越好做实验 |
| Tool 数量 | 3 个（精简）| 覆盖 80% 场景 |
| debug_agent 改造 | ❌ 不改 | 留作对照组 |

**还需你确认的（仅 1 项）**：

> 蒸馏的 LLM 用什么模型？
> A. DeepSeek-Chat（你 mlevolve 现在用的）—— 一致性好
> B. GPT-4 / Claude —— 蒸馏质量更高但贵
>
> 推荐：A，先把方法跑通，模型升级是 future work。

---

## 9. 时间线（v3 紧凑版）

```
Week 1-2:  P1 蒸馏管线（写 patch_proposer + run_distill）
Week 3:    P1 验证（自动 patch vs 手工 insight.md，召回率分析）
           → 里程碑：自动产 30+ patch，召回 ≥ 10/15

Week 4-5:  P2 Skill Graph 构建（schema + builder + 关系检测）
Week 5:    P2 验证（图可视化 + FailureMode 共享分析）
           → 里程碑：能看到 hub 型 FailureMode 节点

Week 6:    P3 toolkit + improve_agent 改造
Week 7:    P3 实验（专项 + 主实验，约 12 GPU-h）
           → 里程碑：Path Diversity / Hidden Link Recall 显著优于向量检索

[决策点]   通过 → 写论文；不通过 → 检查 schema 设计

Week 8-9:  论文写作 + 补 P4（可选）
Week 10:   投稿
```

**总工期：10 周（v2 是 7-8 周但更松散）**。v3 更紧但每个阶段产出更聚焦。

---

## 10. v1 → v2 → v3 演化对比（避免再混淆）

| 维度 | v1（已废）| v2（中间版）| **v3（当前）**|
|------|----------|-----------|-------------|
| **核心 narrative** | 图+agent检索 | Trace2Skill范式+Skill Graph | **多跳发现语义外联系**|
| **图节点是什么** | SearchNode | Skill 蒸馏产物 | **Skill 蒸馏产物（不变）**|
| **冲突处理** | static/dynamic 分类 | 条件化 + metric 判定 | **图原生**（沿 PREVENTS/REFINES 边推理）|
| **Adoption 闭环** | P4 完整章节 | P4 简化 | **推为 future work**|
| **Tool 数量** | 4 个 | 4 个 | **3 个（精简）**|
| **改 mlevolve 文件数** | 6 | 4 | **2（improve_agent + agent_search）**|
| **专项实验** | 无 | 无 | **⭐ Path Diversity / Hidden Link Recall**|
| **方向纪律** | 散 | 中 | **聚焦**|

### v3 的核心收紧

**砍掉**：
- ❌ static/dynamic 冲突分类（MemConflict 概念，对话场景的）
- ❌ Adoption 闭环（最初综述 §4.5，但太大，推后）
- ❌ debug_agent 改造（留作对照）
- ❌ 流式蒸馏（人工触发就够）

**保留 + 强化**：
- ✅ Trace2Skill 蒸馏管线（节点来源）
- ✅ Skill Graph 多粒度节点（核心结构）
- ✅ Agent 多跳检索（核心机制）
- ✅ ⭐ **新增专项实验**：证明"图发现语义外联系"

---

## 11. ⭐ 双层记忆协调设计（关键工程问题）

> **核心原则**：Layer A（trace 记忆，现状保留）和 Layer B（Skill Graph，新增）**绝对不能合并、不能互相替代**——它们回答根本不同的问题。本节详细讲两层如何协调。

### 11.1 不可替代性

| 维度 | Layer A（trace 记忆，**保留**）| Layer B（Skill Graph，**新增**）|
|------|----------------------------|----------------------------|
| **回答的问题** | "这个分支**已经试过**什么？结果如何？" | "在这种情况下，**方法学**告诉我该怎么做？" |
| **语气** | 描述性（事实陈述）| 规范性（方法建议）|
| **时间尺度** | 当前 run（小时级）| 跨 run（周/月级）|
| **信息粒度** | 具体 code + 具体 metric | 抽象 SOP + 适用条件 |
| **来源** | 流式写入（每节点解析完立即）| 批量蒸馏（周期性）|
| **典型用途** | 避免重复、看父子改进链 | 找新方向、查方法学规律 |
| **优先级** | **必看**（agent 强制读）| **按需**（agent tool 决定）|

**互补性证明**：
- Layer B 不知道你"刚才在这个分支试了 DeBERTa-large 失败"——它只知道"历史上 DeBERTa-large 在小数据上有效"。**只信 Layer B → agent 重复犯错**。
- Layer A 不知道"在小数据集上 partial unfreezing 通常胜过 full finetuning"——它只看到当前分支 5-10 个尝试。**只信 Layer A → agent 永远在局部探索**。

### 11.2 信息流向（一图说清）

```
                  当前 run（在线，分钟-小时级）
   ┌─────────────────────────────────────────────────────┐
   │  SearchNode 生成 ──写入──> Layer A (fetch_child_memory)│
   │       ↑                              │               │
   │       │读                            │读              │
   │  ┌────┴───────────────────────────┐ │               │
   │  │  improve_agent (ReAct loop)    │←┘               │
   │  │  Step 1: 读 Layer A（强制）     │                  │
   │  │  Step 2: 决定要不要查 Layer B   │←──tool call──┐  │
   │  │  Step 3: 整合两层信息生成 plan  │               │  │
   │  └────────────────────────────────┘               │  │
   └─────────────────────────────────────────────────────│──┘
                                                          │
                  跨 run（离线，周期性，天-周级）           │
   ┌──────────────────────────────────────────────────────┴──┐
   │   多个 run 完成 ──> trace 收集 ──> Trace2Skill 蒸馏       │
   │                                          │               │
   │                                          ↓               │
   │                                     Layer B 更新（图）    │
   └──────────────────────────────────────────────────────────┘
```

**两个关键事实**：
- 当前 run 内，Layer B **只读不写**——保证图稳定，不被单次实验污染
- Layer A → Layer B 是**通过 trace 收集 + 蒸馏间接实现**，不是直接拷贝

### 11.3 Agent 看到两层的具体方式（分层呈现）

```python
# improve_agent 的 ReAct prompt 结构
prompt = f"""
你正在做 mlevolve 搜索的 improve 阶段。

## 当前任务
{task_description}

## 当前父节点（你要改进的 baseline）
Code: {parent_node.code}
Metric: {parent_node.metric}

## 【Layer A: 本分支已尝试】（事实记录，必看，避免重复）
{parent_node.fetch_child_memory()}

例如：
  Attempt #1: 试了 DeBERTa-large 全参数微调，metric 0.31（劣化）
  Attempt #2: 试了简单 dropout，metric 0.27（持平）

## 任务历史最佳参考（Layer B 自动注入，强制信号）
历史最佳 metric: 0.0725（路线：partial unfreezing）
当前分支最佳: 0.27
⚠️ 当前分支远低于历史最佳，建议考虑切换方向

## 【Layer B: 方法学知识库】
你可以通过以下 tool 主动查询跨 run 的方法学知识：

  - search_skill_with_paths(query, path_strategy)
  - analyze_conflict(sop_a, sop_b)
  - get_implementation(sop_id)

## 你的任务
1. 仔细阅读 Layer A，避免重复已经试过的方法
2. 必要时调用 Layer B 的 tool 查找新方向或方法学指导
3. 综合两层信息，提出改进 plan
"""
```

**关键设计点**：
- **Layer A 强制注入**——agent 一定能看到本分支已尝试
- **Layer B 通过 tool 按需查**——agent 自主决定调不调
- **历史最佳基准**作为 Layer B 的"轻量信号"自动注入（不需要 tool call）

### 11.4 冲突场景：两层矛盾的处理（最关键）

**场景 1**：Layer B 推荐的方法 Layer A 已尝试且失败
- 例：Layer B 说"用 DeBERTa-large"（HIGH），Layer A 显示已试过且 metric=0.31
- 处理：**信 Layer A，但向 Layer B 追问"为什么"**
  ```
  Step 1: 不要再试 DeBERTa-large
  Step 2: 调 Layer B 查"DeBERTa-large 失败对应哪个 FailureMode？"
  Step 3: 沿 PREVENTS(reverse) 找其他规避同 FailureMode 的 SOP
  ```

**场景 2**：Layer B 没相关知识，Layer A 也没尝试过
- 处理：**正常 LLM 探索**（冷启动场景，Skill Graph 不够用是正常的）

**场景 3**：Layer A 当前最佳远低于 Layer B 历史最佳
- 处理：**主动切换方向**——通过 §11.3 的"历史最佳参考"自动提示，agent 调 `path_strategy=via_failure_mode` 找其他路线

### 11.5 ReAct prompt 中的"两层冲突推理模式"

把上面三种场景写进 system prompt：

```
两层记忆冲突处理原则：

1. 当 Layer B 建议 X，而 Layer A 显示 X 已尝试且失败：
   ✗ 不要重复 X
   ✓ 查 Layer B：X 对应哪些 FailureMode？
   ✓ 找规避相同 FailureMode 但不是 X 的其他 SOP

2. 当 Layer A 和 Layer B 都没相关信息：
   → 正常探索（cold start）

3. 当 Layer A 当前最佳远低于 Layer B 历史最佳：
   → 当前路线可能不对
   → 调 Layer B 用 path_strategy=via_failure_mode 找其他路线
```

### 11.6 写入路径的解耦（保证 Layer B 稳定）

```python
# 现状（保留）：每个 SearchNode 解析完立即写 Layer A
result_parse_agent.run() 末尾：
    _save_to_global_memory(agent, node)  # 写 GlobalMemoryLayer ✅

# 新增（独立脚本）：周期性收集 trace，蒸馏后更新 Layer B
$ python paper-skills/distillation/run_distill.py \
    --runs-dir mlevolve/runs/ \
    --output paper-skills/skill_graph/ \
    --since-last-update
```

**核心**：当前 run 内，Layer B **绝对不被在线写入**。这保证了：
- 单次 run 的特殊情况不污染图
- 蒸馏只对"足够多 trace 形成的稳定模式"生效
- Layer B 演化是**审慎的、批量的、可回滚的**

### 11.7 端到端例子：跑 Spooky 第 9 个 run

假设前 8 个 run 已蒸馏进 Skill Graph：

```
Step 1（draft 阶段）：
  Layer A: 空（root 节点）
  Layer B: agent 自主调 search_skill_with_paths("spooky text classification")
           返回：partial unfreezing SOP（HIGH）+ stylometric features SOP（MED）
  → 生成 draft，用 partial unfreezing

Step 5（improve 阶段）：
  Layer A:
    Attempt #1: partial unfreezing on top 4 layers, metric 0.0850
    Attempt #2: partial unfreezing + label smoothing, metric 0.0810
    Attempt #3: partial unfreezing + stylometric concat, metric 0.0820
  Layer B: agent 看到 Layer A 已探索 partial unfreezing 路线
           主动调 search_skill_with_paths("further reduce log loss",
                                          path_strategy="via_failure_mode")
           返回：通过 FailureMode "small-data noise" 共享的其他 SOP：
                 - cosine warm restart
                 - heterogeneous ensemble
  → improve plan：尝试 cosine warm restart + heterogeneous ensemble

Step 12（debug 阶段）：
  Layer A: Attempt #4 引入 ensemble 后报 IndexError
  Layer B: agent 调 trace_failure_to_fix("IndexError in submission CSV")
           沿 FailureMode "submission format mismatch" → SOP "validate columns first"
  → 修复
```

**关键观察**：
- Layer A **始终在 prompt 里**（agent 强制看到本分支事实）
- Layer B 由 **agent 主动决定**何时调（哪步调、调什么 strategy）
- 两层通过 ReAct **推理模式**协调（"已试过 → 找 FailureMode → 找替代"）

### 11.8 协调设计的代码改动清单（最小化）

| 改动位置 | 改什么 | 工作量 |
|---------|--------|------|
| `agents/improve_agent.py` | 新建 `build_layer_a_section()`，含历史最佳基准 | 半天 |
| `agents/improve_agent.py` | system prompt 加"两层冲突推理模式" | 1 小时 |
| `agents/memory/skill_graph_tools.py` | 加 `get_task_historical_best()` 方法 | 1 天 |
| `paper-skills/distillation/run_distill.py` | 独立蒸馏脚本（在线代码无关）| 见 P1 |

**核心原则**：在线代码（mlevolve）只有 `improve_agent.py` 需要改；其他全是离线管线。两层解耦得很干净。

---

## 12. 给博士生的执行建议

> **v3 的纪律**：只做三件事——蒸馏（产 patch）、建图（含关系检测）、agent 多跳查（含冲突处理）。任何超出这三件事的想法，写到 `future_work.md`，不要塞进当前论文。

未来工作清单（不要在当前论文做）：
- Memory Adoption 闭环（最初综述 §4.5）
- static/dynamic 冲突的 MemConflict 评估
- 跨任务跨模型迁移
- 流式蒸馏 + online graph 演化
- skill graph 训练式 reader（SAGE 风格）

这些都是好方向，但**留给下一篇论文**。当前论文聚焦：**用图发现语义外的方法学联系**。
