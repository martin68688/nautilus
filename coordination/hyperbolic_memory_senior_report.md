# Agentic Hyperbolic SOP Memory 阶段性汇报

> 面向组内讨论 / 师兄汇报  
> 项目位置：`Nautilus / MLEvolve` 外部程序性记忆模块  
> 当前版本：v0 bootstrap hyperbolic SOP memory + online V1 agentic navigator runtime  
> 核心口径：本阶段已经把现有 SkillGraph-C compact-card 产物升级成一张可导航的双曲 SOP 记忆地图，并完成运行时 agentic navigation 接入；但最终论文级的 transition-level SOP distillation、真实 `metric_delta` 半径监督、以及严格几何检索消融仍是下一阶段工作。

---

## 1. 一句话总结

本阶段工作把原本扁平或普通图结构的 procedural skill 记忆，重构成一张 **Agentic Hyperbolic SOP Memory**：SOP 是可执行的方法点，Skill 是地图区域，Condition / FailureMode / Evidence 是导航锚点；系统先离线构建双曲坐标与方法论边，再在代码生成前让 `MemoryNavigator` 主动查看地图、选择方向、检查冲突，并把精炼后的 memory pack 注入到 MLEvolve agent。

直观类比：

- 过去的 skill memory 像“搜索框”：给一个 query，返回 top-k 文本。
- SkillGraph 像“普通平面流程图”：有依赖边，可以拓扑排序。
- 我们现在做的是“方法地图”：中心是高置信、通用方法；外圈是低频、条件更窄但可能关键的方法；方向代表语义类别；agent 可以先看地图，再决定往哪个区域走、沿什么边扩展、哪些方法存在冲突。

---

## 2. 背景：为什么要做结构化程序性记忆

MLEvolve 这类自动机器学习 agent 在搜索过程中会反复遇到相似问题：数据泄露、过拟合、OOM、API 不兼容、训练不稳定、错误的验证划分、复杂模型退化等。单次 run 中的经验如果只以日志或代码 diff 存在，很难在后续任务中稳定复用。

已有 memory/RAG 方法通常能解决“找相似文本”，但在我们的场景里还缺三个能力：

1. **条件化适用性**：同一个方法不是永远适用。例如“小数据 transformer 用 label smoothing”有明确条件；盲目注入会造成负迁移。
2. **失败模式与冲突**：两个方法都可能在历史上有效，但在同一条件下互相冲突。例如“full finetune”与“partial freeze”在小数据条件下可能给出相反建议。
3. **低频但关键经验保留**：普通向量检索和多数合并策略容易偏向高频经验，把低频但条件匹配的技巧吞掉。

因此，本项目的研究问题可以表述为：

> 如何把 MLEvolve 搜索轨迹中的 procedural experience 蒸馏成可检索、可验证、可冲突检查的 SOP 记忆，并用一种结构让 agent 在生成代码前主动选择合适方法？

---

## 3. 相关工作：别人做了什么

### 3.1 普通 RAG / flat memory

普通 RAG 或 flat memory 把经验当作独立文本片段，通过 BM25、embedding 或混合检索返回 top-k。它的优点是实现简单、泛用性强，但结构能力弱：

- 不知道方法之间的依赖顺序。
- 不知道两个经验是否冲突。
- 不知道经验适用条件是否匹配当前任务。
- 很难保护低频但关键的 SOP。

在 MLEvolve 场景中，flat memory 更像“把过去日志切块后搜索”，适合召回显式相似文本，但不适合做方法组合和风险判断。

### 3.2 Trace2Skill：把轨迹蒸馏成技能文档

Trace2Skill 的核心价值是把 agent 执行轨迹压缩成可复用 skill，通常经过 analyst 提取、合并、整理，最后形成类似 `SKILL.md` 的程序性说明。

它解决了“raw trace 太长、太杂”的问题，但存在两个明显不足：

- **粒度偏粗**：多个 SOP、条件、失败经验容易被合并进一个大文档，检索时很难精确命中单条方法。
- **冲突被合并或抹平**：合并阶段倾向产生一个一致文档，语义冲突、条件分支和低频经验容易丢失。

我们吸收 Trace2Skill 的思想，但不把最终记忆压成一个大 skill 文档，而是拆成更细粒度的 SOP / Condition / FailureMode / Evidence / Implementation 图节点。

### 3.3 MemP：程序性记忆生命周期

MemP 把 procedural memory 作为一等对象，系统比较 Build / Retrieve / Update 策略。它强调经验库不是一次性追加，而要随执行反馈更新、校验和修正。

它对本项目的启发是：程序性记忆应该有生命周期，且检索数量不是越多越好，需要控制上下文成本和噪声。

与 MemP 的差异是：

- MemP 更关注通用程序性记忆框架与更新策略。
- 我们更关注 MLEvolve 场景中 SOP 的结构化图表示、条件/失败/冲突边，以及双曲几何布局。

### 3.4 SKILL-DISCO：可执行技能与验证

SKILL-DISCO 把可复用技能定义为可跨轨迹匹配的 PFSM 子图，并进一步编译成可调用、可验证的代码技能。它强调“技能不只是自然语言经验，而应能被执行和验证”。

这对我们很重要：长期看，SOP 不能只停留在文字建议，应进一步连接到 implementation/reference 代码证据。

差异是：

- SKILL-DISCO 面向 FSM / web / embodied 场景，重点是可执行技能编译。
- 我们当前先做 MLE 方法论 SOP 的结构化地图，Reference / Implementation 作为叶子证据节点，后续再增强代码级复用。

### 3.5 SkillGraph：最接近的直接 baseline

SkillGraph 是当前最接近本项目的相关工作。它把技能组织成图：

- 节点：`{title, principle, condition, category}` compact skill card。
- 边：`prereq / enhance / co_occur`。
- 检索：seed select → backward BFS → forward beam → topological ordering。
- 演化：Insert / Merge / Split / Deprecate，以及边权 reinforce / decay / prune。

它的重要贡献是证明了“技能需要结构”，而不是孤立 top-k 文本。SkillGraph 的图感知检索在组合任务里非常关键。

但对我们来说，它仍有几个缺口：

1. **没有显式 FailureMode 和 Condition anchor**  
   它的 condition 是节点字段，不是图中可导航锚点。

2. **没有冲突建模**  
   它有正向关系边，但没有 `conflicts_with` / `prevents` 这样的风险结构。

3. **不是 agentic 检索**  
   SkillGraph 检索是程序化图遍历，检索器决定路径，不是 agent 主动看地图后选择方向。

4. **不是双曲几何结构**  
   SkillGraph 是普通图；层级通过 level 计算，但没有“半径=可靠性/通用性、角度=语义方向”的连续几何空间。

5. **依赖 RL co-evolution**  
   原论文完整系统和 policy 强绑定；我们的设计更偏即插即用的 external memory layer，不需要在线改模型权重。

### 3.6 HyperbolicRAG / HyperRAG：双曲检索线

双曲 RAG 系列工作的核心思想是：双曲空间适合表示层级。通常可以把：

- 半径看作抽象层级 / 深度。
- 角度看作语义方向。

这启发我们把 SOP 也放进双曲空间。但已有双曲 RAG 通常处理的是文档、实体、fact 或 passage，而不是程序性 SOP；它们也较少处理方法冲突、失败模式和代码生成 agent 的主动导航。

我们的差异是：用双曲结构承载 **procedural skill / SOP memory**，而不是声明性知识文档。

### 3.7 A-MEM / GAM / MemGPT：agentic memory 检索线

这些工作强调 agent 不应该只被动接收检索结果，而应在运行中主动决定何时检索、检索什么、如何更新记忆。

我们吸收这个方向，提出 `MemoryNavigator`：每次 draft / improve / debug / evolution 前，先让 navigator agent 看地图、选择方向、检查风险，再把 memory pack 给代码生成 agent。

与它们的差异是：

- 我们的 agentic 检索对象不是普通文本 memory，而是结构化 SOP 地图。
- 地图包含 Skill 区域、Condition、FailureMode、Evidence 和冲突边。
- 当前 V1 是子代理预检索模式；V2 才考虑让主代码 agent 在生成过程中直接 tool-call。

---

## 4. 我们做了什么：系统设计

本阶段系统分两层：

1. **离线 v0 双曲 SOP 图构建**  
   从现有 SkillGraph-C artifact 构建 `hyper_graph.json` 和 `hyper_index.npz`。

2. **在线 V1 agentic navigator runtime**  
   在 MLEvolve 每次生成代码前，让 MemoryNavigator 通过地图工具选择 SOP 并注入 prompt。

### 4.1 记忆单元：Skill / SOP / Reference 的分工

当前设计中三者关系如下：

- **Skill**：地图分区和粗路标。  
  例如 `general:universal_general`、`spooky-author-identification`、`leaf-classification`。Skill 的作用不是直接执行，而是帮助 agent 先定位区域。

- **SOP**：地图上的主要可执行记忆点。  
  例如“Use label smoothing for small-data transformer finetuning”。SOP 包含 action、condition、evidence 和 confidence，是最终注入给代码生成 agent 的核心内容。

- **Reference / Evidence / Implementation**：叶子证据和代码细节。  
  它们不参与主检索排序，只在 navigator 需要展开代码证据或实现细节时打开，避免大量 reference 污染 prompt。

换句话说：**双曲地图上主要放 SOP 点，Skill 是区域，Reference 是证据和实现附录。**

### 4.2 离线构建输入：SkillGraph-C

当前输入是：

`paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json`

这是一个 adapted SkillGraph-C artifact：

- 节点数：281
- 边数：1926
- 边类型：
  - `enhance`: 558
  - `co_occur`: 1001
  - `prereq`: 367

它不是原论文 faithful SkillGraph，而是我们为 MLEvolve 做过适配的 stronger baseline：

- 做了 general node normalization。
- 做了 selective general enhance。
- 加了 trace-order prereq，用执行顺序补充依赖。

因此，当前双曲图是基于 SkillGraph-C 的 **v0 bootstrap**，不是最终 transition-level SOP distillation 的产物。

### 4.3 compact-card 到 SOP-like node

SkillGraph-C 原节点是 compact card：

```json
{
  "title": "...",
  "principle": "...",
  "condition": "...",
  "category": "...",
  "scope": "...",
  "level": 0,
  "n_use": 3,
  "n_succ": 2,
  "p_hat": 0.667
}
```

构建器把它临时转成：

```json
{
  "type": "SOP",
  "title": "...",
  "action": "...",
  "applies_when": ["..."],
  "category": "...",
  "skill_id": "...",
  "metric": {
    "p_hat": 0.667,
    "n_use": 3,
    "n_succ": 2,
    "level": 0,
    "signal": "skillgraph_trace_proxy"
  },
  "confidence": "medium",
  "radius": 0.42,
  "angle_theta": "...",
  "angle_phi": "...",
  "poincare": [...],
  "lorentz": [...]
}
```

这里的 `action` 来自原 `principle`，`applies_when` 来自原 `condition`。

### 4.4 双曲坐标：角度和半径

当前 v0 坐标采用轻量、可复现方式：

#### 角度方向

角度由文本语义决定：

```text
title + principle + condition + category + scope
```

实现方式：

- TF-IDF
- TruncatedSVD 到 3 维
- L2 normalize 得到方向向量
- 再转成 spherical angles：`theta / phi`

含义：

- 语义相近的方法朝同一方向。
- 例如 transformer fine-tuning、CV ensemble、data leakage、OOM fix 会形成不同方向扇区。

#### 半径

半径由代理证据决定：

```text
radius = clamp(0.08 + 0.84 * (1 - core))

core = 0.65 * p_hat
     + 0.25 * support
     + 0.10 * (1 - level_norm)
     + general_bonus
```

含义：

- `p_hat` 高、`n_use` 多、层级更基础、更通用的 SOP 靠中心。
- 低频、低置信、条件更窄的 SOP 靠外圈。

这不是最终论文级 metric 半径。最终希望用 transition-level `metric_delta` 替代 `p_hat/n_use/level` 代理。

### 4.5 保留 SkillGraph-C base edges

构建时保留原图三类边：

- `co_occur`
- `enhance`
- `prereq`

这保证我们不是另起炉灶，而是在 SkillGraph-C 强 baseline 上加结构。

### 4.6 Graph Builder Agent 补边

离线构建器中有一个 `GraphBuilderAgent`。它当前是确定性 heuristic patch builder，不是 LLM agent，但结构上已经按 agentic patch protocol 设计：

1. 读取 SOP-like compact card。
2. 提出 graph patches。
3. 不直接改图。
4. 由程序 validator 检查后落图。

它补充四类节点：

- `Skill`
- `Condition`
- `FailureMode`
- `Evidence`

它补充五类主要边：

- `contains`: Skill 包含 SOP
- `applies_when`: SOP 适用于 Condition
- `prevents`: SOP 防止 FailureMode
- `supported_by`: SOP 由 Evidence 支撑
- `refines` / `conflicts_with`: SOP 之间的细化或冲突关系

FailureMode 当前来自规则抽取，例如：

- overfitting
- poor calibration
- data leakage
- out of memory
- timeout or slow execution
- training instability
- syntax or generated-code artifact
- api or checkpoint mismatch

### 4.7 patch validator

为了防止 agent / heuristic builder 乱连边，落图前做校验：

- node id 是否存在。
- edge kind 是否在白名单。
- patch 是否有 evidence 或文本依据。
- `conflicts_with` 必须有明确 opposition terms。
- `conflicts_with` 必须共享 condition 或 failure context。
- 全图 edge endpoint 必须存在。
- 每个 SOP 必须挂到 Skill / Condition / FailureMode / Evidence。

这一步是本系统和“LLM 自由改图”的重要区别：**agent 只能提 patch，程序决定是否写入图。**

---

## 5. 我们做了什么：运行时 Agentic Navigator

离线图只是地图，真正在线用时需要 navigator。当前 runtime 在：

`mlevolve/agents/memory/external_skill_memory.py`

### 5.1 接入方式

保留原 `ExternalSkillMemoryLayer` 入口，不破坏原有 `GlobalMemoryLayer`。两层分工：

- `GlobalMemoryLayer`：当前 run 内产生的短期经验。
- `ExternalSkillMemoryLayer`：预构建、跨 run 的长期 SOP 地图。

当开启 `agentic_hyperbolic` 或 `hyperbolic_agentic_memory` 时，检索逻辑走 MemoryNavigator。

### 5.2 暴露给 navigator 的地图工具

当前 runtime 提供这些 read-only map tools：

- `inspect_map(context)`：查看相关 Skill 区域、radius band、condition/failure hotspots。
- `navigate(region, condition, failure_mode, radius_band, top_k)`：按区域、条件、失败模式、半径层检索 SOP。
- `expand(node_id, edge_types, hops)`：沿边多跳扩展。
- `inspect_sop(sop_id)`：查看单个 SOP 的 action、条件、证据和邻居。
- `check_conflicts(sop_ids, context)`：判断 true conflict / condition branch / complementary / risk warning。
- `open_reference(ref_id, budget)`：打开证据或实现参考。

这和普通 top-k retriever 的差异在于：navigator 不是一次性拿结果，而是先获得地图感，再主动选择探索路径。

### 5.3 三轮导航限制

为了控制 token 和延迟，V1 navigator 最多 3 轮：

1. 必做 `inspect_map(context)`。
2. 再由 LLM 或 deterministic fallback 选择 `navigate / expand / inspect_sop / check_conflicts / open_reference / finish`。
3. 输出 memory pack。

如果没有 cfg 或 LLM navigator 不可用，会走 deterministic fallback：

```text
inspect_map -> navigate -> check_conflicts
```

这样即使没有在线 LLM 工具选择，也能稳定返回一个 agentic-style pack。

### 5.4 注入 prompt 的 memory pack

最终 prompt 标题为：

```text
## Agentic Hyperbolic Memory Navigation
```

包含：

- Navigator Trace
- Map Landmarks
- Selected SOPs
- Conflict / Risk Warnings
- Rejected SOPs
- Reference-backed Implementation Hints

重要约束写在 prompt 中：

- selected SOP 只有 WHEN 条件匹配时才采用。
- risk warning 是约束，不是直接执行命令。
- 不要盲目照抄 warning-only SOP。

这让 adoption 分析更清楚：不仅知道注入了哪些 SOP，也知道 navigator 为什么选择它们。

---

## 6. 当前实现结果

当前已生成三个 artifact：

- `paper-skills/hyper_memory/hyper_graph.json`
- `paper-skills/hyper_memory/hyper_index.npz`
- `paper-skills/hyper_memory/graph_builder_report.json`

### 6.1 图规模

从 SkillGraph-C 输入：

| 项 | 数量 |
| --- | ---: |
| source compact-card nodes | 281 |
| source edges | 1926 |
| source edge kinds | `enhance=558`, `co_occur=1001`, `prereq=367` |

构建后：

| 节点类型 | 数量 |
| --- | ---: |
| SOP | 281 |
| Skill | 6 |
| Condition | 281 |
| FailureMode | 16 |
| Evidence | 281 |
| 总节点 | 865 |

| 边类型 | 数量 |
| --- | ---: |
| enhance | 558 |
| co_occur | 1001 |
| prereq | 367 |
| contains | 281 |
| applies_when | 281 |
| prevents | 411 |
| supported_by | 281 |
| refines | 309 |
| conflicts_with | 91 |
| 总边 | 3580 |

### 6.2 坐标结果

`hyper_index.npz` 包含：

- `node_ids`
- `poincare`
- `lorentz`
- `flat_twin`
- `radius`
- `theta`
- `phi`
- `angle`
- `direction`

坐标自检：

- Poincare max norm: 0.8675，小于 1，合法落在球内。
- Lorentz hyperboloid max error: 约 1.14e-5，可接受。
- radius 分布：
  - core: 122
  - middle: 49
  - edge: 110

这说明当前地图既保留了一批中心高置信 SOP，也保留了相当数量边缘低频/窄条件 SOP。

### 6.3 patch 构建与验证结果

Graph Builder Agent：

- proposed patches: 2633
- applied patches: 2238
- skipped patches: 395
- rejected patches: 0

skipped 的主要原因是 duplicate node id，例如多个 SOP 共享同一个 FailureMode 节点，这是正常去重，不是失败。

验证全部通过：

- source SOP id preserved
- node id unique
- all edge endpoints exist
- all edge kinds allowed
- patch edges have reason or evidence
- every SOP has Skill region
- every SOP has Condition edge
- every SOP has Failure edge
- every SOP has Evidence edge
- conflicts have reason and evidence
- conflicts have opposition terms
- conflicts have condition or failure context

---

## 7. 与 SkillGraph 的核心差异

| 维度 | SkillGraph | 本工作 |
| --- | --- | --- |
| 记忆点 | compact skill card | SOP as executable procedural memory point |
| 分区 | general / task-specific category | Skill region 作为地图区域 |
| 条件 | 节点字段 condition | Condition 节点 + `applies_when` 边 |
| 失败模式 | 无显式 FailureMode | FailureMode 节点 + `prevents` 边 |
| 冲突 | 无 `conflicts_with` | 显式冲突边，要求 opposition + context |
| 证据 | `p_hat/n_use` 等节点字段 | Evidence 节点 + `supported_by` 边 |
| 检索 | 程序化 BFS/beam/topo | MemoryNavigator agent 主动看地图并选择工具 |
| 几何 | 普通图 + level | Poincare / Lorentz 坐标，半径与角向分解 |
| 运行方式 | 与 RL co-evolution 结合 | external memory prompt injection，不改 policy 权重 |
| 当前状态 | 文献方法 / 我们有 adapted baseline | 已实现 v0 图构建 + V1 runtime navigator |

最关键的差异不是“我们也有图”，而是：

> SkillGraph 解决的是 skill 依赖与有序检索；本工作进一步把 skill 拆成 SOP / Condition / FailureMode / Evidence，并让 agent 在这张结构地图上主动导航和检查冲突。

---

## 8. 与 Trace2Skill / SKILL-DISCO / MemP 的核心差异

### 8.1 与 Trace2Skill

Trace2Skill 重在从轨迹中总结 skill，但常见输出是线性文档。我们的差异是：

- 不把经验压成单个大文档。
- 保留细粒度 SOP。
- 保留条件、失败模式、证据和冲突关系。
- 为图检索和双曲坐标准备结构化字段。

### 8.2 与 SKILL-DISCO

SKILL-DISCO 重在把可复用轨迹结构编译成可执行技能。我们的差异是：

- 当前不直接编译成可执行函数，而是构建 ML 方法论 SOP 地图。
- 更强调条件化检索、失败模式和冲突检查。
- 后续 Implementation 节点可以吸收 SKILL-DISCO 的验证思想，把 SOP 连接到可执行代码模板。

### 8.3 与 MemP

MemP 强调 procedural memory 的 build / retrieve / update 生命周期。我们的差异是：

- MemP 主要比较不同构建、检索、更新策略。
- 我们聚焦图结构和几何结构，把 procedural memory 做成可导航地图。
- 当前图只读，在线不写入；后续可以借鉴 MemP 做执行后更新。

---

## 9. 为什么需要双曲结构

双曲结构的直觉是：它适合表达层级和树状扩张。程序性方法天然有这种结构：

- 中心：通用、稳定、高支持的方法，例如数据流检查、训练集内 fit、smoke test。
- 中层：任务相关但常用的方法，例如某类模型训练策略。
- 外圈：低频、条件窄、但可能在特定失败模式下关键的方法。

普通向量空间常见问题是：低频方法容易被高频语义相近方法挤掉。双曲空间有指数体积，外圈可以容纳大量细粒度条件分支，让低频 SOP 仍可达。

在本系统中：

- `theta / phi` 表示方法方向。
- `radius` 表示可靠性、支持度和通用性。
- `Skill` 区域帮助粗路由。
- `Condition / FailureMode` 作为导航锚点。
- `conflicts_with` 作为风险边。

当前 v0 仍未完全发挥双曲距离检索的能力；已经生成 Poincare/Lorentz 坐标，但 runtime scoring 仍以 lexical / feature / graph scoring 为主。下一阶段需要把 `navigate` 的核心评分换成真实 Poincare 或 Lorentz distance，并加入 Flat-Twin 对照。

---

## 10. 学术贡献可以怎样表述

较稳妥的贡献表述如下：

1. **提出一种面向 MLE agent 的 Agentic Hyperbolic SOP Memory 设计**  
   将 procedural skill memory 表示为 SOP-centered hyperbolic map，其中 Skill 作为区域、Condition/FailureMode 作为导航锚点、Evidence/Implementation 作为叶子支撑。

2. **把 SkillGraph compact skill card 扩展为多类型 SOP graph**  
   在保留 `co_occur/enhance/prereq` 的基础上，增加 `applies_when/prevents/refines/conflicts_with/supported_by` 等方法论关系，显式表达条件、失败和冲突。

3. **实现一个 patch-based Graph Builder Agent 协议**  
   Graph Builder 只提出 patch，不直接写图；程序验证 node id、edge kind、evidence 和 conflict guard 后落图，降低 LLM/agent 乱改结构的风险。

4. **实现在线 V1 MemoryNavigator runtime**  
   在 MLEvolve 生成代码前，navigator 通过 `inspect_map/navigate/expand/check_conflicts/open_reference` 等工具主动探索 SOP map，并输出 memory pack，而不是被动接收一次性 top-k。

5. **完成 v0 artifact 与可验证构建结果**  
   当前从 281 个 SkillGraph-C compact nodes 构建出 865 节点、3580 边的 hyperbolic SOP memory，并通过结构验证。

需要避免的夸大表述：

- 不应说“已经证明双曲结构有效”。目前只有结构构建与 runtime 接入，还没有完整 A/B 实验。
- 不应说“半径由真实 metric_delta 监督”。当前是 `p_hat/n_use/level` 代理。
- 不应说“Graph Builder Agent 已经是 LLM 自主建图”。当前是 deterministic heuristic patch builder，接口为未来 LLM builder 预留。
- 不应说“runtime 已经用双曲距离作为主检索”。当前坐标已生成，但 runtime navigate 仍主要是 lexical/feature/graph scoring。

---

## 11. 当前系统的边界与风险

### 11.1 v0 bootstrap，不是最终蒸馏系统

当前 hyper graph 来源于 SkillGraph-C compact cards。它可以作为可运行原型，但不是最终论文级 memory source。最终需要从 transition-level trace 中蒸馏 SOP，直接得到：

- `metric_delta`
- `evidence_ids`
- `implementation_ids`
- parent→child improvement evidence
- condition-specific support

### 11.2 FailureMode 和 conflict 仍是启发式

当前 FailureMode 来自关键词规则，`conflicts_with` 来自 opposition terms + same condition/failure context。它可解释、可复现，但召回和精度都需要评测。

后续可升级为：

- LLM Graph Builder Agent 提 patch。
- validator 继续做硬约束。
- 人工 gold set / heldout run 做 conflict precision/recall。

### 11.3 几何有效性尚未证明

当前已构建 Poincare/Lorentz 坐标，但没有完成：

- Hyperbolic distance retrieval。
- Flat-Twin Agentic Navigator 对照。
- Hyperbolic vs Euclidean 在 Rare Recall、Condition Precision、Conflict False Deletion 等指标上的比较。

因此现在能说“已建成 v0 双曲结构记忆”，不能说“已证明双曲比平面更好”。

### 11.4 数据清洁和 ground truth 风险

项目历史上存在泄露 run 和 contaminated KB。任何论文级实验都必须：

- 使用 clean run allowlist。
- 对 gold SOP 做 INDEX_BUG / leakage audit。
- 区分 pilot result 与 paper-grade result。

---

## 12. 下一步实验计划

### 12.1 Offline retrieval evaluation

比较四组：

1. Flat retrieval
2. SkillGraph-C
3. Hyperbolic automatic top-k
4. Agentic Hyperbolic Navigator

指标：

- Rare Recall@5
- Condition Precision
- Conflict False Deletion Rate
- Category Coverage
- Redundancy Rate
- Evidence Coverage
- Navigation Efficiency

其中必须包含 Flat-Twin：

> 同样 agentic 工具、同样 SOP，只把双曲距离换成欧氏/feature score。如果 Hyperbolic 只赢非 agentic baseline，但不赢 Flat-Twin，就不能声称双曲几何本身有效。

### 12.2 Runtime A/B pilot

在线保留当前 GlobalMemory，不混淆变量。比较：

- SkillGraph-C external memory
- Agentic Hyperbolic memory

指标：

- best metric
- steps to threshold
- bug rate
- repeated-error rate
- rare-SOP adoption
- risk-warning adoption
- token/time cost

### 12.3 Graph Builder Agent 升级

把当前 heuristic builder 升级成 LLM-backed Graph Builder Agent：

- 输入 SOP cards + local neighborhoods。
- 输出 patch JSON。
- validator 负责 hard constraints。
- report 记录 proposed / rejected / applied patches。

这一步可以让系统更 agentic，但不能放弃程序验证。

### 12.4 transition-level SOP distillation

从真正的 parent→child transition 抽取 SOP：

```json
{
  "type": "SOP",
  "title": "...",
  "action": "...",
  "applies_when": ["..."],
  "prevents": ["..."],
  "metric_delta": -0.012,
  "confidence": "medium",
  "evidence_ids": ["run/branch/parent->child"],
  "implementation_ids": ["impl_x"]
}
```

这会替代当前 `p_hat/n_use/level` 代理信号，让半径真正由 metric evidence 控制。

---

## 13. 可汇报时的主线

建议汇报按这个顺序讲：

1. **问题**：MLEvolve 经验不能只做文本检索，需要条件、失败、冲突和低频经验保留。
2. **别人怎么做**：Trace2Skill 会蒸馏但线性；SkillGraph 有图但缺失败/冲突/双曲/agentic；HyperbolicRAG 有几何但载体是文档；MemP 有生命周期但不是结构地图。
3. **我们的想法**：把 SOP 放到双曲地图上，Skill 是区域，Condition/FailureMode 是导航锚点，Evidence/Reference 是叶子。
4. **当前实现**：从 SkillGraph-C 281 nodes 构建成 865 nodes / 3580 edges 的 hyperbolic SOP graph，生成 Poincare/Lorentz 坐标和验证报告。
5. **runtime**：MemoryNavigator 在代码生成前主动 inspect/navigate/check_conflicts，输出 memory pack。
6. **边界**：这是 v0 bootstrap，不是最终 transition-level metric_delta 系统；几何有效性还需 Flat-Twin 对照证明。
7. **下一步**：离线检索评测 + 在线 A/B + LLM Graph Builder Agent + transition-level distillation。

---

## 14. 自查清单

我按“能不能经得住师兄追问”的标准检查了这份报告：

- **有没有把当前 v0 说成最终系统？**  
  没有。报告多次说明当前是 SkillGraph-C bootstrap，半径是 proxy，不是真实 `metric_delta`。

- **有没有把 heuristic GraphBuilderAgent 说成 LLM agent？**  
  没有。明确说当前是 deterministic heuristic patch builder，只是采用 patch-agent 协议，未来可替换成 LLM builder。

- **有没有声称双曲结构已经实验有效？**  
  没有。明确说只完成构建和接入，几何有效性需要 Hyperbolic vs Flat-Twin 对照。

- **有没有讲清楚我做的和别人做的差异？**  
  有。分别对比了 flat RAG、Trace2Skill、MemP、SKILL-DISCO、SkillGraph、HyperbolicRAG、agentic memory。

- **有没有可复核数据？**  
  有。报告列出了输入源图、输出图规模、坐标数组、patch 和 validation 结果。

- **有没有下一步可执行计划？**  
  有。包括 offline retrieval evaluation、online A/B、LLM Graph Builder Agent、transition-level distillation。

---

## 15. 结论

本阶段工作的核心价值不是“又做了一个检索器”，而是把 MLEvolve 的长期程序性记忆从文本条目推进到 **可导航、可验证、可冲突检查的 SOP 方法地图**。

在相关工作中，SkillGraph 证明了 skill 需要图结构，Trace2Skill 证明了轨迹可以蒸馏成技能，HyperbolicRAG 证明了双曲空间适合层级检索，MemP 证明了 procedural memory 需要生命周期管理。我们的系统尝试把这些线索合并到 MLE 场景中：

> 用 SOP 做记忆载体，用 Skill/Condition/FailureMode/Evidence 建图，用双曲坐标提供地图结构，用 MemoryNavigator 做 agentic 多跳检索，用 validator 保证补边安全。

当前已经完成 v0 构建与 runtime V1 接入；下一步的关键不是继续堆功能，而是用 Flat-Twin 和 SkillGraph-C 做严格对照，证明双曲结构和 agentic navigation 分别带来了什么增益。
