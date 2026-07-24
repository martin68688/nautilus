# Decision Admissibility 完整实施与实验计划书

> 版本：v1.1（Result / Adoption / Causal Writeback 修订）  
> 初版日期：2026-07-19  
> 修订日期：2026-07-21  
> 用途：新 Codex/开发窗口的唯一实施交接文档  
> 工作目录：`/Users/haoming/Downloads/nautilus`  
> 当前分支：`codex/dual-time-procedural-memory`  
> 原始 baseline HEAD：`422f9051a47be02cfd627766de0b9a3266b88362`  
> 2026-07-21 只读审查 HEAD：`b47dab63b7861f3ea0871094d6dd07b77e6b81a4`（当前 worktree 有大量未提交用户修改，不得把该 HEAD 当成完整实现快照）  
> 基线测试：215 passed in 40.59s  
> 基线测试日期：2026-07-19

---

## 0. 新窗口如何使用这份文档

### 0.0 强制交接前置：先 checkpoint，再新开任务

不得在当前长对话中直接开始 WP0–WP8。必须先完成下列交接顺序：

1. 在当前窗口盘点当前代码、测试、计划文档和 dirty worktree。
2. 只将当前可复现的代码、必要测试/fixture、轻量实验 manifest/report 和本计划书加入 checkpoint；排除密钥、运行日志、原始大数据、临时缓存、PPT/TIFF 和可再生成大产物。
3. 运行基线测试和 secret scan；确认 checkpoint 不包含本地凭据。
4. 创建一个明确的 baseline commit，commit message 建议为：

   ```text
   chore: checkpoint before decision admissibility implementation
   ```

5. 将该 commit push 到当前远程分支，并验证远程 branch HEAD 与本地 checkpoint commit 一致。
6. 记录：`baseline_branch`、`baseline_commit`、`remote_branch`、`push_verified_at`、基线测试结果。
7. 新开一个独立 Codex 任务/窗口（这里的“新节点”指新的 Codex 工作任务，不是 Kubernetes GPU 节点）。
8. 新任务必须从已 push 的 `baseline_commit` 开始，完整读取本计划书，再进入 WP0。

本次用户授权的 commit/push 仅是“实施计划前的 baseline checkpoint”。新任务后续的阶段性 commit/push 仍需按用户后续指令执行，不得自动 push。

### 0.1 可直接复制给新窗口的启动提示词

```text
请在 /Users/haoming/Downloads/nautilus 中实施：
/Users/haoming/Downloads/nautilus/coordination/decision_admissibility_complete_execution_plan_20260719.md

这是一个从已 commit/push baseline 开始的新任务。先完整读取计划书和 AGENTS.md，核对当前分支/HEAD 是否等于交接记录中的 baseline_commit，再检查 dirty worktree、现有接口和基线测试。
按计划的 Work Package 顺序执行，不得跳过验收门，不得覆盖用户未跟踪文件，不得将旧历史分数因静态审查而升级为可排名证据。
必须先阅读第 0.4 节的 v1.1 修订：不得把 Result Fact、Adoption Edge 和 Causal Edge 合并；不得重复实现已经存在的 PROMOTE_RESULT 拆分。进入 WP5 前先做 delta audit，只实现尚未接线的 edge publisher、fixed-holdout terminal finalizer 和 Positive Result/Adopted SOP 拆分。
优先完成 WP0→WP4 的本地正确性闭环，在 shadow parity 通过前不开启全局 enforce，在用户明确授权前不创建 Kubernetes Job、不 push、不修改论文头条结论。
每完成一个 WP，报告：修改文件、新增接口、测试命令/结果、已知风险、是否通过该 WP 的 stop gate。
```

### 0.2 新窗口的第一个回合必须做什么

1. 读取 `AGENTS.md` 与本文档全文。
2. 执行：

   ```bash
   git branch --show-current
   git rev-parse HEAD
   git status --short
   ```

3. 保存 dirty-worktree manifest，不删除、移动或提交与本任务无关的用户文件。
4. 如果从原始 baseline commit 启动，运行第 20.1 节基线测试，预期 215 passed；如果从后续 WP checkpoint 启动，先读取对应 WP 报告和 dirty-worktree manifest，再运行 full suite 与第 0.4 节相关 targeted suite，不得机械地把“不是 215”视为回归。
5. 检查本文档中标注的当前缺口是否已被其他窗口修改；已完成的不重复实现。

### 0.3 实施权限边界

- 本文档授权的是代码、测试、本地离线 artifact 和说明文档修改。
- 不自动授权提交、push、创建 PR、提交集群 Job、开启长时间 GPU 实验或发布论文结论。
- 需要集群只读盘点时，优先使用已有 Pod；需要新 Pod/Job 时必须另行获得用户授权。
- 不得把 API key、token 或 `.env` 内容写入命令行、manifest、日志或计划书。

### 0.4 v1.1 修订摘要：不要再混淆“节点入库”和“经验采用”

本次修订确认，写回至少包含三个不同的权限对象：

```text
PROMOTE_RESULT
  当前实际运行节点作为独立 result fact 入库
  不声称任何注入历史经验影响了它

PUBLISH_ADOPTION
  发布 historical experience → current artifact 的采用边
  需要 static_actuation + runtime_actuation

PUBLISH_CAUSAL
  发布 historical experience → current artifact 的因果边
  在 Adoption 证据之外再需要 counterfactual_actuation
```

2026-07-21 只读代码审查确认当前源代码已经部分完成该拆分：

- `Operation` 已有 `PROMOTE_RESULT / PUBLISH_ADOPTION / PUBLISH_CAUSAL`；
- `ProtocolCompiler` 不再为 `PROMOTE_RESULT` 强制要求 actuation；
- 正常 positive-memory gate 已调用 `PROMOTE_RESULT`；
- Session Overlay 的 result event 使用 `publication_class=result_fact`，并固定 `derived_from_refs=[]`；
- 相关 Authority 测试 40/40 通过。

仍未完成、必须由本计划补齐的接线：

1. 生产路径中尚未找到 `PUBLISH_ADOPTION / PUBLISH_CAUSAL` 的正式 edge publisher；
2. fixed-holdout 模式在 result parse 阶段主动跳过 positive writeback，但尚未找到 sealed terminal scoring 完成后的唯一最终写回器；
3. Positive SOP distillation 仍需区分“从当前成功节点产生的新 SOP”和“宣称历史经验被当前节点验证”；前者不应要求历史经验 actuation，后者必须要求；
4. legacy `PROMOTE` 仅可读取旧 ledger/config，不得再作为新结果入库的生产调用。

后续窗口不得重新把这三条权限合并成一个 `PROMOTE`，也不得因为旧 shadow 报告出现 `missing_static_actuation/missing_runtime_actuation` 就认定当前独立 result 被正确拒绝；必须先检查该记录使用的是 legacy `PROMOTE` 还是 `PROMOTE_RESULT`。

---

## 1. 一句话目标与论文主线

### 1.1 一句话目标

> 在递归式 MLE Agent 中，只让“粒度适合当前决策阶段，且具体 Claim 在当前 Protocol 下有权影响当前 Operation”的历史经验进入排序和 Prompt；本次实际运行节点只要具有合法执行、数据、评分和协议 Receipt，就可以作为独立 Result Fact 写回；只有当系统进一步声称该节点采用了某条历史经验、由该经验造成，或将其权限继承给后代 SOP 时，才要求 AST、运行和反事实 Actuation Receipt。

### 1.2 不再使用的三个并列贡献叙事

不将下列三项独立包装为三个 novelty：

- Dynamic Hybrid；
- Protocol Repair；
- Provenance-gated Promotion。

它们是同一条 Decision Admissibility pipeline 中的三个职责：

1. `Stage/Granularity Gate`：决定当前应该看 SOP 还是 RunForest，看 L1/L2/L3 中的哪一粒度。
2. `Claim/Authority Gate`：决定某条内容是否有资格影响当前操作。
3. `Actuation/Writeback Gate`：分别决定当前 Result Fact 能否独立写回、历史经验采用边能否发布、因果边能否发布，以及哪些权限能继承到后代记忆。

### 1.3 真正的研究单元

研究单元不是整条 memory item 的 `valid=true/false`，而是：

```text
(Claim, Operation, Generation Stage, Governance Stage, Protocol Version, Task Context)
```

这使得同一条混合价值经验可以同时满足：

- OOF 索引修复可用于 Debug/Repair Seed；
- 泄漏警告可用于 Inspect/Protocol Repair；
- 0.92 污染分数不得用于 Rank/Select/Promote；
- 未证明被采纳的历史经验不得作为后代成功知识；
- 但一个未采用任何历史经验、由 cold-start 独立产生且协议合法的新节点，仍可作为自己的 Result Fact 入库。

---

## 2. 成功标准、范围与非目标

### 2.1 工程成功标准

1. 原始运行、RunForest、SOP Clause、Claim、Receipt、AuthorityDecision 可以逐级双向追溯。
2. SOP 在 clause level 执行可见性，被拒绝内容在候选排序和 Prompt 之前就被排除。
3. `attached_sop_ids`、projection、cache、legacy GlobalMemory 不得绕过 Authority。
4. 原始污染历史永不被原地改成 clean；Clean Replay 只新建证据路径。
5. 当前运行使用只读 Base Bundle + append-only Session Overlay，运行后离线发布新版 Bundle。
6. 从 shadow 到 enforce 有差异报告、false allow/deny 审计和明确回滚开关。
7. Result Fact、Adoption Edge 和 Causal Edge 分别授权、分别落盘；任何一种缺失不得伪造另外两种。
8. fixed-holdout 只允许 sealed terminal scorer 完成后进行一次幂等 `PROMOTE_RESULT` 写回；不得提前写，也不得永远漏写。

### 2.2 实验成功标准

1. 证明错误粒度经验在真实决策点中会改变 Agent 动作/代码，而不只是检索标注更准。
2. 证明 Claim-specific Authority 在同等安全性下比 Global Validity Bit 保留更多合法知识。
3. 证明未授权 Claim 不会经 Summary→SOP→Merged SOP→Code Template 洗白。
4. 证明 memory exposure 不等于 adoption，并报告 L2 Static、L3 Runtime 和 L4 Counterfactual 证据。
5. 完成 seed-heldout 与 task-heldout 两类数据隔离实验。

### 2.3 非目标

- 不以“首次提出分阶段记忆路由”作为主 novelty。
- 不将 RunForest 的父子 Transition 未经反事实执行就称为 causal transition。
- 不以双曲几何超越欧氏作为主结论。
- 不追求对任意 ML 任务的零配置通用泄漏检测。通用性来自 ProtocolSpec 配置和稳定 Authority Kernel。
- 不用旧回顾性分数直接宣称新系统提高了 MLE 任务成绩。

---

## 3. 当前代码与资产基线

### 3.1 当前已实现

| 系统 | 当前代码 | 已有能力 |
|---|---|---|
| Authority substrate | `mlevolve/authority/` | Claim/Operation/Protocol/Receipt/Decision，Registry，Compiler，EvidenceGraph，Replay Certifier，Derivation Guard |
| MLEvolve adapters | `mlevolve/authority/adapters/mlevolve/` | ranking/selection/promotion/replay 接线，主要仍在 shadow |
| Dynamic Hybrid | `mlevolve/agents/memory/stage_aware_hybrid_memory.py` | Draft/Improve/Debug 路由、SOP/Tree 融合、Debug abstention |
| Global Memory | `mlevolve/agents/memory/global_memory.py` | 保存/检索节点和审计记录，现有 enforce 检查不完整 |
| RunForest builder | `paper-skills/hyper_memory/build_run_forest_memory.py` | 保存 Run/RunNode/Transition/Evidence/FailurePattern/SOP 图 |
| SOP distillation | `paper-skills/distillation/` | DeepSeek 从 branch T+/T- 生成 SOP，带 source branches/evidence turns |
| Protocol policy diagnostic | `paper-skills/eval_composite_memory/` | 混合价值、Dual View 和 DeepSeek prompt exposure 机制探针 |

### 3.2 当前已验证基线

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_causal_granularity_benchmark_v2.py \
  tests/test_protocol_repair.py \
  tests/test_run_forest_memory.py
```

2026-07-19 结果：

```text
215 passed in 40.59s
```

### 3.3 当前必须修复的缺口

1. `ClaimType` 没有 `DEBUG_REPAIR`、`METHOD_HYPOTHESIS`、`AUDIT_FINDING`。
2. Dynamic Hybrid 与 Authority 的 stage ontology 不一致。
3. 当前 node adapter 主要自动产生 SCORE Claim，混合 Claim 拆分不完整。
4. `receipt_bridge.py` 很多 Receipt 依然是从聚合 audit 翻译，不是独立宿主采集。
5. `GlobalMemory` enforce 主要检查 decision ref 存在与 protocol string 相同，未完整检查 outcome/scope/policy version。
6. `StageAwareHybridMemoryLayer._build_sop_reverse_index()` 会消费所有 `distills_to`，未严格执行 edge authority outcome。
7. `attached_sop_ids` 被 Debug projection 直接消费，可绕过边级权限。
8. 当前 SOP 不是 clause-level 权限对象。
9. 当前 281 条 SOP 中，`protocol_ref/claim_refs/receipt_refs/operation_scope` 覆盖均为 0。
10. 当前 2,773 条 `distills_to` 边全部是 quarantine，原因为缺 runtime/counterfactual actuation，但检索层仍可消费。
11. 现有 `run_forest_graph.json` 是旧语料/旧 schema artifact，不是最近原版 MLEvolve 非 Spooky 运行构建的新包。
12. 不存在 Base Bundle + Session Overlay + sleep-time publication 的完整版本化闭环。
13. 2026-07-21 已完成 `PROMOTE_RESULT/PUBLISH_ADOPTION/PUBLISH_CAUSAL` 的权限语义拆分，但只有 Result Fact 有生产写回器；Adoption/Causal edge 仍停留在模型、compiler 和测试层。
14. fixed-holdout 在 pre-score 阶段正确禁止 positive write，但尚缺 sealed terminal scoring 后的幂等 Result Fact finalizer。
15. 当前 `DISTILL_POSITIVE` 仍把“本次节点自身的新 SOP”和“历史经验被采用/验证后的 SOP”混为一条 actuation-gated 路径，可能在 SOP 层重复 cold-start false denial。
16. GlobalMemory 只保存 code summary，Overlay 保存 code hash/pointer；必须验证每个 Result Fact 都能解析回 Journal/RunForest 的完整真实代码，不能把摘要误当事实主体。

### 3.4 当前数据边界

最近原版 MLEvolve 运行的逻辑来源：

```text
PVC: haoming-storage
relative root: mlevolve-original-runs/be034ec/3h120
source implementation: third_party/MLEvolve @ be034ec
```

最近盘点的预期快照（执行时必须重新生成 manifest 验证）：

| 项目 | 预期数量 |
|---|---:|
| run 目录 | 90 |
| 完整 `journal.json` | 79 |
| 非 Spooky 任务 | 16 |
| RunNode | 1,656 |
| 代码节点 | 1,577 |
| metric 节点 | 589 |
| 不完整目录 | 11 |

不得将这些数字写死为“必须匹配否则删数据”；正确行为是生成 drift report，人工理解新增/缺失后再冻结 corpus version。

---

## 4. 目标系统总体架构

### 4.1 离线记忆生产环

```text
Original MLEvolve Runs (immutable)
  → Corpus Inventory + SHA256
  → Audit Sidecars
  → Legacy Claim Decomposition
  → Full RunForest
  → DeepSeek SOP-Clause Distillation
  → Clause Binder + Lineage Guard
  → SOP Visibility Materialization
  → raw-audited Bundle
  → selected Clean Replay
  → certified Bundle
```

### 4.2 在线决策环

```text
DecisionContext
  → StageOntology
  → granularity/source routing
  → Claim/Operation/Protocol Authority
  → pre-ranking Visibility Mask
  → SOP/RunForest scoring + Dynamic Hybrid
  → prompt pack + ExperienceContracts
  → Agent action/code
  → CodeExecution + Protocol Receipts
  → PROMOTE_RESULT → Result Fact event
  → optional Static/Runtime Actuation
  → optional PUBLISH_ADOPTION → Adoption Edge
  → optional Counterfactual Actuation
  → optional PUBLISH_CAUSAL → Causal Edge
  → append-only Session Overlay（各对象分别落盘）
```

### 4.3 递归写回环

```text
Base Bundle vN (read-only)
    +
Session Overlay (append-only)
  → run completes
  → fixed-holdout terminal finalizer closes Result Facts
  → sleep-time audit
  → claim decomposition
  → separate Result/Adoption/Causal lineage materialization
  → diagnostic/candidate/positive-result/positive-adopted distillation
  → lineage/authority validation
  → build Bundle vN+1 in staging
  → validate hashes and invariants
  → atomically publish CURRENT pointer
```

### 4.4 四层记忆的职责

| 层 | 存储 | 作用 | 不能替代的东西 |
|---|---|---|---|
| Working Memory | 当前 branch/prompt/code/error | 一次决策的当前状态 | 不是长期证据 |
| SOP Procedural Memory | 细粒度 SOPClause | 方法/战术/修复导航 | 不自动证明分数合法 |
| RunForest Execution Memory | RunNode/Transition/Evidence/Failure | replay/debug/因果追溯 | 不自动获得高风险使用权 |
| Authority Memory | Claim/Receipt/Protocol/Decision/Derivation | 判断谁能影响什么 | 不代替语义检索 |

---

## 5. 不可违反的系统不变量

### I1. 事实与派生分离

- RunForest parent/child/execution/error 是事实层，不由 LLM 重写。
- SOP 是派生层，任何 clause 必须有 source Claim/Artifact refs。

### I2. 派生不升权

```text
Scope(child) ⊆ intersection(required parent scopes)
```

摘要、翻译、合并、改写、多次引用和多个污染来源互相“佐证”都不能扩权。

### I3. 可见不等于可采用

- `INSPECT` 可以显示全量并带警告。
- `RANK/SELECT/PROMOTE_RESULT/PUBLISH_ADOPTION/PUBLISH_CAUSAL/CODE_SEED` 必须分别有该 Operation 的 ALLOW scope。
- `PROMOTE_RESULT` 的 ALLOW 不自动产生 `PUBLISH_ADOPTION` 或 `PUBLISH_CAUSAL` 权限。

### I4. 过滤先于排序和 Prompt

被拒绝 clause 不得影响 ANN 候选竞争、RRF、projection、token budget 和 Prompt。

### I5. 旧历史不原地洗白

- 旧 Claim 保留当时 protocol/audit/outcome。
- Clean Replay 生成新 Run、新 Receipt 和新 support path。
- 如方法改变，生成 Successor Claim。

### I6. Agent 不能给自己签 Receipt

LLM/Agent 可以提出 Claim 和解释，但 split/evaluator/execution/method identity/runtime 等 Receipt 必须由宿主受信 collector 产生。

### I7. Exposure 不等于 Adoption

记忆出现在 Prompt 中只是 L0；论文中的 adoption 至少需要 L2/L3，causal adoption 需要 L4。

### I7-A. Result Existence 不等于 Experience Adoption

- `CODE_EXECUTION + protocol-legal score` 可以证明本次节点存在并成功，不证明历史经验影响了它。
- `static_actuation + runtime_actuation` 证明的是历史经验合同在本次代码和运行中被落实。
- `counterfactual_actuation` 才支持“去掉/替换该经验会改变动作、代码或结果”的因果 Claim。
- 一个没有任何 exposure/adoption 的 clean cold-start 节点可以 `PROMOTE_RESULT`，但 `derived_from_refs` 必须为空。

### I7-B. 不得用 Result Event 偷渡派生边

- Result event 固定 `publication_class=result_fact`。
- Result event 的 `derived_from_refs=[]`；即使存在 L3/L4 report，也只能记录为尚未发布的 report reference。
- Adoption/Causal 必须分别经过 `PUBLISH_ADOPTION/PUBLISH_CAUSAL` 并写为独立、可撤销、可审计的 edge event。

### I8. Bundle 不可变

已发布 Bundle 的 manifest/hash/index/lineage 不得修改；修正必须发布新版本。

### I9. 测试语料不得进入它的记忆包

Seed-heldout 的 held-out seed 和 Task-heldout 的 held-out task 必须在 raw journal、SOP、index、audit-derived text 和 score metadata 中全部隔离。

---

## 6. 核心数据模型与公开接口

### 6.1 StageOntology

将现有单一 `DecisionStage` 拆为两个正交轴：

```python
class GenerationStage(str, Enum):
    DRAFT = "draft"
    MODEL_DESIGN = "model_design"
    IMPROVE = "improve"
    DEBUG = "debug"
    EVOLUTION = "evolution"
    FUSION = "fusion"

class GovernanceStage(str, Enum):
    RETRIEVAL = "retrieval"
    BRANCH_SELECTION = "branch_selection"
    MEMORY_WRITEBACK = "memory_writeback"
    DISTILLATION = "distillation"
    REPLAY = "replay"
```

兼容规则：

- 保留旧 `DecisionStage` 一个迁移周期；
- `stage_ontology.py` 实现唯一显式映射；
- 新 `AuthorityRequest` 同时记录 generation/governance stage；
- 序列化时保留 legacy stage 以读取旧 ledger，但新逻辑不再依赖它。

### 6.2 ClaimType

在现有类型上新增：

```python
METHOD_HYPOTHESIS = "method_hypothesis"
DEBUG_REPAIR = "debug_repair"
AUDIT_FINDING = "audit_finding"
```

定义：

- `EXECUTED`：代码路径被执行，不表示分数有效。
- `SCORE`：某 protocol 下报告一个指标。
- `METHOD_HYPOTHESIS`：方法/结构/特征/训练战术候选，不自动表示 superiority。
- `DEBUG_REPAIR`：可定位的修复动作，不等于 causal attribution。
- `AUDIT_FINDING`：泄漏、evaluator、selection、protocol drift 等审计结论。
- `PAIRWISE_SUPERIORITY`：严格对照下 A 优于 B。
- `CAUSAL_ATTRIBUTION`：有反事实证据支持经验导致动作/结果变化。
- `GENERALIZATION`：至少多任务族证据支持。

### 6.3 Operation

现有 Operation 保留，新增或显式区分：

```python
PROMOTE_RESULT = "promote_result"
PUBLISH_ADOPTION = "publish_adoption"
PUBLISH_CAUSAL = "publish_causal"
DISTILL_DIAGNOSTIC = "distill_diagnostic"
DISTILL_CANDIDATE = "distill_candidate"
DISTILL_POSITIVE_RESULT = "distill_positive_result"
DISTILL_POSITIVE_ADOPTED = "distill_positive_adopted"
```

迁移期内：

- `PROMOTE_RESULT` 发布当前节点自身的 protocol-legal Result Fact，不需要证明其采用历史经验；
- `PUBLISH_ADOPTION` 发布历史经验到当前节点的采用边，至少需要 L2/L3；
- `PUBLISH_CAUSAL` 发布因果边，至少需要 L4；如声称效果则需要 L5；
- legacy `PROMOTE` 只为旧 ledger/config 读取兼容保留；任何新生产调用必须静态检查失败；
- legacy `DISTILL` 默认 fail-closed，不得在未区分 result/adopted 语义时自动映射为 positive；
- Diagnostic distillation 只产生警告/调试可见条款，不支持成绩或晋升；
- Candidate distillation 可产生方法候选，仅允许 Generate/Inspect/Replay Seed，不声称已证明有效；
- `DISTILL_POSITIVE_RESULT` 从本次实际运行节点产生新的正向 SOP Clause，需要节点执行、协议、评分和 Derivation Receipt，但不要求历史经验 actuation；
- `DISTILL_POSITIVE_ADOPTED` 声称某条历史经验已被当前节点采用/验证，需要对应 `PUBLISH_ADOPTION`，如声称造成改进还需要 `PUBLISH_CAUSAL`；
- 两类 positive distillation 生成不同 Claim/lineage path，不得只用一个布尔字段区分。

### 6.4 CorpusManifestV1

```python
CorpusManifestV1(
    corpus_id: str,
    created_at: str,
    source_repo: str,
    source_commit: str,
    source_root: str,
    exclusion_rules: list[dict],
    runs: list[CorpusRunEntry],
    expected_snapshot: dict,
    actual_snapshot: dict,
    split_manifests: list[str],
    manifest_sha256: str,
)

CorpusRunEntry(
    run_id: str,
    task_id: str,
    canonical_task_id: str,
    task_family: str,
    seed: str,
    status: str,  # complete | partial | invalid_json | excluded
    journal_path: str,
    config_path: str,
    filtered_journal_path: str | None,
    best_solution_path: str | None,
    artifact_hashes: dict[str, str],
    node_count: int,
    code_node_count: int,
    metric_node_count: int,
)
```

### 6.5 AuditSidecarV1

```python
AuditSidecarV1(
    artifact_id: str,
    run_id: str,
    node_id: str,
    code_sha256: str,
    detector_schema: str,
    detector_version: str,
    active_protocol_ref: str,
    status: str,
    issues: list[dict],
    legacy_receipt_level: str,
    generated_at: str,
)
```

审查 sidecar 不修改原 journal，不伪装为原运行时 Receipt。

### 6.6 SOPClauseV1

```python
SOPClauseV1(
    clause_id: str,
    sop_id: str,
    text: str,
    retrieval_text: str,
    claim_refs: tuple[str, ...],
    source_artifact_refs: tuple[str, ...],
    source_transition_refs: tuple[str, ...],
    protocol_scope: tuple[str, ...],
    task_scope: dict,
    permitted_operations: tuple[str, ...],
    permitted_generation_stages: tuple[str, ...],
    permitted_governance_stages: tuple[str, ...],
    publication_class: str,  # diagnostic | candidate | certified
    authority_decision_refs: tuple[str, ...],
    receipt_refs: tuple[str, ...],
    derivation_refs: tuple[str, ...],
    protocol_agnostic: bool,
    legacy_status: str,
)
```

`retrieval_text` 与显示文本分离，防止不允许参与排名的分数/警告 token 影响高风险检索。

### 6.7 VisibilityRequest / VisibleSOPPack

```python
VisibilityRequest(
    operation: Operation,
    generation_stage: GenerationStage,
    governance_stage: GovernanceStage,
    active_protocol: ProtocolRef,
    task_context: TaskContext,
    memory_bundle_version: str,
    token_budget: int,
    requesting_component: str,
)

VisibleSOPPack(
    request_id: str,
    visible_positive_clauses: list[SOPClauseV1],
    visible_diagnostic_clauses: list[SOPClauseV1],
    warning_clauses: list[SOPClauseV1],
    suppressed_clause_refs: list[str],
    authority_decision_refs: list[str],
    visibility_trace: dict,
)
```

`suppressed_clause_refs` 仅存于宿主 trace，不进 Prompt。

### 6.8 MemoryBundleManifestV1

```python
MemoryBundleManifestV1(
    bundle_id: str,
    bundle_version: str,
    parent_bundle: str | None,
    corpus_manifest_hash: str,
    protocol_registry_hash: str,
    authority_policy_version: str,
    detector_version: str,
    deepseek_model: str,
    deepseek_prompt_hash: str,
    graph_hashes: dict,
    index_hashes: dict,
    lineage_hash: str,
    split_id: str,
    certification_level: str,
    build_report: str,
)
```

---

## 7. Protocol Registry 与受信 Receipt 设计

### 7.1 Rules as Data

Authority Kernel 不写死任务语义。以下内容放入 ProtocolSpec：

- task objective/unit of analysis；
- split family/group/time key；
- forbidden overlap；
- preprocessing fit scope；
- evaluator 与 metric direction；
- selection freeze；
- seeds 和 aggregation；
- holdout 使用次数；
- protocol repair surface；
- pairwise superiority 所需证据。

### 7.2 最少三类正式 ProtocolSpec

1. `random-classification-v1`：分层随机分割，macro-F1/log-loss，train-fold-only preprocessing。
2. `grouped-classification-v1`：group-level split，group 不重叠，macro-F1。
3. `chronological-regression-v1`：时间顺序分割，禁止 future→past，RMSE minimize。

如 Tier 2 包含 ranking，再增加 `grouped-ranking-v1`（query/group split，NDCG maximize）。

### 7.3 受信 collectors

| Collector | 证明什么 | 关键 payload |
|---|---|---|
| CodeExecutionCollector | 目标代码真实运行 | exit status, executed path, run hash |
| SplitLineageCollector | train/valid/test/group/time 边界 | sample/group/time lineage, overlap |
| FitScopeCollector | scaler/vectorizer/feature fit 范围 | fit sample IDs, fold IDs |
| EvaluatorIntegrityCollector | evaluator 未被篡改 | evaluator hash, inputs, metric direction |
| SelectionFreezeCollector | 候选/超参在 final 前冻结 | candidate set hash, timestamp |
| SeedAggregationCollector | 没有 best-seed cherry-pick | declared seeds, all results, aggregation |
| MethodIdentityCollector | replay 是否保持方法 | AST/method fingerprint, protected surface |
| StaticActuationCollector | 代码是否满足经验合同 | MustPreserve/Change/NotUse checks |
| RuntimeActuationCollector | 目标路径是否执行 | runtime events/observations |
| CounterfactualCollector | memory-on/off 是否改变行为 | paired action/code/outcome delta |
| DerivationCollector | Summary/SOP 是否保留 Claim/scope | parent-child clause mapping |

### 7.4 信任边界

- Collector 运行在 Agent 进程外或宿主控制路径中。
- Agent 只能提交候选 artifact，不能自己填 `verified=true`。
- Receipt 使用 collector/version/protocol/payload hash 生成稳定 ID。
- 事件链使用 `parent_event_hash/event_hash`，检测删除或重排。

---

## 8. 原版 MLEvolve 语料与长期记忆打包

### 8.1 新增工具目录

```text
paper-skills/memory_bundle/
├── build_corpus_manifest.py
├── audit_corpus.py
├── build_split_manifests.py
├── build_memory_bundle.py
├── validate_memory_bundle.py
├── schema.py
└── README.md
```

### 8.2 完整运行的定义

一个 run 进入正式 corpus 需要：

1. `logs/journal.json` 存在且 JSON 可解析；
2. journal 至少包含一个 node；
3. `logs/config.yaml` 存在，可提取 task/seed/source config；
4. canonical task ID 不是 Spooky；
5. 没有因 secret-safety 问题被明确标为 aborted source。

`best_solution.py` 为空不单独导致 run 被排除；它会作为 artifact completeness warning。

### 8.3 盘点命令接口

```bash
python paper-skills/memory_bundle/build_corpus_manifest.py \
  --runs-root "$PVC_MOUNT/mlevolve-original-runs/be034ec/3h120" \
  --source-commit be034ec \
  --exclude-task spooky-author-identification \
  --output "$OUT/corpus_manifest.json" \
  --report "$OUT/corpus_inventory_report.json"
```

脚本必须是只读的，不得改 journal、config 或 solution。

### 8.4 离线审查

```bash
python paper-skills/memory_bundle/audit_corpus.py \
  --manifest "$OUT/corpus_manifest.json" \
  --protocol-registry mlevolve/config/protocols \
  --default-protocol mlevolve-default@1 \
  --output-dir "$OUT/audit_sidecars" \
  --report "$OUT/audit_report.json"
```

每个 code node 产生独立 sidecar。不得将“现在 v5 静态审查干净”写成“原运行有完整 Runtime Receipt”。

### 8.5 三类 bundle

1. `memory-full-v1`：使用所有完整非 Spooky 运行，仅用于未来新运行，不得用于同一批历史回顾成绩。
2. `memory-seed-heldout-v1`：同任务不同 seed 隔离，证明同任务后续改进。
3. `memory-task-heldout-v1`：完整任务隔离，证明跨任务泛化。

### 8.6 包目录

```text
mlevolve-memory-be034ec-nonspooky-v1/
├── manifest.json
├── raw_journals/
├── audit_sidecars/
├── runforest/
│   ├── graph.json
│   └── index.npz
├── sop/
│   ├── clauses.jsonl
│   ├── containers.json
│   ├── graph.json
│   └── taxonomy.json
├── authority/
│   ├── claims.jsonl
│   ├── receipts.jsonl
│   ├── decisions.jsonl
│   ├── derivations.jsonl
│   └── replay_receipts.jsonl
├── visibility/
│   ├── clause_metadata.jsonl
│   └── precompiled_masks/
├── splits/
│   ├── full.json
│   ├── seed-heldout.json
│   └── task-heldout.json
└── reports/
```

### 8.7 打包产物

- 目录版本，供 Agent 运行时直接 mmap/load；
- `tar.zst` 便携归档；
- `SHA256SUMS`；
- `build_report.json`；
- `lineage_completeness_report.json`；
- `visibility_coverage_report.json`；
- `split_leakage_report.json`。

---

## 9. RunForest 重建计划

### 9.1 修改入口

将 `build_run_forest_memory.py` 从“扫描 runs dir + 旧 allowlist”改为：

```bash
python paper-skills/hyper_memory/build_run_forest_memory.py \
  --corpus-manifest "$OUT/corpus_manifest.json" \
  --audit-dir "$OUT/audit_sidecars" \
  --sop-clauses "$OUT/sop/clauses.jsonl" \
  --bundle-id mlevolve-be034ec-nonspooky-v1 \
  --out-dir "$OUT/runforest"
```

保留 `--runs-dir` 一个迁移周期，但只能输出 `legacy_uncertified` bundle，不得标记 paper-grade。

### 9.2 RunForest 保存范围

- 所有完整 run 的所有 RunNode；
- 父子边、branch、step、stage、metric、error、code hash；
- 所有审查 issue，包括 blocked/warning/unavailable；
- 所有 FailurePattern；
- 所有可解析 Transition；
- Claim/Receipt/Decision/Protocol refs；
- 原始 source artifact hash。

不因污染删除 RunNode。安全性由消费权限控制，不由删除事实实现。

### 9.3 边语义

使用显式边类型：

- `parent_of`：事实树边；
- `has_transition/transition_to`：改动容器；
- `supported_by`：证据边；
- `has_failure_pattern`：审计/失败边；
- `navigation_attached_to`：可用于 Inspect/Debug 导航；
- `authorized_distills_to`：在指定 scope 下可用于程序性驱动；
- legacy `distills_to`：只保留兼容，新检索器不得将其默认视为授权边。

### 9.4 索引

分开存储：

- execution index：RunNode/Transition/Failure/Evidence；
- clause index：每个 SOPClause 一个向量/索引行；
- visibility metadata：以 clause ID 为 key；
- precompiled masks：常用 protocol × operation × stage；
- session overlay mask：在线计算，默认高风险 fail-closed。

---

## 10. SOP 拆分、DeepSeek 蒸馏与合并

### 10.1 Branch 提取

修改 `paper-skills/distillation/extract_branches.py`：

- 删除硬编码 `CLEAN` 列表；
- 读取 `CorpusManifestV1` 和 split manifest；
- 每个 turn 使用全局稳定 ref：`run_id/node_id/transition_id`；
- 附 audit status/issue refs，但不要把 blocked score 改写为 success；
- 输出 `trace_manifest.json`，记录所有输入 hash。

### 10.2 DeepSeek 的职责

DeepSeek 负责：

- 拆分小而单一的 SOP clause；
- 建议 Claim statement/type；
- 生成 action/applies_when/prevents；
- 区分 method/debug/audit/score 内容；
- 引用输入中存在的 evidence refs。

DeepSeek 不负责：

- 签发 Receipt；
- 自己决定 ALLOW/DENY；
- 宣称 protocol-agnostic；
- 删除污染来源警告；
- 把 score correlation 升级成 causal attribution。

### 10.3 DeepSeek 输出 schema

```json
{
  "sop_containers": [
    {
      "title": "...",
      "clauses": [
        {
          "text": "...",
          "claim_type_proposal": "debug_repair",
          "source_refs": ["run::...::transition::..."],
          "evidence_refs": ["run::...::node::..."],
          "applies_when": ["..."],
          "prevents": ["..."],
          "publication_class_proposal": "diagnostic"
        }
      ]
    }
  ]
}
```

### 10.4 Deterministic Binder

DeepSeek 输出后必须通过 binder：

1. 所有 source/evidence refs 真实存在；
2. Claim type 与 source facts 不矛盾；
3. score 文本不能进 method/debug clause 的 `retrieval_text`；
4. publication class 由 Authority Kernel 重新编译；
5. 将 parent Claim scope 与 requested scope 做 non-escalation validation；
6. 失败时进 quarantine，不静默丢弃。

### 10.5 可复现性

- 使用 `DEEPSEEK_MODEL` 与 MLEvolve 保持一致；
- temperature 设为 0 或 API 支持的最低值；
- 以 model + system prompt + user prompt + input hashes 作为 cache key；
- 保存原始 response、usage、retry 和 parse report；
- 后续重建默认使用冻结 cache，除非显式发布新 distiller version。

### 10.6 SOP 合并

- 合并容器，不合并授权事实；
- 每个 clause 保留独立 support path；
- 不对权限取并集来放行混合条款；
- 同一文本如有一条合法 support path，网关可仅依赖该路径，但不得伪称其他污染路径也合法。

---

## 11. P4-B SOP Visibility Gateway（SOP 可见性网关）完整修改计划

### 11.1 核心定义

SOP Visibility Gateway 是一个在检索之前执行的 clause-level authorization layer。它不是：

- 整条 SOP 的 global valid bit；
- 只在 Prompt 中显示 warning；
- 检索后删除文本；
- 让 LLM 自己忽略禁止内容。

### 11.2 固定数据流

```text
DecisionContext
→ normalize Generation/Governance Stage
→ construct VisibilityRequest
→ enumerate candidate clause metadata (not text ranking)
→ compile/evaluate AuthorityRequest per clause
→ build allowed clause ID bitmap
→ slice embedding/index rows using bitmap
→ semantic/geometry score
→ RRF/projection/rerank
→ token-budget pack
→ render only visible clauses
→ emit host-only visibility trace
```

### 11.3 严禁的数据流

```text
retrieve all
→ rank all
→ concatenate all
→ ask LLM to obey tags
→ post-filter output
```

这种后置过滤只能作为 baseline，不能作为 Full system。

### 11.4 分阶段可见性矩阵

| 决策 | 可见内容 | 禁止影响 |
|---|---|---|
| Draft | 当前 task/protocol 允许的 L1 method/candidate/certified clause | L3 报错噪声、污染分数 |
| Model Design | 获准 L2 结构/特征/loss/training clause | 不相容任务族的实现细节 |
| Improve | L2 clause + 获准改进 Transition | 泄漏成绩与不同 protocol score |
| Debug | DEBUG_REPAIR、AUDIT_FINDING、FailurePattern、repair Transition | 污染 SCORE 对修复候选排名 |
| Inspect | 所有条款，显示完整警告/来源 | “可看”不得自动变成“可采用” |
| Rank/Select | 当前 Protocol 下有完整 Receipt 的 SCORE/PAIRWISE | v2 score、test leakage、best-seed-only |
| Repair Seed | 方法冻结且允许修协议的 clause | 换模型/特征/loss 后伪称 protocol-only |
| Promote Result | 当前节点有执行、数据、evaluator、selection 和 active Protocol 的合法 Receipt | 因未采用历史经验而拒绝独立 clean result |
| Publish Adoption | source experience 已暴露，ExperienceContract 满足 L2 Static + L3 Runtime | 只 exposed、只 Agent 声称采纳、或仅代码相似 |
| Publish Causal | 已有 Adoption + L4 memory-on/off counterfactual | 仅 activation/correlation 就声称因果 |
| Distill Diagnostic | 有来源的审计/失败/修复条款 | 成绩结论和 positive recommendation |
| Distill Candidate | 静态合法的 method hypothesis，显示 provisional | superiority/promote 含义 |
| Distill Positive Result | 当前节点自身的合法执行/协议/评分/derivation 证据 | 强迫 cold-start result 提交历史经验 actuation |
| Distill Positive Adopted | 已发布 Adoption；如声称效果还需 Causal/Efficacy | 用当前节点成功反向洗白所有 exposed 经验 |
| Code Seed | 明确允许复用且代码契约完整 | 混合 SOP 中未授权代码片段 |

### 11.5 Mixed-value 标准案例

同一 SOP 包含：

```text
C1: 使用 sample_id 对齐 OOF 预测
C2: 历史运行使用 test labels 选模
C3: 历史分数 0.92
```

Debug View：

```text
ALLOW C1
ALLOW_WITH_WARNING C2
SUPPRESS C3 as positive evidence
```

Rank View：

```text
DENY C3
C1/C2 不作为成绩候选
```

Inspect View：

```text
SHOW C1/C2/C3 with audit status
NO adoption permission implied
```

### 11.6 预编译与在线 mask

- Base Bundle 对固定 protocol/policy 预编译常用 operation-stage mask。
- Task-specific 约束在请求时与预编译 mask 取交集。
- Session Overlay 新 clause 在线评估，高风险默认不可见。
- 如索引库不支持 metadata mask，对高风险视图建立独立 clause shard，不使用全量 post-filter。

### 11.7 必须关闭的旁路

1. `_build_sop_reverse_index()` 不得无条件消费 `distills_to`。
2. `_causal_attachment_rows()` 不得直接使用 `attached_sop_ids`。
3. `_tree_sop_projection()` 只能投影当前 VisibilityRequest 允许的 clause。
4. Prompt formatter 不得从原 SOP container 重新读取已 suppress 文本。
5. GlobalMemory 不得仅因为存在 decision ref 就放行。
6. Cache key 必须包含 protocol hash、operation、stage、task scope、bundle version、policy version。

### 11.8 Fail-closed 规则

- 高风险请求缺 claim/protocol/decision/scope 任一项，默认 suppress。
- 旧 SOP 可在 Inspect/Debug 以 `legacy_uncertified` 显示。
- `protocol_agnostic=true` 需要明确证明，不能用缺 tag 推断。
- 无可见条款时返回 empty/abstain，不放宽权限。
- Authority Engine 不可用 LLM 输出作为最终 ALLOW。

### 11.9 新增/修改文件

```text
mlevolve/agents/memory/sop_visibility_gateway.py          [new]
mlevolve/authority/adapters/mlevolve/retrieval_gate.py   [new]
mlevolve/authority/stage_ontology.py                      [new]
mlevolve/agents/memory/stage_aware_hybrid_memory.py       [modify]
mlevolve/agents/memory/global_memory.py                   [modify]
mlevolve/authority/models.py                              [modify]
paper-skills/hyper_memory/build_run_forest_memory.py      [modify]
```

### 11.10 P4-B 验收条件

1. 混合 SOP 在 Debug 保留 OOF repair 与泄漏警告，但不将 0.92 当作有效分数。
2. 在 Rank 中，0.92 不存在于 embedding candidate、RRF trace、Prompt 或 token count。
3. 删除/拒绝边后，`attached_sop_ids`/projection/cache 无法复活条款。
4. Unauthorized Prompt Exposure = 0。
5. Unauthorized Activation = 0。
6. Oracle-allowed Debug/Repair knowledge retention = 100% 于 deterministic mixed-value suite。
7. 旧 281 SOP shadow migration 输出 visible/suppressed/legacy/empty 覆盖报告。
8. 报告 latency、token、empty-pack 开销，不在 pilot 前人为设有利阈值。

---

## 12. Dynamic Hybrid 与 Visibility Gateway 的集成

### 12.1 先准入，再融合

Dynamic Hybrid 不再直接面对全量 SOP/Tree。正确顺序：

```text
Stage route
→ SOP clause visibility
→ RunForest claim visibility
→ rank each legal source
→ Dynamic Hybrid fusion
→ prompt pack
```

### 12.2 默认路由

| Stage | SOP | RunForest | 失败回退 |
|---|---|---|---|
| Draft | L1 method/candidate positive | 仅必要的获准整体路径 | SOP-only |
| Model Design | L2 tactic | 相似方法实现 Transition | SOP-only |
| Improve | L2 tactic | 已获准改进 Transition | SOP-only |
| Debug | Diagnostic/repair clause | 失败→修复 Transition/FailurePattern | 低 confidence 时 Diagnostic SOP-only |
| Fusion/Evolution | 组合原则 | 获准候选路径 | 收紧到 SOP |

### 12.3 不改变的对照性

为了论文因果对照，Full system 不修改：

- Agent backbone；
- token/GPU budget；
- candidate generation 主体；
- evaluator；
- 非记忆 prompt；
- 训练时间限额。

只修改“什么经验可以以什么粒度进入当前决策”。

---

## 13. Experience Contract 与 Runtime Actuation

### 13.1 采纳等级

| Level | 名称 | 需要什么 |
|---:|---|---|
| L0 | EXPOSED | 经验进入 Prompt |
| L1 | CLAIMED_ADOPTION | Agent 声称使用 |
| L2 | STATIC_CONFORMANT | AST/code diff 满足 Contract |
| L3 | RUNTIME_CONFORMANT | 目标路径真实执行 |
| L4 | CAUSAL_CONFIRMED | memory-on/off 配对重放显示动作/代码改变 |
| L5 | EFFECTIVE | 上述变化带来 protocol-legal outcome |

### 13.2 ExperienceContract 生成

获准 clause 编译为：

```python
ExperienceContract(
    preconditions=[...],
    must_preserve=[...],
    must_change=[...],
    must_not_use=[...],
    expected_runtime_observations=[...],
)
```

例如 OOF 修复：

```text
MustPreserve: model family, features, loss, search budget
MustChange: OOF index alignment by sample_id
MustNotUse: test labels, holdout metric for selection
ExpectedRuntimeObservations: one prediction per training sample; no duplicate/missing IDs
```

### 13.3 两种反事实

1. Influence counterfactual：移除经验后 Agent action/code 是否变化。
2. Efficacy counterfactual：采用/不采用修改后的 protocol-legal outcome 是否变化。

前者证明影响，后者才支持效果。

### 13.4 写回所需最低等级

| 写回操作 | 默认最低证据 |
|---|---|
| Diagnostic SOP | source lineage + audit finding |
| Candidate SOP | EXECUTED + static-clean method claim，显示 provisional |
| Promote Result Fact | 当前节点 `CODE_EXECUTION` + 与 Claim 相符的 split/fit/prediction/evaluator/selection/protocol Receipts；**不要求历史经验 actuation** |
| Publish Adoption Edge | source ExperienceContract + L2 Static + L3 Runtime；只发布 `source_claim/clause → target_artifact` 边 |
| Publish Causal Edge | 已发布 Adoption + L4 Counterfactual；如声称 outcome 改善则需要 L5 |
| Positive Result SOP | 当前节点 Result Fact + Derivation Receipt + clause-level protocol scope；**不要求历史经验 actuation** |
| Positive Adopted SOP | 已发布 Adoption；如文案声称“该历史经验导致改进”则需要 Causal/Efficacy |
| Code Seed | 明确 CODE_SEED authority + L2/L3 + no forbidden dependency |
| Causal Claim | L4 |
| Effective Repair Claim | L5 |

### 13.5 四种 Receipt 不得互相替代

```text
CODE_EXECUTION
  证明本次训练代码运行过

STATIC_ACTUATION
  证明注入历史经验的 MustPreserve/MustChange/MustNotUse
  体现在本次生成代码中

RUNTIME_ACTUATION
  证明上述历史经验对应的目标路径在本次运行中真实触发

COUNTERFACTUAL_ACTUATION
  证明移除/替换该历史经验后，动作、代码或合法 outcome 会改变
```

不得用 `runtime_actuation` 代替普通 `code_execution`，也不得因为本次代码成功运行就自动签发历史经验的 `runtime_actuation`。

---

## 14. Clean Replay（干净重放）

### 14.1 用途

Clean Replay 不是重跑所有历史，而是为有方法价值但协议证据不足/不合法的候选建立新证据路径。

### 14.2 候选选择

每个 task 最多 3 个：

- 有完整 source/parent/child/code；
- 静态审查未发现方法自身致命泄漏；
- 存在可识别 method hypothesis；
- 历史 metric 改善仅用于排定 replay 优先级，不作为论文正结果；
- 选择方法族尽量不重复；
- 选择规则写入 `replay_queue.jsonl`，不人工挑最好结果。

### 14.3 方法冻结

冻结：

- model family/constructor/signature；
- feature logic；
- loss/objective；
- hyperparameter search space；
- compute/training budget；
- inference/ensemble family。

允许修改的 repair surface 来自 ProtocolSpec，例如：

- split API；
- fold-local preprocessing scope；
- evaluator 输入/方向；
- seed aggregation；
- holdout access path；
- protocol logging/instrumentation。

### 14.4 重放结果

```text
METHOD_PRESERVED + receipts satisfied
  → new support path for replay Claim

SUCCESSOR_METHOD
  → new successor artifact/Claim
  → old Claim remains restricted

REQUIRE_HUMAN_REVIEW
  → quarantine replay result
```

### 14.5 不可做的事

- 不修改旧 Claim 的原 protocol_ref；
- 不删除旧 audit failure；
- 不把方法变更后的高分当成旧方法合法化证据；
- 不只比较自然语言声明的“我没换方法”。

---

## 15. Base Bundle、Session Overlay 与版本化写回

### 15.0 写回对象与存储职责

| 对象 | 主体 | 存储位置 | 是否包含完整代码 | 是否需要历史经验 Actuation |
|---|---|---|---:|---:|
| Raw Run / RunNode | 本次真实执行节点 | immutable Journal / RunForest | 是 | 否 |
| Result Fact | 本次节点的执行、score、protocol Claim | Session Overlay / Authority Memory | code hash + artifact pointer；不重复完整代码 | 否 |
| GlobalMemory Record | 本次节点的检索摘要 | GlobalMemory | code summary，不是完整代码 | 否 |
| Adoption Edge | 历史 Claim/Clause → 本次节点 | 独立 Overlay edge event | 否 | L2 + L3 |
| Causal Edge | 历史 Claim/Clause → 本次节点 | 独立 Overlay edge event | 否 | L4；效果需 L5 |
| SOP Clause | 从本次节点蒸馏的程序知识 | staged/certified Bundle | 自然语言/结构化 clause + source pointer | 取决于它是 Result SOP 还是 Adopted SOP |

完整训练代码的事实源是 Journal/RunForest。GlobalMemory、Overlay 和 SOP 不必复制完整代码，但必须通过 `artifact_id + code_sha256 + corpus/bundle manifest` 可解析回不可变的原始节点。无法解析的 Result Fact 不得发布到 certified Bundle。

### 15.1 Runtime 加载

```python
MemorySnapshot(
    base_bundle_id: str,
    base_bundle_path: str,
    session_overlay_path: str,
    active_protocol_ref: str,
    authority_policy_version: str,
)
```

Base Bundle 只读；Session Overlay append-only。

### 15.2 在线检索合并

- Base 使用预编译 visibility mask。
- Overlay 使用当前 Authority Engine 在线判定。
- 合并后再执行 Dynamic Hybrid RRF，但两边事先都已经准入过滤。
- Overlay 中未审查 score 只能 Inspect，不得 Rank/Promote。

### 15.2-A 三条在线写回路径

#### A. Result Fact

```text
terminal node
→ trusted execution/protocol/evaluator receipts
→ Authority(PROMOTE_RESULT)
→ append memory_claim(publication_class=result_fact, derived_from_refs=[])
```

该路径允许没有任何历史经验 exposure 的 clean cold-start 节点入库。

#### B. Adoption Edge

```text
source ExperienceContract + target artifact
→ static_actuation
→ runtime_actuation
→ Authority(PUBLISH_ADOPTION)
→ append memory_derivation_edge(kind=adoption)
```

edge payload 至少包含：`source_claim_ref/source_clause_ref`、`target_artifact_id`、`contract_hash`、static/runtime receipt refs、decision ref、protocol/policy version 和 edge hash。

#### C. Causal Edge

```text
authorized Adoption Edge
→ memory-on/off counterfactual
→ Authority(PUBLISH_CAUSAL)
→ append memory_derivation_edge(kind=causal)
```

不得根据 result event 中存在 `verified_adoption_report_refs` 自动生成 Adoption/Causal Edge；report 是候选证据，edge 只有在对应 Operation 被 ALLOW 后才存在。

### 15.2-B Fixed-holdout 终局写回

fixed-holdout 模式必须保留“评分前不写 positive memory”的安全边界，同时增加唯一的后置 finalizer：

```text
candidate execution completes
→ pre-score: no positive writeback
→ sealed terminal scorer computes trusted metric
→ host closes split/fit/prediction/evaluator/selection receipts
→ select final artifact(s) under declared policy
→ Authority(PROMOTE_RESULT)
→ idempotent Result Fact append
→ optional Adoption/Causal edge publication
```

实现要求：

- finalizer 只能由宿主 terminal scorer 调用，Agent 不可调用；
- idempotency key 至少绑定 `run_id + artifact_id + protocol_hash + publication_class`；
- crash/restart 重放不得产生重复 Result Fact 或重复 edge；
- failed、timeout、missing-score、protocol-blocked 节点不得写 positive Result Fact，但原始 RunNode 仍进入 RunForest；
- 如果 terminal scorer 不存在或 Receipt closure 失败，必须产出显式 `writeback_incomplete` 状态，而不是静默漏写；
- 任何测试或生产路径不得在 sealed score 产生前调用 `PROMOTE_RESULT`。

### 15.3 Sleep-time 发布

1. 冻结 session journal 与 runtime receipts。
2. 生成 overlay manifest/hash。
3. 分别消费 Result Fact、Adoption Edge 和 Causal Edge，运行审查、Claim decomposition、SOP distillation；缺 edge 时不得凭文本相似度补造 lineage。
4. 运行 derivation/visibility/bundle validator。
5. 在 staging 目录构建 vN+1。
6. 全部验收通过后，原子更新 `CURRENT.json`。
7. 保留 vN 用于回滚和实验复现。

### 15.4 并发与崩溃安全

- Bundle publisher 使用 file lock；
- staging 不能被 runtime 加载；
- validator 失败不改 `CURRENT.json`；
- ledger 和 overlay 使用 append + fsync/atomic rename；
- 发布报告包含 parent bundle 和全部 hash。

---

## 16. 分阶段实施工作包

## WP0：冻结基线与资产边界

### 输入

- 已 push 且经远程验证的 baseline commit/branch 交接记录；
- 当前代码分支/HEAD；
- dirty worktree；
- 现有 215-test baseline；
- 旧 RunForest/SOP artifacts。

### 动作

- 生成 `coordination/decision_admissibility_baseline_20260719.json`；
- 记录分支、HEAD、Python/dependency 版本、测试命令和结果；
- 仅记录 untracked 文件路径，不打包密钥/大数据；
- 不覆盖旧 graph/index。

### 交付

- baseline manifest；
- 当前缺口对照表；
- 旧 artifact hash report。

### Stop gate

- 新任务 HEAD 与交接的远程 baseline commit 一致；
- 215-test baseline 可重现；
- 所有用户未跟踪资产未被修改。

## WP1：Authority 语义正确性

### 代码

- 扩展 ClaimType/Operation；
- 新建 StageOntology；
- 修复 GlobalMemory decision outcome/scope/policy checks；
- 将高风险异常处理为 fail-closed；
- 保留 shadow 兼容。

### 测试

```text
tests/authority/test_stage_ontology.py
tests/authority/test_claim_types.py
tests/authority/test_global_memory_authority_scope.py
tests/authority/test_high_risk_fail_closed.py
```

### Stop gate

- legacy 测试不回归；
- 每个 runtime stage 有唯一双轴映射；
- 伪 decision ref/错 protocol/错 operation 不能通过 enforce。

## WP2：Claim decomposition 与 trusted collectors

### 代码

```text
mlevolve/authority/claim_decomposer.py
mlevolve/authority/collectors/
mlevolve/authority/adapters/mlevolve/transition_adapter.py
```

### 逻辑

- deterministic fact extraction first；
- LLM 仅提议 statement/boundary；
- 稳定 Claim ID；
- source/evidence binder；
- 宿主 Receipt 采集；
- 旧记录标记 `legacy_static_only`。

### 测试

```text
tests/authority/test_claim_decomposition.py
tests/authority/test_mixed_value_authority.py
tests/authority/test_trusted_collectors.py
tests/authority/test_receipt_trust_boundary.py
```

### Stop gate

- 混合 RunNode 至少拆成合法 repair/audit claim 与受限 score claim；
- Agent 伪造 verified payload 不能生成 trusted Receipt；
- 不会因为 code executed 就放行 score superiority。

## WP3：P4-B SOP Visibility Gateway

### 代码

- 新增 gateway/retrieval gate/types；
- 将 clause mask 接到 `_rank_sops`、projection、debug attachment、RRF 和 formatter；
- 分离 navigation/authorized projection；
- 关闭 `attached_sop_ids` 旁路；
- 增加 visibility trace。

### 测试

```text
tests/authority/test_sop_visibility_gateway.py
tests/authority/test_mixed_value_sop_visibility.py
tests/authority/test_visibility_pre_prompt.py
tests/authority/test_visibility_projection_bypass.py
tests/authority/test_legacy_sop_visibility.py
```

### Stop gate

- 第 11.10 节全部通过；
- 当前 2,773 quarantine edges 在 high-risk enforce view 中不可消费；
- Inspect/Debug 仍能导航并显示 warning。

## WP4：Corpus、RunForest 和 SOP-Clause Bundle

### 代码

- 实现 `paper-skills/memory_bundle/`；
- 重构 branch extractor/DeepSeek distiller/binder/merger；
- 重构 RunForest builder 读 manifest/sidecars/clauses；
- 产生 full/seed/task 三套 split-aware bundle。

### 测试

```text
tests/test_corpus_manifest.py
tests/test_corpus_split_isolation.py
tests/test_run_forest_bundle_v2.py
tests/test_sop_clause_distillation_schema.py
tests/test_memory_bundle_validation.py
```

### Stop gate

- Spooky=0；
- 所有正式 run hash 完整；
- 每个 code node 有 sidecar；
- 每个 clause 来源可解析；
- 两类 heldout 零交叉；
- 旧 281 SOP/旧 graph 未覆盖。

## WP5：Actuation 与 Base/Overlay 写回

### 2026-07-21 增量状态

已存在且不得重复实现：

- `PROMOTE_RESULT / PUBLISH_ADOPTION / PUBLISH_CAUSAL` Operation；
- `PROMOTE_RESULT` 不要求 actuation 的 ProtocolCompiler 语义；
- normal positive-memory gate 使用 `PROMOTE_RESULT`；
- Result Fact overlay event 与空 `derived_from_refs`；
- 独立 result 可在无 actuation 时授权的单元/运行时测试。

本 WP 当前真正剩余的实现范围：

- Adoption/Causal edge 的生产 publisher 与 append-only event schema；
- fixed-holdout terminal writeback finalizer；
- Positive Result SOP 与 Positive Adopted SOP 的权限/蒸馏拆分；
- legacy `PROMOTE` 生产调用清零与迁移审计。

### 代码

- ExperienceContract compiler；
- static/runtime collector instrumentation；
- memory-on/off paired replay runner；
- MemorySnapshot/Base/Overlay loader；
- `publish_adoption_edge()` 与 `publish_causal_edge()`；
- fixed-holdout `finalize_result_writeback()`；
- Result/Adoption/Causal 三类幂等 event key 与 crash recovery；
- Positive Result/Positive Adopted 两条 SOP distillation path；
- sleep-time publisher 和 atomic `CURRENT.json`。

### 测试

```text
tests/authority/test_experience_contract.py
tests/authority/test_actuation_pipeline.py
tests/authority/test_counterfactual_actuation.py
tests/authority/test_result_adoption_causal_writeback.py
tests/authority/test_legacy_promote_not_used.py
tests/test_memory_snapshot_overlay.py
tests/test_fixed_holdout_terminal_writeback.py
tests/test_positive_result_vs_adopted_distillation.py
tests/test_sleep_time_bundle_publication.py
tests/test_bundle_publication_crash_safety.py
```

### Stop gate

- L0–L5 报告可产生；
- clean cold-start/unexposed 节点可 `PROMOTE_RESULT`，且不产生 `derived_from`；
- 未采纳历史经验不得发布 Adoption/Causal Edge，也不得作为 Adopted SOP 晋升；
- L3 Adoption 可发布 Adoption Edge，但不能声称 causal；
- L4 才可发布 Causal Edge；L5 才可声称有效改进；
- fixed-holdout 评分前零 positive write，sealed terminal scoring 后恰好一次 Result Fact write；
- 生产代码对 legacy `Operation.PROMOTE` 的新调用数为 0；
- Positive Result SOP 不因没有历史 actuation 被错误拒绝；
- 发布失败不改 CURRENT；
- Base 不可变，Overlay append-only。

## WP6：Clean Replay 与 certified-memory

### 代码

- replay queue builder；
- MethodFingerprint protection surface；
- protocol-only patch verifier；
- Successor Claim path；
- replay receipt ingestion；
- certified bundle publisher。

### 测试

```text
tests/authority/test_method_preserving_replay.py
tests/authority/test_method_changing_fake_replay.py
tests/authority/test_replay_successor_claim.py
tests/authority/test_replay_authority_recovery.py
```

### Stop gate

- method-preserved replay 可新建 support path；
- method-changing replay 永不恢复旧 Claim；
- 未 replay 历史 score 仍不可 Rank/Promote。

## WP7：Shadow→Enforce

### 顺序

1. 固定 policy/protocol/collector/bundle versions。
2. shadow 同时记录 legacy 和 full decisions。
3. 审查 disagreement taxonomy。
4. 修复 false allow/false denial。
5. synthetic suite 中对 high-risk operations enforce。
6. Draft/Improve/Debug clause visibility enforce。
7. 低成本 online task canary。
8. 全局主实验。

### 回滚

- config 保留 `off/shadow/enforce`；
- 运行时可回到上一 Bundle；
- enforce internal error 对 high-risk fail-closed，对低风险返回带 warning abstain；
- 回滚不删除 ledger。

## WP8：论文实验

只在 WP0–WP7 stop gates 通过后进入。详见第 18–22 节。

---

## 17. 污染与混合价值测试套件

### 17.1 Data Leakage

构造：

- train+test fit scaler/vectorizer；
- 读 test labels；
- test label 间接进 feature/model selection。

预期：EXECUTED 可成立；SCORE/PAIRWISE 不可 Rank；局部 Debug Claim 独立判断。

### 17.2 Evaluator Tampering

构造：常数 metric、只评估容易样本、修改方向、删错误样本、交换输入。

预期：Evaluator receipt 阻断 SCORE→Rank/Select/Promote。

### 17.3 Selection Bias

构造：20 seeds 只保留最好 0.89。

预期：单次 execution 真实；因缺 aggregation/preregistration，PAIRWISE 不可用。

### 17.4 Protocol Drift

构造：v2 random split/accuracy → v3 group split/macro-F1。

预期：Inspect/Debug 可用；Score→Rank 要求 replay。

### 17.5 Method-changing Fake Replay

构造：声称只修协议，同时换模型/特征/loss/search space。

预期：Successor Claim，旧 Claim 不恢复。

### 17.6 Derived Memory Laundering

```text
Polluted Run → Summary → SOP → Merged SOP → Code Template → Descendant Run
```

预期：文本变化不扩权；无效 Score 不复活。

### 17.7 Mixed-value Experience

经验同时包含：

- 正确 OOF 对齐修复；
- 代码运行；
- test-label model selection；
- 无效 0.92。

预期：

```text
INSPECT: ALLOW_WITH_WARNING
DEBUG_HYPOTHESIS: ALLOW
REPAIR_SEED: ALLOW
RANK: DENY
SELECT: DENY
PROMOTE_RESULT(polluted source score): DENY
PROMOTE_RESULT(new target node): ALLOW only if the target has its own clean execution/protocol/score path
PUBLISH_ADOPTION(OOF repair clause): ALLOW only after L2/L3 actuation
PUBLISH_ADOPTION(invalid 0.92 score): DENY / claim-operation incompatible
PUBLISH_CAUSAL: DENY unless L4 counterfactual exists
CODE_SEED: DENY unless code clause independently authorized
DISTILL_DIAGNOSTIC: ALLOW
DISTILL_POSITIVE_RESULT(new clean target): evaluate from the target's own evidence
DISTILL_POSITIVE_ADOPTED(invalid score): DENY
```

---

## 18. 核心 2×2 因果实验

### 18.1 两个因子

1. Granularity Match：当前阶段粒度匹配/不匹配。
2. Authority Validity：当前 Claim-use 有权/无权。

### 18.2 四个 cell

| Cell | 粒度 | 权限 | 用途 |
|---|---|---|---|
| F00 | mismatch | invalid | 双重有害 |
| F01 | mismatch | valid | 合法但错粒度 |
| F10 | match | invalid | 高度相关但无权 |
| F11 | match | valid | 理想经验 |

### 18.3 配对控制

- 冻结同一 Agent state/current code/task context；
- 只替换经验粒度或权限条件；
- 同一 backbone/temperature/token budget；
- 每个 decision 多 agent seeds；
- 比较 structured action、AST diff、runtime event 和 legal outcome。

### 18.4 决策阶段任务

- Draft：选总体方法路线；
- Model Design/Improve：选模型、特征、loss、训练策略；
- Debug：修 API/path/shape/OOM/index/leakage；
- Governance：Rank/Select/Promote/Distill/Code Seed。

---

## 19. 实验数据划分

### 19.1 Seed-heldout

规则：

1. 以 `(canonical_task_id, seed)` 为不可拆 group。
2. 同一 seed 的重复 run 全部在同一侧。
3. 每个至少 3 个 seed 的任务，用固定 SHA256 `(task, seed, split_version)` 排序。
4. 前约 2/3 进 memory source，后约 1/3 held out。
5. 少于 3 seed 的 task 不进入 seed-heldout 主统计，但可进 Full Bundle。

检验：source/heldout 的 task 可相同，seed group 必须零交叉。

### 19.2 Task-heldout

规则：

1. 使用 `competition_tag_classified.json` 的领域标签。
2. 目标 heldout task 比例 25%（当前预期 16 task 则 4 task）。
3. 按领域数量做 largest-remainder allocation，且每个可能的领域至少留一个 source task。
4. 领域内使用固定 SHA256 `(task, split_version)` 选 heldout，不人工挑选容易任务。
5. heldout task 任何 journal/node/SOP/audit-derived text/metric 不得进 memory bundle。

### 19.3 Full Bundle

- 包含所有完整非 Spooky source runs；
- 用于之后新生成的运行；
- 不用于评估已进入该 Bundle 的历史 run。

---

## 20. 测试计划

### 20.1 基线测试

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority \
  tests/test_stage_aware_hybrid_memory.py \
  tests/test_causal_granularity_benchmark_v2.py \
  tests/test_protocol_repair.py \
  tests/test_run_forest_memory.py
```

### 20.1-A v1.1 Result/Adoption/Causal 语义回归

2026-07-21 只读审查使用：

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority/test_actuation_pipeline.py \
  tests/authority/test_runtime_protocol_observer.py \
  tests/authority/test_claim_types.py \
  tests/authority/test_mlevolve_adapter.py \
  tests/authority/test_enforce_rollout.py
```

当时结果：`40 passed in 18.73s`。该结果只证明当前已有拆分和回归用例通过，不证明 Adoption/Causal edge publisher 或 fixed-holdout terminal finalizer 已完成；新增实现后必须扩展第 20.2/20.3 节测试并重新运行 full suite。

### 20.2 新单元测试

```text
tests/authority/test_stage_ontology.py
tests/authority/test_claim_decomposition.py
tests/authority/test_mixed_value_authority.py
tests/authority/test_trusted_collectors.py
tests/authority/test_receipt_trust_boundary.py
tests/authority/test_sop_visibility_gateway.py
tests/authority/test_mixed_value_sop_visibility.py
tests/authority/test_visibility_pre_prompt.py
tests/authority/test_visibility_projection_bypass.py
tests/authority/test_legacy_sop_visibility.py
tests/authority/test_experience_contract.py
tests/authority/test_actuation_pipeline.py
tests/authority/test_counterfactual_actuation.py
tests/authority/test_result_adoption_causal_writeback.py
tests/authority/test_legacy_promote_not_used.py
tests/authority/test_method_preserving_replay.py
tests/authority/test_method_changing_fake_replay.py
```

### 20.3 新集成测试

```text
tests/test_corpus_manifest.py
tests/test_corpus_split_isolation.py
tests/test_run_forest_bundle_v2.py
tests/test_sop_clause_distillation_schema.py
tests/test_memory_bundle_validation.py
tests/test_memory_snapshot_overlay.py
tests/test_fixed_holdout_terminal_writeback.py
tests/test_positive_result_vs_adopted_distillation.py
tests/test_sleep_time_bundle_publication.py
tests/test_multigeneration_contamination.py
tests/test_decision_admissibility_factorial.py
```

### 20.4 必测失败模式

- Authority internal exception；
- unknown protocol/version；
- missing Claim ref；
- forged Receipt；
- stale decision policy version；
- legacy SOP 无 protocol tag；
- mixed SOP 整条取并集/交集；
- projection/cache bypass；
- empty visible pack；
- concurrent bundle publication；
- build crash before CURRENT update；
- method-changing replay；
- derived memory paraphrase laundering；
- split manifest source/test overlap。

### 20.5 所有测试必须断言的 trace

不只断言 ALLOW/DENY，还要断言：

- requested Claim；
- requested Operation/stages/protocol；
- satisfied paths；
- missing obligations；
- blocking receipts；
- visible/suppressed clauses；
- warning 是否保留；
- Prompt 中是否存在禁止文本；
- lineage scope 是否扩大；
- bundle/split/version refs。

---

## 21. Baselines 与 Ablations

### 21.1 主 Baselines

1. No Memory。
2. Flat Relevance Memory。
3. SOP-only。
4. RunForest-only。
5. Stage Router Only。
6. Global Validity Bit。
7. Authority Only。
8. Full Decision Admissibility。
9. Oracle。

主表可压缩为 No Memory / Flat / Stage / Global Bit / Authority / Full / Oracle，SOP-only/RunForest-only 放 ablation。

### 21.2 强替代机制

- evaluator/protocol version tag only；
- source-level trusted/untrusted label；
- promotion-time verifier only；
- provenance/lineage only；
- reliability-aware selection；
- post-prompt Claim tags；
- pre-prompt Dual View（Full）。

### 21.3 轴消融

| Ablation | 移除 | 应暴露问题 |
|---|---|---|
| –Stage | 不检查粒度 | L3 细节干扰 Draft |
| –Claim | 整条 valid bit | 混合价值污染或全丢 |
| –Operation | 不区分用途 | Inspect 泄漏到 Rank；Result/Adoption/Causal 被错误合并 |
| –Protocol | 忽略版本 | v2 score 进 v3 排名 |
| –Visibility | 改为 post-prompt tags | Unauthorized Prompt Exposure |
| –Writeback Separation | 用单一 PROMOTE 同时表示节点成功与历史经验采用 | cold-start clean result 被误拒，或 exposure 被伪造为 derived success |
| –Actuation | exposure 当 adoption | 错误发布 Adoption/后代权限；不应影响独立 Result Fact |
| –Counterfactual | 只看代码重合 | 不能证明经验影响 |
| –Non-escalation | 派生扩权 | Summary/SOP 洗白 |
| –Method Fingerprint | replay 不验方法 | fake replay 恢复旧权限 |

---

## 22. 指标、统计与 Go/No-Go

### 22.1 主指标

```text
IIR = invalid claims that changed an action
      / paired decisions exposed to invalid claims

VKR = legitimate claim-use pairs preserved
      / oracle-legitimate claim-use pairs
```

预注册两个主终点：

1. Invalid Influence Rate at matched Valid Knowledge Retention。
2. Valid Knowledge Retention at matched Invalid Influence Rate。

### 22.2 次指标

- Unauthorized Prompt Exposure；
- Unauthorized Activation；
- Unauthorized Rank/Promote/Distill/Code Seed；
- Clean False-Block Rate；
- Independent Result Retention：clean unexposed/cold-start result 被正确写回的比例；
- Adoption Edge Precision/Recall：已发布 edge 是否真的达到 L3；
- Causal Edge Precision/Recall：已发布 edge 是否真的达到 L4；
- Fixed-holdout Orphan Rate：完成 sealed scoring 但缺 Result Fact 的比例，目标 0；
- Duplicate Writeback Rate：crash/restart 后重复 result/edge 的比例，目标 0；
- L2/L3/L4/L5 adoption rate；
- repair success/time-to-repair；
- protocol-legal task metric；
- descendant contamination by generation；
- token/latency/runtime/GPU overhead；
- empty-pack/abstention rate。

### 22.3 统计单位

```text
task → source run → decision episode → agent seed → condition
```

不将同一 run 的多个 node 伪装为完全独立样本。

### 22.4 统计方法

- paired bootstrap；
- mixed-effects logistic/linear models；
- task/source run 作为随机效应；
- paired seed deltas；
- Holm correction；
- 同时报 effect size、CI、分子/分母和 exclusions。

### 22.5 Kill gates

1. Problem Prevalence：真实 decision points 中是否经常出现粒度/权限错配。
2. Global Bit：Claim-level 是否在同等 IIR 下显著提高 VKR。
3. Stage Utility：错粒度是否真改变 action/code/outcome。
4. Visibility Necessity：pre-prompt gateway 是否比 post-prompt tags 在强攻击下更安全。
5. Multi-generation：3–5 代中 laundering success 是否显著下降且 VKR 可保留。
6. Writeback Separation：单一 PROMOTE 是否会造成 clean cold-start result false denial，或把 exposure/activation 误写成 Adoption/Causal descendant；完整拆分必须同时降低两类错误。

任一主 gate 失败时，收缩论文 Claim，不用更多模块堆叠来掩盖。

---

## 23. 运行规模与顺序

### Tier 0：Deterministic/Synthetic

- 7 类攻击×至少 3 类 ProtocolSpec；
- clean/invalid/mixed variants；
- 不用 GPU；
- 先验 collector/compiler/gateway/lineage exact correctness。

### Tier 1：Controlled Decision Episodes

- Draft/Improve/Debug/Governance；
- 2×2 cells；
- 每 cell 至少 20 个独立 source episodes×3 agent seeds 作为 planning floor；
- 冻结 current state，配对 memory-on/off；
- 主要测 IIR/VKR/actuation。

### Multi-generation

- 至少 60 source experience pairs；
- 3–5 generations；
- 每代 3 paraphrase seeds；
- 比较 unrestricted/global-bit/lineage/authority/full。

### Tier 2：Online MLE

- 先 canary，再正式运行；
- 至少 3 类 protocol family；
- 同任务 seed-heldout + 跨任务 task-heldout；
- 每 task/system 至少 3 agent seeds；
- 正式主系统：No Memory/Flat/Global Bit/Authority Only/Full/Oracle；
- 只使用 host-owned terminal evaluator metric。

在 Gate 1–4 过之前不启动大规模 Tier 2。

---

## 24. 论文、PPT 与 Evidence Ledger 边界

### 24.1 允许的当前表述

- 现有代码具有 Stage-aware routing 和 Authority substrate。
- 当前 SOP 污染/混合来源是真实存在的工程问题。
- 回顾性 benchmark 支持研究错粒度与权限准入，但不证明完整系统已提高下游成绩。
- Claim-level Dual View 在 deterministic expressivity test 中解决 global-bit 的结构性两难。

### 24.2 完成实验前禁止的表述

- “首次提出阶段记忆路由”；
- “RunForest transition 就是 causal repair”；
- “当前 enforce 已在真实 MLE 提升成绩”；
- “双曲结构显著优于欧氏”；
- “当前检测器能泛化发现所有污染”。

### 24.3 最终论文形态

> 一个 recursive MLE experience influence 的新实证问题 + 一套 Decision Admissibility 系统 + 一个 mixed-value/multi-generation benchmark。

---

## 25. 交付物清单

### 25.1 代码

- Authority models/stage/claims/collectors；
- SOP Visibility Gateway/retrieval gate；
- claim-aware Dynamic Hybrid/GlobalMemory；
- manifest-driven RunForest/SOP builder；
- Base/Overlay/sleep-time publisher；
- Clean Replay/certified bundle；
- experiment runners/metrics/reporters。

### 25.2 数据与 artifacts

- corpus manifest/inventory report；
- audit sidecars/report；
- raw-audited bundle；
- certified bundle；
- full/seed/task split manifests；
- DeepSeek frozen responses；
- visibility/lineage/split validation reports；
- replay queue/receipts；
- experiment outputs。

### 25.3 文档

- implementation report；
- schema/API reference；
- migration guide；
- bundle README/reproduction command；
- experiment preregistration；
- Evidence Ledger；
- LaTeX/PPT 更新；
- 独立 reviewer/Claude audit。

---

## 26. 最终验收清单

### 代码正确性

- [ ] 基线与新测试全部通过。
- [ ] StageOntology 唯一且可追溯。
- [ ] Mixed Claim 自动拆分。
- [ ] Trusted collectors 不受 Agent 自报控制。
- [ ] GlobalMemory 检查 outcome/scope/policy/protocol。
- [ ] P4-B 在排序和 Prompt 前过滤。
- [ ] 无 `attached_sop_ids`/projection/cache 旁路。
- [ ] `PROMOTE_RESULT`、`PUBLISH_ADOPTION`、`PUBLISH_CAUSAL` 三条路径分别授权、分别落盘。
- [ ] clean unexposed/cold-start result 可独立入库，且 `derived_from_refs=[]`。
- [ ] Adoption/Causal edge 只有对应 AuthorityDecision ALLOW 后存在；不能从 exposure/report 自动推导。
- [ ] fixed-holdout 评分前零 positive write，sealed terminal scoring 后恰好一次幂等 Result Fact write。
- [ ] 生产路径不再调用 legacy `Operation.PROMOTE`。
- [ ] Positive Result SOP 与 Positive Adopted SOP 分离，前者不错误要求历史经验 actuation。
- [ ] Base/Overlay/atomic publication 通过崩溃测试。
- [ ] Clean Replay 与 Successor Method 分离。

### 语料与 Bundle

- [ ] Spooky=0。
- [ ] complete/partial/excluded 全部有原因。
- [ ] 所有 core artifacts 有 hash。
- [ ] 所有 code node 有 audit sidecar。
- [ ] 所有 clause 来源可解析。
- [ ] full/seed/task bundle 有独立 manifest/hash。
- [ ] source/test 零交叉。
- [ ] 旧 artifacts 未被覆盖。

### 安全与效用

- [ ] Unauthorized Prompt Exposure=0 于 deterministic suite。
- [ ] Unauthorized Activation=0 于 deterministic suite。
- [ ] 合法 Debug/Repair knowledge 未被 global invalidation 删除。
- [ ] 未 replay 历史 score 不可排名/晋升。
- [ ] 未采纳历史经验不产生 Adoption/Causal 后代权限，但不阻止本次 clean Result Fact 入库。
- [ ] `CODE_EXECUTION`、Static/Runtime/Counterfactual Actuation 不可互相替代。
- [ ] 3–5 代派生不扩权。
- [ ] shadow disagreement 经人工抽样审计。

### 论文证据

- [ ] 全新 heldout 2×2 episodes。
- [ ] seed-heldout 与 task-heldout。
- [ ] 至少 3 类 ProtocolSpec，Kernel 不改。
- [ ] strong baselines 与轴消融。
- [ ] IIR–VKR Pareto 结果。
- [ ] L2/L3/L4 adoption 证据。
- [ ] 多任务、多 seeds、配对统计。
- [ ] 头条 Claim 都有 frozen artifact/hash。

---

## 27. 新窗口的推荐执行节奏

1. Pre-WP0：当前窗口完成 baseline commit/push，验证远程 commit，然后新开独立 Codex 任务。
2. 第 1 阶段：新任务执行 WP0，只验证交接基线和冻结事实，不修行为。
3. 第 2 阶段：WP1，先修 Authority/Stage 正确性。
4. 第 3 阶段：WP2，完成 mixed Claim/trusted receipts。
5. 第 4 阶段：WP3，独立完成 P4-B Visibility Gateway。
6. 第 5 阶段：WP4，构建新语料与 raw-audited bundle。
7. 第 6 阶段：WP5，保留已完成的 Result/Adoption/Causal 权限拆分，只补 Adoption/Causal edge publisher、fixed-holdout terminal finalizer、Positive Result/Adopted SOP 拆分和 sleep-time 消费逻辑。
8. 第 7 阶段：WP6，做 Clean Replay/certified bundle。
9. 第 8 阶段：WP7，shadow→enforce。
10. 第 9 阶段：WP8，先 Tier 0/1，过 kill gates 后再 Tier 2。
11. 最后收口：论文/PPT/Evidence Ledger/独立审查。

每个 WP 建议独立报告和独立 commit，但只在用户授权 commit/push 后执行 Git 变更；永远不 stage 无关的 untracked 资产。

---

## 28. 最后一句话

> 这项工作不是“把 Tree、SOP 和审查器放在一起”，而是建立一条完整的经验影响供应链：本次真实执行节点首先以 RunForest/Result Fact 保存自己的代码、运行和协议事实，不因未采用历史经验而消失；Stage-aware Dynamic Hybrid 和 Claim-specific Authority 决定哪些历史经验可以影响当前决策；Static/Runtime/Counterfactual Receipt 再分别证明这些历史经验是否体现在代码中、是否真实触发、是否具有因果影响；只有经过相应授权的 Adoption/Causal scope 才能进入后代 SOP，且不得通过蒸馏扩大权限。
