# Claude Agent MCP: RunForest Leakage Audit V2 (Ultracode-Equivalent)

- Date: 2026-07-10
- Session: `0f4793a4-fcea-4a23-a7f7-143e1fc7574c`
- Runtime: Claude Agent MCP
- Mode: `Opus + effort=max + three read-only Task subagents`
- Permissions: `Read`, `Glob`, `Grep`, `Task`, `TaskOutput`, `TaskStop`
- Mutation permissions: none

The text below is a faithful structured record of the final public report returned by Claude Agent MCP. Intermediate hidden reasoning is intentionally not included.

## RunForest 泄漏审计 V2 - 对抗性审计

### 方法论

派遣了三名只读子审计员（detector / propagation / replay）进行独立检查。随后主审计员亲自复核关键代码路径。`coordination/runforest_leakage_audit_v2.md` 被作为未经证实的主张处理。

### Blockers

#### B1 - LLM 审计不可用时，指标仍可进入 best-solution 竞争

**确认 Bug。** `mlevolve/agents/data_leakage_agent.py:152-159` 在异常时返回 `has_leakage=False, classification="audit_unavailable"`。`mlevolve/agents/leakage_audit.py:444-445` 将其变为无 issue、`hard_block=False` 的不可用审计。`mlevolve/agents/result_parse_agent.py:384` 只在 `hard_block` 时重置 metric。

不可用审计会被挡在正向 memory 之外，但指标仍保留。`solution_manager` 没有读取 `metric_disposition`，因此静态检测未覆盖、LLM 又失败的节点仍可能更新 `best_node`。

#### B2 - deterministic detector 可被简单重命名或未列出的 transformer 绕过

**确认 Bug。** `_HOLDOUT_PARTS` 仅覆盖 `val/valid/validation/holdout/test`；`X_dev`、`eval`、`oos` 等不会被标记。`_STATEFUL_TRANSFORMERS` 是硬编码列表，未覆盖 `Pipeline`、`TargetEncoder` 和用户自定义 transformer。几个 regex detector 也依赖字面变量名和循环形式。

例如将 `X_val` 改名为 `X_dev`，或使用 `Pipeline.fit`，就可能绕过静态预检和 replay 里的同一 `audit_code()`。

#### B3 - FailurePattern 反馈只作为提示文本，不是真正的阻断器

**确认缺口。** FailurePattern 在 `external_skill_memory.py` 中只被格式化成 warning/failure card。生成前没有语义匹配阻断；运行时 preflight 也不会查询 FailurePattern。FailurePattern id 又依赖 `code_sha256 + issue_code`，轻微改名就会生成不同 hash。

因此系统能阻止已知代码或静态 detector 再次命中的代码，却不能保证阻止语义相同的变体。

### Majors

#### M1 - 缺少 leakage_audit 时，正向准入 fail-open

**确认 Bug。** `external_skill_memory.py:1902-1906` 在 audit 缺失或为空时回退到 `node.get("is_buggy") is not True`。旧 schema、部分构建或被手工修改的节点可能因此进入正向检索。应改为缺 audit 时返回 `False`。

#### M2 - debug/improve 子节点不继承结构化 leakage_audit

**确认 Bug。** `triggers.py:67-93` 继承 draft role、source refs 和 replay 字段，但没有继承 `leakage_audit`。debug/improve prompt 可能收到软警告，但子节点结构化字段从空字典开始。若新代码绕过 detector，就会被重新认证为 clean。

#### M3 - solution_manager 不执行 metric_disposition

**确认 Bug。** `solution_manager.py` 选择 top candidate/best solution 时只检查 `is_buggy`、`metric`、`is_valid`，不检查 `leakage_audit` 或 `metric_disposition`。因此 `protocol_biased` 或 `audit_unavailable` 节点仍可能成为 `best_solution/solution.py`。

#### M4 - registry 缺失/损坏不会形成强制阻断

**确认 Bug。** `leakage_audit.py:537-550` 对缺失 registry 返回 `None`，对损坏 registry 返回无 issue、`hard_block=False` 的 unavailable dict。merge 后可能没有有效贡献。Replay 仍有 fresh static audit 兜底，但普通检索没有同等保护。

#### M5 - 单个默认 False 配置可关闭 deterministic 和 LLM 两层审核

**风险。** `result_parse_agent.py:342` 使用 `getattr(agent.acfg, "check_data_leakage", False)`。缺配置字段会静默关闭 preflight；post-run LLM 审核也受同一开关控制。

### Minors

- LLM severity 最高只到 `high`，无法产生 `critical`，分级不一致。
- 空代码在 `audit_code` 中会得到 clean；replay 另有非空检查兜底。
- negative GlobalMemory 与 positive records 共用索引，依赖调用方一直正确过滤。
- WorstMetricValue 节点会跳过 post-run LLM 审核。
- 现有测试没有覆盖 rename bypass、Pipeline、LLM 异常、registry 损坏、FailurePattern 阻断闭环或 protocol-biased 节点成为 best solution。

### d93 降级闭环

**PASS。** `run_forest_replay.py:101-116` 有三层门：fresh deterministic audit、target manifest 的 `verified_clean` 要求、`paper_grade_eligible is True` 要求。代码 hash 同时与 manifest 和图节点核对。测试也验证历史三模型源会因两个 issue code 被拒绝。

残余风险：`REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION` 单独存在时是 warning/protocol bias，不是 execution hard block；未来只有 ensemble selection bias 而没有 transform-fit leakage 的节点仍会执行。

### 下次阻止同类方案闭环

**FAIL。** FailurePattern 是建议性上下文，不是强制门；匹配依赖精确 SHA；fresh static audit 又可被重命名和未列出的 transformer 绕过。因此当前只能说“已知 d93 被可靠降级”，不能说“同类方案不会再次被采用”。

### 最小修复

1. 在 `solution_manager` 的 top/best 准入中强制执行 `metric_disposition`，拒绝 `protocol_biased`、`unverified`、`reject`。
2. LLM audit unavailable 时拒绝该 metric 成为 best candidate，或将 metric 置为 WorstMetricValue。
3. `_positive_memory_eligible` 缺 audit 时返回 `False`。
4. 子节点继承父节点的 leakage verdict 作为 baseline，再由子节点 fresh audit 覆盖。
5. FailurePattern 从“prompt warning”升级为生成/执行前的结构化 gate，并采用 AST/语义 fingerprint，而非原始代码 SHA。
6. 扩展 holdout alias 与 transformer 覆盖，增加 `Pipeline`、常见 target encoder 和自定义 fit 接收者分析。

### 必需测试

- `X_dev` rename bypass 必须被检测。
- `Pipeline.fit` on all/holdout data 必须被检测。
- LLM 异常节点不得成为 best node。
- registry 损坏时 replay/检索 fail closed。
- blocked 父节点的 debug/improve 子节点继承 verdict。
- d93 变体的 FailurePattern 能参与强制阻断。
- protocol-biased 节点不能成为 `best_solution/solution.py`。
- 无 audit 的 legacy node 不具备正向 memory 资格。

### Verdict

**APPROVE WITH CHANGES。** d93 的降级可靠，但更广泛的防泄漏闭环尚未成立。当前能支持的表述是：系统阻断了已知 d93 模式，并降低了部分新变体的风险；不能声称已经普遍封闭泄漏。

### 子审计纠错

- Detector 子审计曾把 delayed exception path 误报为 fail-open；主审计确认 `agent_search.py:462` 会重新抛出异常，因此该路径 fail-closed。
- Propagation 子审计曾把 replay audit 未附加到节点评为 Blocker；主审计降为 defense-in-depth 问题，因为 replay gate 在节点加载前已强制执行。
