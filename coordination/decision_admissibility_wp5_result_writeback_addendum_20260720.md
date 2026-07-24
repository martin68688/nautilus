# WP5 v1.1 Result / Adoption / Causal Writeback 增量报告

日期：2026-07-20  
分支：`codex/dual-time-procedural-memory`  
范围：重新打开的 WP5 Stop Gate（不覆盖既有 WP5 报告）  
状态：**PASSED（本地代码与离线 Bundle 管线）**

## 语义边界

本次实现固定三种不同对象：

```text
PROMOTE_RESULT
  当前实际执行节点 -> Result Fact
  derived_from_refs 必须为空

PUBLISH_ADOPTION
  历史 ExperienceContract -> 当前节点 Adoption Edge
  需要 contract-bound static + runtime actuation

PUBLISH_CAUSAL
  已发布 Adoption Edge -> 当前节点 Causal Edge
  还需要 contract-bound counterfactual actuation
```

训练代码的事实源仍是 Journal/RunForest；Result Fact 只保存
`artifact_id + code_sha256 + immutable journal pointer`。Static/runtime
actuation 只证明历史经验影响当前代码/运行路径，不证明训练代码本身执行。

## 本次补齐的生产接线

- `mlevolve/authority/adapters/mlevolve/runtime.py`
  - Result、Adoption、Causal event 分别写入 Overlay；
  - 每个 event 绑定内容寻址的 immutable Authority snapshot；
  - Edge 写入 `edge_claim_ref`，避免只凭报告反推 Adoption/Causal。
- `mlevolve/fixed_holdout/writeback.py`
  - sealed terminal score 前不写 positive memory；
  - terminal snapshot 在 Result Fact 前封存并绑定 hash；
  - 重试仍按幂等 key 只产生一个 Result Fact。
- `mlevolve/authority/bundle_publisher.py`
  - 校验 event 链、Claim/Receipt/Decision/snapshot 完整绑定；
  - 构造 `positive_writeback_plan`；
  - typed event 未被 distillation report 精确消费时 fail closed。
- `mlevolve/authority/positive_distillation.py`
  - Authority 先验证原始 Score/Adoption Claim；
  - 再创建新的 `METHOD_HYPOTHESIS` derived Claim 和独立 evidence path；
  - Score Claim 不再伪装成可生成方法。
- `mlevolve/authority/writeback_distillation.py`
  - 将 Result/Adoption/Causal inventory 映射为独立 Positive Result/Positive Adopted 候选；
  - exposure 或 `verified_adoption_report_refs` 不会自动变成 Adoption；
  - causal 文案必须显式声明 `assertion_level=causal` 且存在 Causal Edge。
- `paper-skills/memory_bundle/bind_positive_writeback.py`
  - host-owned formal binder，输出 claims/receipts/paths/decisions/clauses/derivations；
  - 每条 positive clause 经过 `bind_sop_clauses.py` 的 typed lineage validator。
- `paper-skills/memory_bundle/positive_writeback_pipeline.py`
  - 将 typed material 合并到新 Bundle 的 authority、SOP、RunForest/index、visibility mask 和 manifest；
  - 父 Base 保持不变；metadata 不再伪称 Clean Replay。
- `paper-skills/memory_bundle/publish_sleep_time_bundle.py`
  - `--positive-proposals --protocol-registry` 选择上述严格 host-owned pipeline；
  - 任意 pipeline 仍必须显式提供，避免 copy-and-bless 默认路径。
- `mlevolve/authority/adapters/mlevolve/retrieval_gate.py`
  - 识别 `positive_result` / `positive_adopted` publication class；
  - 实际生成权限仍由 derived Method Claim + live Authority 决定，跨任务不会因 positive 标签绕过 certified-method 门。

## Stop Gate 证据

| 条件 | 证据 |
|---|---|
| L0–L5 可表示、顺序有效 | `tests/authority/test_experience_contract.py`, `test_actuation_pipeline.py`, `test_counterfactual_actuation.py` |
| clean cold-start 可独立写 Result Fact | `tests/authority/test_result_adoption_causal_writeback.py`；Result event 的 `derived_from_refs=[]` |
| 未采纳经验不产生 Adoption/Causal | 同上；result-only inventory / plan 的 adopted count 为 0 |
| L3 可发布 Adoption、不能冒充 causal | 同上；无 counterfactual 时 Causal denied |
| L4 才可发布 Causal | 同上；先发布 Adoption，再补 counterfactual 后才允许 Causal |
| fixed-holdout 评分前零写回、终局恰一次 | `tests/test_fixed_holdout_terminal_writeback.py` |
| legacy `Operation.PROMOTE` 新生产调用为 0 | `tests/authority/test_legacy_promote_not_used.py`（AST 检查） |
| Positive Result 不要求历史 actuation | `tests/test_positive_result_vs_adopted_distillation.py` |
| Positive Adopted 必须有 adopted evidence | 同上；缺 L3 fail closed |
| sleep-time 不得忽略 typed event | `test_result_adoption_causal_writeback.py`：空泛 distillation report 被拒；正式 binder/pipeline 通过 |
| Base 不可变、Overlay append-only、发布失败不改 CURRENT | `tests/test_memory_snapshot_overlay.py`, `test_sleep_time_bundle_publication.py`, `test_bundle_publication_crash_safety.py` |

## 测试结果

```text
WP5 targeted（含正式 binder/staging builder） 47 passed
tests/authority                              170 passed
计划 §20.1                                  369 passed
计划 §20.1-A                                 43 passed
WP6 focused                                  17 passed
完整 suite（排除冻结 composite benchmark）     512 passed
compileall mlevolve paper-skills tests       passed
scoped git diff --check                       passed
```

冻结例外保持原状：
`tests/test_composite_memory_benchmark.py` 为 `18 passed, 1 failed`，唯一失败是
heldout lock 中旧 detector SHA 与当前/基线 detector SHA 不一致；本次没有改写
lock 或 detector 来制造通过。

## 边界与后续

- 本报告只证明 WP5 本地离线实现和 staging Bundle 管线；尚未授权或启动新的 Kubernetes Pod。
- r25 legacy canary 不被重新解释为 corrected semantics；新的 WP7 canary 必须使用新冻结 source。
- WP6 既有 Stop Gate 重新复验通过；WP7 corrected canary、独立 oracle/gate 仍是下一阶段。
- 本轮未创建 commit、未 push、未提交 Kubernetes Job，也未覆盖 dirty worktree 用户资产。

**WP5 v1.1 Stop Gate：PASSED。WP6 可复验，随后才可进入 corrected WP7。**
