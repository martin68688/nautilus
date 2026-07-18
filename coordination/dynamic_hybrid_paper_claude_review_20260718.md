# ClaudeAgent 独立审查

审查日期：2026-07-18  
审查方式：Claude Code 2.1.185，只读对照论文、生产实现、测试与 v2 benchmark 报告。

## 总评

论文的代码、公式和数值一致性扎实。排序分数、证据可信度、0.50
门槛、60% Tree 上限、40% SOP 下限，以及 Flat-Twin 只参与排序而不参与
可信度，均与生产实现一致。论文也正确地把当前结果限制为离线检索与路由
诊断，没有将规则型可信度表述为成功概率，也没有宣称已证明下游 MLE 收益。

最重要的问题是选择性报告风险：v2 JSON 中仍有旧固定 0.75 Tree 权重消融，
其 covered-episode Hit@1/MRR 高于动态方法，但 fallback、route 和 selective
accuracy 较低。如果论文完全不披露，审稿人可能认为隐藏了不利指标。建议将其
作为废弃、非生产消融在附录透明报告，并解释排名覆盖与安全回退之间的取舍。

## 发现（按严重程度）

### 中：旧固定权重消融未披露

- **证据**：`causal_granularity_report_v2.json` 中
  `causal_tree_fixed_075` 的 test 指标为 route=0.7600、selective=0.7200、
  Hit@1=0.7500、MRR=0.7500、fallback=0.6923；动态方法为 0.8000、
  0.7600、0.5833、0.5833、0.9231。
- **风险**：动态方法的优势是更准确地决定何时不用 Tree，而非所有检索指标
  都更高。完全省略固定权重结果会造成选择性报告印象。
- **建议**：在附录披露完整 trade-off，明确旧方案不是生产路由，不把它恢复
  为主方法。

### 低至中：机制图与结果图部分重复

- `dynamic_hybrid_explainer` 的 panel C 与 `retrieval_results` panel B
  重复展示部分 selective accuracy 数字。
- 建议后续精简机制图的定量 panel 或移到附录。当前仍可读，不是事实错误。

### 低：0.50 硬阈值的跳变未讨论

- `c<0.50` 时 Tree 权重为 0，而 `c=0.50` 时立即变为 0.30。
- 建议明确这是 fail-closed 设计而非校准后的最佳阈值，并补阈值敏感度实验。

### 低：多 gold transition 下的指标定义不够精确

- Hit@1、Recall@5 和 MRR 使用 episode 的 silver-gold transition 集合，但正文
  原先没有说明多 gold 情况和宏平均方式。
- 建议给出集合口径：Hit@1 检查首项是否命中任一 gold；Recall@5 是 Top-5
  覆盖的 gold 比例；MRR 取首个 gold 的倒数排名，再在 covered episodes 上
  宏平均。

### 低：摘要中的粒度满分容易被误读为 Dynamic 独有

- SOP-only 在 stage-granularity track 同样达到 1.0000。
- 建议摘要明确该 track 只验证抽象层门禁，不证明 Tree 的额外价值。

### 可选代码整洁

- `_rank_debug_transition_rows` 已在缺失 failure signature 时直接返回，因此后续
  confidence 的 no-signature 分支不可达。它不影响论文公式或当前行为，可后续
  单独清理。

## 已验证无误

1. 排序公式与实现一致：
   `0.40 failure + 0.20 lexical + 0.20 task + 0.10 attachment + 0.10 geometry`。
2. 可信度公式与实现一致：
   `0.55 failure + 0.20 task + 0.15 attachment + 0.10 lexical`。
3. Flat-Twin geometry 不进入可信度。
4. 可信度低于 0.50 时回退 SOP-only；放行后 Tree 权重为
   `0.60 * confidence`，最高 0.60，SOP 至少 0.40。
5. 可信度被正确描述为 deterministic applicability score，而非校准成功概率。
6. 25 个 test episodes 的算术正确：SOP-only `13/25=0.52`；legacy Tree
   `4/25=0.16`；dynamic route `(8+12)/25=0.80`；dynamic selective
   `(7+12)/25=0.76`。
7. retrospective、非盲、silver labels、样本规模与无下游执行等限制披露充分。
8. 论文明确区分 retrieval diagnostics 与 downstream executed utility。
9. 旧固定 75% 方案已从生产方法主张中移除。

## 建议补充实验

### 投稿前优先

1. 披露固定 0.75 权重消融及其排名/回退 trade-off，不需要新增算力。
2. 扫描 0.40、0.45、0.50、0.55、0.60 的可信度门槛，报告 route、selective、
   fallback 和 MRR 稳定性。

### 后续增强

1. 冻结路由器，在未见 source runs 上做盲测并使用独立专家标签。
2. 运行 SOP-only、legacy Tree、dynamic Hybrid 的并发下游对照，固定模型、
   memory corpus、预算与至少三个 seeds。
3. 跨任务完成 static-clean、preservation-clean、runtime-clean 的 certified replay。
4. 做多轮 memory writeback 的污染与错误提升实验。

## 本轮采纳情况

- 已采纳：摘要补充 SOP-only ceiling；精确定义多 gold 指标；说明 0.50
  硬门的跳变与证据边界；附录披露固定 0.75 废弃消融。
- 暂未采纳：删除动态图 panel C。当前图仍可读，先保留机制与数字同图的汇报
  价值；投稿压页时可优先移除。
- 暂未修改生产代码的不可达分支，因为本轮范围是论文更新，且该分支不影响
  当前行为。

## 修订后二次复核

ClaudeAgent 再次只读核对修订后的正文、ledger、verifier 与 v2 JSON，确认四项
修复均准确完成，未引入新的事实错误或过度主张。唯一文字建议是把指标定义写成
“per-episode Hit@1”，已采纳。仍未解决的中等问题不是本文数字错误，而是待补
证据：0.50 阈值敏感度尚未运行、结构签名对全局部变量重命名仍有一项召回失败、
下游三臂对照与跨任务 certified replay 尚未完成。论文继续明确关闭这些主张。
