# Claude Code Audit — Codex Run-Forest Memory (online pilot + full evidence chain)

> 日期：2026-07-10  
> 范围：Codex 在 `codex/hyperbolic-structural-memory`（已改名为 `codex/dual-time-procedural-memory`）分支上的 Run-Forest memory 全部代码、评估、runtime 接入、online pilot、协作文档。  
> 方法：48 个并行 subagent 分 6 个维度（设计/评估完整性/runtime/300KB handoff/加固/一致性），每条结论对真实文件复验；Claude 对最严重的几条又亲自核验。被驳回的指控也记录。  
> **结论先行：工程不少地方扎实，但科学框架目前不诚实，且不可重复、不可部署。在重新诚实定性之前不应再投入 GPU。**

---

## 0. 一句话结论

所谓「Run-Forest Poincaré memory 赢了」细读到代码层后塌缩成：**在手画的环形树布局上、用 Poincaré 球距离比欧氏距离更能排祖先节点（parent_lookup +0.096 MRR，p≈0）——但这不是学习出来的双曲 embedding；同一个距离函数在找后代时是输的（debug_child −0.087，p=1.0）；离线赢的检索机制在 runtime 根本不存在（train/serve 完全脱节）；online pilot 没有任何下游提升。** 这不是一个能支撑「双曲 embedding」标题的结果。

---

## 1. Codex 做对的地方（先说公道话）

- **污染事件处理是对的**：r3 的 run-forest 图确实吃进了被隔离的 `20260512_112908`（0.0725 泄露 run），用户发现后，Codex 正确根因（builder 当时扫所有 journal、没 allowlist）并修好了——加 `--allowlist`/`--require-clean-provenance`、runtime `RunForestMemoryLayer` 拒绝 `leak_verified!=true || paper_grade!=true` 的图、config 关掉 contaminated methodology KB、tests 断言无 `20260512`。当前 `run_forest_graph.json` grep 任意隔离 run = 0 命中（Claude 已验）。
- **统计机器是干净的**：paired bootstrap 正确、BH-FDR / Holm-Bonferroni 多重比较校正做了、Flat-Twin 真的同坐标（`np.array_equal(poincare, flat_twin)=True`，evaluator 只读 `index["poincare"]`），所以 parent_lookup 的 +0.096 是纯距离函数效应。
- **诊断方向对**：Codex 自己的「停滞诊断」正确指出 online 瓶颈是 **actuation（生成器不照搬检索到的模板）而不是 retrieval**——这是全项目最重要的洞察。
- **树嵌入几何直觉对**：`assign_run_coords` 是标准 hyperbolic radial tree layout（角度=叶子跨度质心，半径=tanh(depth·edge_len/2)），parent_lookup 赢的方向和合成树复现（Poincaré corr 0.86 vs Euclidean 0.60）一致。

---

## 2. 三个 Blocker（Claude 亲自复核）

### Blocker A — 「Poincaré 坐标」不是学习出来的 embedding，是手画的布局

`build_run_forest_memory.py:331-365 assign_run_coords` 是**确定性闭式**的：半径只依赖树深度 `tanh(depth·0.82/2)`，角度只依赖叶子计数跨度。**没有 loss、没有梯度、没有任何学习。** 但 `hyperbolic_run_forest_memory_final_report.md:7,29` 引用 Nickel & Kiela 2017（一个**学习型** embedding 方法）来支撑「双曲 embedding 赢」的论点。

这是「已证明」和「已声称」之间最大的裂缝。能站住的真实表述是：**「在手画的 radial tree 布局上，Poincaré 球距离比弦欧氏距离更能排祖先」**——这比「双曲 embedding 是更好的表示」窄得多、新意小得多。

### Blocker B — 几何退化：41% 节点半径饱和塌陷到同一个值

半径公式 `min(tanh(level·0.82/2), tanh(5.1/2))`，深度≥6 全部饱和到 `tanh(2.55)=0.9879`。实测 **550/1346 = 41% 的 RunNode 卡在这一个半径**，另有 **28% 节点 6 位小数坐标完全相同**（单孩子链塌缩）。Poincaré 距离在边界处爆炸，所以大部分排序信号实际只落在角度一维上。

这同时解释了 mixed 结果：祖先任务赢（祖先半径小、Poincaré 放大径向分离），后代任务输（孩子和父母角度近、半径都被钉在边界，Poincaré 分不开）。reviewer 重算一遍就会判定 parent_lookup 的赢很脆弱。

### Blocker C — Train/serve 完全脱节（离线赢的机制 runtime 根本不存在）

- 离线 evaluator 用 **transductive 检索**：query = 某个节点**自己已算好的坐标**（`evaluate_run_forest_memory.py` ranks_for_query），在该坐标上排其它节点。
- live runtime 对一个生成上下文**根本没有坐标**，于是 `external_skill_memory.py:540-577,700-721` 用 **TF-IDF/SVD 文本方向 × 硬编码关键词带半径**（`'core':0.20,'middle':0.50,'edge':0.30`，靠子串如 `minimal_context` 命中）造一个 pseudo-anchor，几何只是对 lexical 命中结果的轻量重排。

也就是说，**那个 +0.096 离线赢用的 query 机制，runtime 从来不会发生**。离线赢对 live 检索质量几乎零证据。online pilot 的停滞（clean-r9：4 任务只跑到第 1 个、10/80 步、~2h22m、best 0.3553）才是唯一的 live 信号，而它 是 null-to-negative。handoff doc :8253 自己也承认 `RunForestMemoryLayer` 不消费 `geometry_query_radius_mode`（和 `agent_search.py:147` 的代码路径矛盾）。

---

## 3. 几个 Major

- **恒等式 claim gate 把输的任务标成 PASS**。`evaluate_run_forest_memory.py:205` 给 `local_best_graph_follow`/`debug_graph_expansion` 每个节点硬塞 `rr=1.0`，于是 `claim_gates` 报「3/3 passed」。真实距离任务是 **1 赢 / 1 null / 1 输**（parent +0.096 / local_best vs Euclidean p=0.21 / debug_child −0.087）。`local_best_pure_distance_context` gate 的 reason 文本自己都承认「Poincaré 没赢 Euclidean」，却标 passed:true。
- **provenance 是自证，不是审计**。`certify_skillgraph_provenance.py:126` 仅靠 allowlist **membership** 就盖 `leak_verified:true`，**完全没有 INDEX_BUG 检测逻辑**。allowlist 本身是从 shared_memory 散文誊抄的（`status=candidate_verified_from_coordination_shared_memory`）。r3 污染已证明这个戳会错。所以 `paper_grade_provenance:true` 是自章，不是审计。
- **online 无下游提升 + 无同期对照臂**。clean-r9 best 0.3553，不比污染的 r3（0.3697）好，没实现完整 ensemble；`summarize_runforest_online_matrix.py:110` 对比的是**历史 run**，不是同 Job 的 no-memory 对照臂。端到端 claim 是 null。
- **300KB handoff 系统性少披露**。0.096 / 0.087 / 0.028 在全文出现 **0 次**；debug_child 的输**从不作为结果披露**，只在 claim_gates 里被「化解」掉；Known Caveats（:4451）列了 API 成本/baseline 不新鲜/LLM fallback，却**漏了** gold 自播、allowlist 誊抄、debug_child 输、坐标非学习这四件最要命的事。2400 行的「Plateau/Retrieval Diagnostic」还把**已被删除的污染 run 的 0.369656 当「当前 live 状态」**反复引用，从不收回。
- **三份文档的 headline 数字互相打架**。final_report 引 +0.0864/+0.0321/−0.0555，当前 JSON 是 +0.0962/+0.0408/−0.0868，全对不上；parent_lookup 在三份文档里差 ~3x。
- **Euclidean 对照维度/特征空间不匹配**：`run_forest_euclidean` 是 16D TF-IDF 文本，poincare/flat_twin 是 2D 几何——「Poincaré 赢 Euclidean」把 geometry-vs-text 和 hyperbolic-vs-euclidean 两个变量混在一起。唯一干净的同空间对照只有 Poincaré vs Flat-Twin。

---

## 4. Claude 上一轮判断的更正

上一轮我（Claude）说 run-forest pivot「拿到了一个真阳性（+8.6 MRR）」、是「目前最强的证据」。**这个判断过于乐观，需要收回。** 经过本轮代码级审计：那个赢是 (a) 在手画布局上、(b) 41% 节点饱和的退化几何上、(c) 只对祖先任务、(d) runtime 复现不出来的离线机制。它真实但**窄且脆**，不能作「双曲 embedding 有效」的证据。我当时没把「坐标是确定性手画的、不是学习出来的」和「train/serve 脱节」这两点连起来——这是我的疏漏，本轮 workflow 补上了。

---

## 5. 唯一一句现在能站住的结论

> 在 agent 自己 run-forest 的**手画 radial 树布局**上，Poincaré 球距离比同坐标欧氏距离更能排祖先节点（parent_lookup +0.096 MRR, p≈0, n=1324）；但这是**固定布局上的距离函数比较**，不是学习型双曲 embedding；同一距离函数在后代检索上是输的（debug_child −0.087, p=1.0）、在 local_best 上对独立 Euclidean 是 null（p=0.21）；SOP-only 双曲 thesis 仍被证伪；且该离线赢**未被证明能带来任何下游 agent 指标提升**。

mixed 模式（祖先赢/后代输/local_best null/SOP null/online 停滞）有单一一致的几何解释：Poincaré 距离放大径向分离，所以找祖先（向内、半径小）占便宜，找后代（孩子角度近、半径被钉在边界）吃亏。**这是一个可发表的「带一个正面子结论的 negative result」——但前提是把框架从「Poincaré embeddings」改掉、删掉恒等式 gate。**

---

## 6. 建议（一条主线）

**在投入任何更多 GPU 之前，先做一次诚实重定性：**

1. **重写 final_report**：把坐标如实称为「确定性 radial layout」（引 `assign_run_coords` 代码），**删掉 Nickel & Kiela 2017 引用**（除非真的训了学习型变体），逐字引用真实数字（+0.0962 / −0.0868 / +0.0280）**包括 debug_child 的输**。
2. **删掉/重标** `local_best_graph_follow` 和 `debug_graph_expansion` 的恒等式 gate，让 `claim_gates` 只反映真实距离任务（1 赢 / 1 null / 1 输）。
3. **二选一**：(a) 真的跑一个**学习型双曲 embedding**（Riemannian SGD 在同一图上），做报告声称的那件事；或 (b) 把工作**改名为**「Poincaré 球距离 vs 欧氏距离在固定 radial 布局上」，停止 embedding 论点。
4. **建确定性 INDEX_BUG guard**（拖了几个月的「recommended next hardening」），让 `leak_verified:true` 是审计章而不是 membership 章。
5. online 要么加一个**同期 no-memory 对照臂**，要么承认 online 端到端 claim 暂时 null。

secondary：回收 0.369656 污染 metric；统一三份文档的数字；把 18MB `run_forest_graph.json` 移出 git history；人工审 gold（SOP benchmark 仍自播循环）；跑通 run-forest 测试套件（当前 3/8 fail）。

**核心**：当前 online pilot **救不了**一个离线结果框架的问题。先修框架，再谈算力。

---

## 附：可自测核验命令

```bash
# Blocker A：坐标是否学习（assign_run_coords 是确定性闭式）
sed -n '331,365p' paper-skills/hyper_memory/build_run_forest_memory.py

# Blocker B：半径饱和比例
python3 -c "import numpy as np; d=np.load('paper-skills/hyper_memory/run_forest_index.npz'); import collections; r=d['radius']; print('unique 6dp radii:', len(set(round(float(x),6) for x in r))); print('at saturating 0.9879:', sum(1 for x in r if abs(float(x)-0.987885)>1e-4), '/', len(r))"

# Blocker C：train/serve — runtime query anchor vs offline node-owned coord
grep -nE "geometry_query_radius_mode|query_direction|predicted_distribution|pseudo" mlevolve/agents/memory/external_skill_memory.py | head
# 恒等式 gate：
grep -n "rr.*1.0\|rank.*1" paper-skills/hyper_memory/evaluate_run_forest_memory.py | head

# 真实距离任务结果（1 赢 / 1 null / 1 输）
python3 -c "import json; d=json.load(open('paper-skills/eval_skill_memory/reports/run_forest_memory_evaluation.json'))['comparisons']; [print(k, '->', {kk:round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items() if kk in ('observed_mean_diff','p_value_one_sided_left_gt_right','n_pairs')}) for k,v in d.items() if 'flat_twin' in k and 'mrr' in k]"

# provenance 是否有 INDEX_BUG 检测
grep -nE "INDEX_BUG|index_bug|detect|leak_verified" paper-skills/eval_skill_memory/certify_skillgraph_provenance.py
```
