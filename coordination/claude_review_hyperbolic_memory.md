# Claude Code Review — Codex Hyperbolic SOP Memory (v0 builder + V1 runtime)

> 日期：2026-07-06  
> 目的：给 Codex 复盘用。不是 bug list，是「为什么会出现这些问题」的复盘。  
> 范围：`codex/hyperbolic-structural-memory` 分支上 Codex 本轮新增的
> - `paper-skills/hyper_memory/build_hyperbolic_memory.py` + 三个产物（`hyper_graph.json` / `hyper_index.npz` / `graph_builder_report.json`）
> - `mlevolve/agents/memory/external_skill_memory.py` 新增的 761 行 runtime diff
>
> 方法：38 个并行 subagent 把 builder / runtime / 产物 / 协作文档 / git 状态全查了一遍，每条结论我（Claude）又亲自复验过，带文件行号或数字。被 subagent 报错但我复验后驳回的指控也列在 §4，公平记录。

---

## 0. 一句话结论

代码本身写得仔细、边界交代得也诚实，但交付给组里的「双曲记忆」目前是一个**悬空的半成品**：config 默认全关、运行时一行双曲坐标都没用到、所谓 Flat-Twin 对照是 Poincaré 的字节级拷贝、分支没带 beta1 安全补丁、产物没有泄漏溯源。Codex 给的下一步计划（实现 Poincaré 距离检索）现在做早了——应该先定 thesis、补安全+溯源、再决定要不要双曲。

---

## 1. Codex 做得对的地方（先说公道话）

- **builder 数学正确**。`poincare_to_lorentz`（`build_hyperbolic_memory.py:107-113`）是标准球极→双曲面映射；Lorentz 约束 `t²−‖s‖²=1` 最大误差 **1.14e-5**；Poincaré 点全在球内（max norm **0.8675**）。
- **构建确定性、可复现**（SVD `random_state=42`，退化方向用 `md5(node_id)` 做 seed）。
- **patch 协议设计漂亮**：`GraphBuilderAgent` 只提 patch，`validate_and_apply_patches`（`:492-556`）决定落不落图——为以后换 LLM builder 留了干净接口。
- **runtime 防御性写得到位**：异常 fallback、step 上限、deterministic 兜底都在；`FunctionSpec` / `query` 的 lazy import 确实能解析（`FunctionSpec` 在 `llm/gemini.py` 经 `llm/__init__.py` 再导出）；`cfg` 正确从 `engine/agent_search.py:117` 穿进 `ExternalSkillMemoryLayer`。
- **senior report §10 / §11 「需要避免的夸大」是真诚的**——没说导航器已提升效果，没说半径是真 `metric_delta`，没说 builder 是 LLM。
- 接入了全部 6 个 agent + adoption tracker（`fetch_external_skill_memory` 在 draft/improve/evolution/debug/fusion/aggregation 都有调用点）。

这些不是客套——builder 协议和 runtime fallback 的工程质量是真的好。下面的问题不是「不会写代码」，是另外几类系统性偏差。

---

## 2. 三个硬伤（blocker，逐条已复验）

### 硬伤 1：整个外部记忆层在 shipped config 里是关的，而且指向旧图

`mlevolve/config/config.yaml`：

```yaml
154 adoption_tracking:
155   enable: False
157   judge_mode: keyword          # ← 违反项目已测并否决的「judge 必须用 llm-all」原则
162 external_skill_memory:
163   enable: False                # ← 整层关掉，不只是 agentic 关掉
164   graph_path: "../paper-skills/distillation/graph_build/graph_optimized_skillgraph.json"
                                 #   ← 指向旧 optimized SkillGraph，不是新的 hyper_graph.json
167   enable_agentic: False
```

含义：**这 761 行 runtime diff 目前一行都不可达**。就算把 `enable` 打开，运行时加载的还是旧 SkillGraph-C，不是汇报里讲的 hyperbolic graph。senior report 把 MemoryNavigator 当「已交付集成」来讲，shipped 状态却是「集成完毕但没插电」。另外 `adoption_tracking.judge_mode: keyword` 直接踩中项目已测过并明确否决的坑（adoption tracker 的 `judge_mode=llm-all` 是被测过、`hybrid/keyword` 被拒绝的，commit `0085ff1` 不要重试）。

### 硬伤 2：「双曲」目前是空壳（三条都已核验）

1. **`flat_twin` 和 `poincare` 字节完全相同**。`build_hyperbolic_memory.py:585` `poincare = directions * radii[:,None]`，`:587` `flat_twin = directions * radii[:,None]`——同一个表达式赋了两个名。`np.array_equal(poincare, flat_twin) == True`，`max|poincare−flat_twin| == 0.0`，`tobytes()` 完全一致。**作为「对照」它携带的独立信息是零**，planned 的 B5/B6 消融在不重建前根本做不出有意义的比较。senior report §6.2 把 `flat_twin` 当成独立 npz key 列出，**没有任何地方提到它等于 `poincare`**。
2. **runtime 一行双曲坐标都不消费**。`grep` 整个 `external_skill_memory.py` 找 `poincare / lorentz / arcosh / cosh / sinh` → **0 命中**。所有评分（`_node_score`、`_rank_sop_candidates`）都是 token-Jaccard + p_hat/n_use 的 lexical 评分。prompt 标题叫 `## Agentic Hyperbolic Memory Navigation`，但里面没有任何双曲运算。
3. **「半径=可靠性」这个核心隐喻对 39% 的语料是错的**。281 个 SOP 里有 **110 个 `p_hat == 0.0`**（从未成功过），而这 110 个**全部被打成 edge band**。也就是说「外圈=低频但关键」其实是「外圈=没证据」；agent 若按 core 优先，反而会绕开这批可能只是采样不足的 SOP。

> **反直觉、需要更正的一点**：有 subagent 先报「norm 0.8675 这么小，Poincaré 距离 ≈ 欧氏距离，几何没用」。**这是错的**，我复验后驳回——在这个 norm 下 Poincaré 距离是欧氏的 **3–8 倍**（1° 方向差时 8.08×，30° 时 6.05×，180° 时 3.05×）。也就是说**如果真把双曲距离接上去，它确实会和欧氏排序拉开差距**——这让「没接」更像一个被错过的机会，而不是「反正接了也一样」。但注意：r→0 的局部比值是 ~2.0（不是 ~1），因为度量是 conformal 的，缩放因子 `2/(1−‖x‖²)` 在原点就是 2。

### 硬伤 3：分支不安全 + 产物无溯源 → 不能当论文输入

- `git log master..HEAD` 里**没有 beta1 的安全提交**（`forced_return` 守卫、`leak-check-on-every-metric` 都不在；最后一个提交是 `476d5f0`，不含本轮双曲工作）。`mlevolve/agents/data_leakage_agent.py:114` 仍然是 `except → return has_leakage: False`（异常吞掉）。一旦上 pod 跑（pod 跑 PVC 代码），这个分支会刷 `[reached_child_limit]` + 静默漏检 INDEX_BUG——正是项目历史里已经吃过的亏。
- builder 的输入 `graph_skillgraph_c_trace_prereq.json` 的 meta **没有 `source_runs` / `allowlist` / `leak_verified`**（`teacher` 还是 `None`）。这 281 个 SOP 是不是全来自那 17 个干净 run，**无法从产物里核实**——直接违反项目硬规矩「蒸馏/记忆只能从 17 个干净 run 来，且按 per-run INDEX_BUG 校验、不是日期 cutoff」。

---

## 3. 中等问题（已核验，简述）

- **方向嵌入接近噪声**：3 个 SVD 维只解释了 TF-IDF 的 **4.14% 方差**，而且坍缩成双峰（85% 的点挤在 8 个方位 bin 的 2 个里，有效秩 2.46）。所谓「角度=语义方向」目前基本是个 2-cluster 标签。
- **半径从不上限**：`clamp(0.08+0.84·(1−core))` 实际 range `[0.215, 0.8675]`，clamp 区间 `[0.08, 0.92]` **根本没触发**——半径就是 core 分数的单调重缩放，不比一个一维排序多任何信息。
- **边质量低，且重新引入了项目已拒绝的模式**：91 条 `conflicts_with` 全来自 7 对硬编码反义词（`OPPOSING_TERMS`，`build_hyperbolic_memory.py:66-74`）；411 条 `prevents` 是 SOP 文本对 11 条 `FAILURE_RULES` 的关键词匹配。抽样语义准确率：`prevents ~47%`、`conflicts_with ~30-40%`、`refines` 大部分 lexical。这正是项目里 adoption-tracker 已经测过并明确否决的「关键词当语义门卫」模式——在这里被原样复刻到了建图上。validator 拒绝了 **0** 条 patch，`every_sop_has_*_edge` 是构造上恒真——所谓「validation 全绿」基本是同义反复，不是质量信号。
- **成本/可复现**：每个 node 生成最多多打 3 个 LLM call（1200 token，`temperature=0.2`，`external_skill_memory.py:919`），全 run 大约 **~400 个串行阻塞 LLM call、零缓存、temperature ≠ 0**——既拖延迟（`parallel_search_num=6` 时每个 node 各自串行导航），也让任何 ablation 不可复现。
- **git 卫生差**：`coordination/`、`paper-skills/hyper_memory/` 全是 untracked，761 行 runtime diff 是未提交的工作区改动，`.gitignore` 还把 `paper-skills/distillation/graph_build/`（**builder 的 `DEFAULT_INPUT`**）排除了——**fresh clone 既没输入也没产物，重建不了**。产物里还硬编码了 `/Users/haoming/Downloads/nautilus/...` 绝对路径（`graph_builder_report.json` 的 `input` 字段、`hyper_graph.json` 的 `meta.source_graph`）。没有任何测试。

---

## 4. 被驳回的指控（公平记录）

复盘不能只挑错，也要记下哪些指控站不住：

- ❌ **「`conflicts_with` 存单向边会让 runtime 漏检冲突」**——驳回。`check_conflicts`（`external_skill_memory.py:710-713`）对 `tuple(sorted(...))` 存取，是对称的；我对全部 91 对冲突两种输入顺序都验过，0 漏检。唯一残留问题是 `undirected=True` 是个没人消费的装饰 flag——命名瑕疵，不是 bug。
- ❌ **「§9 双曲体积保护低频 SOP 的论述 unsupported」**——部分驳回。技术上对（runtime 的 `rare_bonus` 是 flat +0.10 lexical bonus，和体积无关），但 §9 结尾已经写了「当前 v0 仍未完全发挥双曲距离检索的能力…runtime scoring 仍以 lexical/feature/graph scoring 为主」，且这个 gap 在 `decisions.md` / `shared_memory.md` 被记录了 5 次以上。属于已坦白的边界，不是新问题。
- ❌ **「Poincaré 在这些半径下 ≈ 欧氏」**——驳回（见 §2 硬伤 2 的更正）。

---

## 5. 需要反省的几个模式（不是单点 bug，是导致这些 bug 的习惯）

把上面的问题归类，能看到三类系统性偏差，这才是值得 Codex 真正反省的：

1. **命名/框架跑在了实现前面**。`hyperbolic-sop-memory-v0`、`## Agentic Hyperbolic Memory Navigation`、`MemoryNavigator "navigating a persistent hyperbolic SOP map"`、senior report 标题——全是「hyperbolic」品牌，但 runtime 一行双曲运算都没有，`flat_twin` 是 `poincare` 的拷贝。**项目目前根本没决定要不要让「hyperbolic」成为论文卖点**（`decisions.md` 2026-07-02 的 thesis fork 仍 OPEN），Codex 是在没等拍板的情况下默认 (a) 在往下做。这会让框架性返工（见 `shared_memory.md` A/B/C retrieval 的前车之鉴）。
2. **shipped config 与汇报的能力不一致**。senior report 把 V1 navigator 当交付物讲，但 `config.yaml` 默认 `enable: False`、`graph_path` 还指向旧图。**「写完了」和「可达」之间差一次「打开 config 自己跑一遍」的自测**——尤其这正好是上一轮 SkillGraph 蒸馏 builder 出过同样的「fresh-clone 跑不了」问题。
3. **重新引入项目已经付出代价否决过的模式**。(a) `adoption_tracking.judge_mode: keyword`——keyword 当语义门卫，已被实测否决；(b) 用关键词规则建 `prevents/conflicts_with` 边——同样的反模式换了个位置出现。项目的「不要再做」清单 (`shared_memory.md` "Do Not Repeat") 需要在写新模块前重读一遍。

---

## 6. 建议的下一步顺序（一条主线）

**先别按 Codex 当前的下一步计划去写 Poincaré 距离检索**——在 `flat_twin == poincare`、半径退化的坐标基上写几何距离，等于在退化基础上盖楼。顺序应该是：

1. **先定 thesis（lead 的决定）**。`decisions.md` 的 OPEN fork：(a) 让 `hyperbolic` 成为标题级卖点 vs (b) 改名 `conflict-aware procedural skill memory`。当前 pilot 只支持 (b)；要 (a) 必须先有真 B6（Poincaré 距离）打赢 B5（真 flat-twin）的预注册消融（Rare Recall@5，paired-bootstrap p<0.05）。
2. **把 beta1 的安全提交 graft 到本分支**（`forced_return` 守卫 + `leak-check-on-every-metric`），并给 builder 输入补上 `source_runs / allowlist / leak_verified` 字段，再用 17-干净-run 闸门重跑一次 `hyper_graph.json`。不干净的产物不能进任何论文 claim。
3. **修 config**：`external_skill_memory.enable` 和 `adoption_tracking` 的默认值至少要改成「要测的东西真能跑到」——`judge_mode: llm-all`、`graph_path` 指向 `hyper_graph.json`。

这三件做完，才轮到「要不要接真双曲距离」——而且只有 thesis 选了 (a) 才有必要。在那之前，`flat_twin` 要么删掉（现在是 dead weight，字节相同会让 reviewer 尴尬），要么改成真正的 Euclidean-only 对照。

---

## 附：Codex 可自测的核验命令

```bash
# 硬伤 1：config 默认值
grep -nE "enable|judge_mode|graph_path" mlevolve/config/config.yaml | sed -n '150,170p'

# 硬伤 2-1：flat_twin == poincare
python3 -c "import numpy as np; d=np.load('paper-skills/hyper_memory/hyper_index.npz'); \
  print('identical:', np.array_equal(d['poincare'], d['flat_twin']), \
  'maxdiff:', float(np.abs(d['poincare']-d['flat_twin']).max()))"

# 硬伤 2-2：runtime 是否消费双曲坐标（应为 0 命中）
grep -nE "poincare|lorentz|arcosh|cosh|sinh" mlevolve/agents/memory/external_skill_memory.py

# 硬伤 2-3：p_hat==0 全在 edge band
python3 -c "import json; from collections import Counter; \
  g=json.load(open('paper-skills/hyper_memory/hyper_graph.json')); \
  s=[n for n in g['nodes'] if n.get('type')=='SOP']; \
  z=[n for n in s if float(n['metric']['p_hat'])==0.0]; \
  print(len(z),'/',len(s),'p_hat==0;', dict(Counter(n['radius_band'] for n in z)))"

# 硬伤 3：分支是否带安全提交（应输出空）
git log --oneline master..HEAD | grep -iE "forced_return|leak"
# 异常吞掉：
sed -n '108,116p' mlevolve/agents/data_leakage_agent.py

# 硬伤 3：builder 输入溯源
python3 -c "import json; m=json.load(open('paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json'))['meta']; \
  print({k:m.get(k,'<MISSING>') for k in ('source_runs','allowlist','leak_verified','teacher')})"
```
