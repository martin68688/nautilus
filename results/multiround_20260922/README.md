# 多轮记忆实验结果总览（截至 2026-09-22）

这份目录汇总了目前在两个 agent harness 上跑出的多轮记忆实验，对应 Overleaf 论文新增的
`\subsection{Multi-round memory accumulation on two agent harnesses}`
（`paper/overleaf_decision_admissibility/sections/multiround_two_harness.tex`）。

## 两个 harness

| Harness | 是什么 | 代码位置 |
|---|---|---|
| **MLEvolve** | 多 agent 搜索框架（planner / coder / reviewer + 内层 AgentTuner），冻结的 v7 runtime | `mlevolve/` |
| **autoresearch** | 单 agent 研究循环（移植自 `ruz048/autoresearch` @ `1e98497`）：agent 改候选训练脚本，host 运行后决定保留或丢弃；没有 MLEvolve 的 searcher / reviewer / 内层 agent | `experiments/autoresearch_imet_plant_20260920_v001/common/`（`harness.py`、`memory_bridge.py`、`memory_decision_validation.py`） |

两者只共享记忆包格式（`mlevolve_search_experience_v1` / `inner_tuning_memory_v1`）和执行容器。

## 协议（所有实验共同）

- R1 不读任何记忆；之后每轮只能读前面轮次**蒸馏出的记忆卡**，不读代码、权重或任务包。
- 每轮 7200 s 预算，单节点上限 3600 s，单 seed，一个固定内部划分。
- **Host 分数**：host 从已验证 checkpoint 在 agent 看不到标签的 search 划分上复算。
  **tune 分数**：候选在 agent 可见的 tune 划分上自报。
- **都不是 Kaggle 官方成绩**（所有 LAUNCH 均为 `official_submission: false`）。

## 结果汇总

| Harness | 任务（指标） | 轮次：记忆 → 最佳成绩 |
|---|---|---|
| MLEvolve | Leaf（Host search log-loss ↓） | R1 无 0.443894 · R2 读R1 0.442665 · R3 读R1+R2 0.066774 · R5 读91卡累计包 0.042636 · R6 同R5重复 0.035167 |
| MLEvolve | Spooky（Host search log-loss ↓） | A 无 0.35620 · C 读A 0.31031 · D 读A+B+C 0.27670 · E 读A+B+C+D 0.25928 |
| MLEvolve | iMet 2020（Host micro-F1 ↑） | R1 无 0.42416 · R2 读R1 0.50491 · R3 读R1+R2 0.51130 · R4 读R1–R3 0.58865 |
| MLEvolve | Plant Pathology 2021（Host micro-F1 ↑） | R1 无 0.83857 · R2 0.85573 · R3 0.86316 · R4 重跑 0.86077 |
| autoresearch | iMet 2020（tune micro-F1 ↑） | R1 无 0.38823 · R2 读R1 0.43860 · R3 读R1+R2 0.45849（Host 复算 0.38917 / 0.44095 / 0.45859） |
| 混合* | Spooky（Host search log-loss ↓） | R1(MLEvolve) 0.35028 · R2(AR) 读R1 0.35111 · R3(AR) 读R1+R2 0.34911 · R2 重跑(AR) 0.35625 |
| 混合* | Leaf（Host search log-loss ↓） | R1(MLEvolve) 0.44389 · R2(AR) 读R1 0.06357 · R3(MLEvolve) 0.06677 |

\* 混合序列：只有标 AR 的轮次跑在 autoresearch 上，MLEvolve 轮次是记忆来源或参照。
Leaf 混合图里的 R3 读的是**原始 MLEvolve R2**，不是图上画的 autoresearch R2。
Spooky "R2 重跑" 读的是一个新的 autoresearch 无记忆 R1，并在约 60 分钟时因 provider 403 中断。

## 图 → 源数据对照

`figures/` 里是上传到 Overleaf 的图（中文工作版，宽 2400 px）。原始导出与脚本、CSV 在原目录：

| 论文图 | 文件 | 原目录（脚本 / 源数据） |
|---|---|---|
| MLEvolve Leaf | `mle_leaf_search_rounds.png` | `outputs/leaf_five_selected_rounds/`（`plot_aligned.py`、`aligned_search.csv`、`aligned_summary.json`、`*_QA.md`） |
| MLEvolve Spooky A/C/D/E | `mle_spooky_acde_inner.png` | `artifacts/spooky_acde_resumed_20260915/`（`plot_clean.py`、`inner_trials.csv`、`host_scores.csv`、`QA.md`） |
| MLEvolve iMet 四轮 | `mle_imet_four_rounds.png` | `outputs/imet_four_rounds_20260918/`（`plot_results.py`、`source_data.csv`、`summary.json`、`QA.md`） |
| MLEvolve Plant Pathology | `mle_pp21_four_rounds.png` | `outputs/pp21_four_rounds_reference_style/`（`plot_clean.py`、`source_data.csv`、`run_summary.json`、`QA.md`） |
| autoresearch iMet 三轮 | `ar_imet_three_rounds_tune.png` | `outputs/imet_three_rounds_20260921/`（`plot_imet_tune_staircase.py`、`source_tune_nodes_imet.csv`、`FIGURE_NOTES.md`） |
| 混合 Leaf 三轮 | `ar_leaf_three_rounds.png` | `outputs/leaf_three_rounds_r2_replaced_20260920/`（`plot_curves_clean.py`、`source_data_replaced.csv`） |
| 混合 Spooky R1/R2/R3 | `ar_spooky_rounds.png` | `artifacts/autoresearch_rounds_20260919/`（`plot_spooky_with_current_r2.py`、`tune_observations.csv`、`EXCLUSIONS.json`、`FIGURE_NOTES.md`） |

autoresearch 各轮的协议、启动与结果记录：
`experiments/autoresearch_imet_plant_20260920_v001/`（iMet，含 `COORDINATION.json`、`distill/`、`imet/`）、
`experiments/autoresearch_spooky_20260918/`、`experiments/autoresearch_r2_spooky_20260919/`、
`experiments/autoresearch_leaf_20260918/`、`experiments/leaf_r2_autoresearch_rerun_20260920/`、
`experiments/autoresearch_migration_20260918/`。

## 必须一起说明的限制

这些是单 seed 的描述性轨迹，不是受控消融。与记忆同时变化的因素：

1. **模型**：MLEvolve 的 iMet、Spooky 链中 R1/A 用 Luna，后续用 Terra；Leaf 跨 harness 也换了模型。
2. **中断续跑**：Leaf R5/R6、Spooky A/C、iMet R4 在 provider 故障后换了模型 / provider / GPU 主机续跑；部分运行由 watchdog 结束。
3. **runtime 补丁**：iMet R3 起加入 recipe contract；Plant Pathology R1 用墙钟预算，R2–R4 用有效时间（扣除 LLM 等待）。
4. **有筛选**：图中未画 Leaf R4 与原始 Leaf R2、Spooky B、Plant Pathology 第一次 R4（0.85489）、若干较低的 iMet R2 尝试。
5. **Leaf 反证**：一个 autoresearch 无记忆 Leaf R1 重跑在中途快照已达 0.0168，所以 Leaf 的差距不能单归因于记忆。
6. MLEvolve iMet 与 autoresearch iMet 的数据划分不同（data manifest 不同），绝对分数不可横向比较。

下一步需要：固定模型与 runtime、同 harness 内的无记忆对照、多 seed 重复。

## 本分支未包含的内容

原始数据、权重、运行日志、AWS/NRP 主机相关文件（含实例 ID、IP、账号信息的文件已排除）、
以及单个超过 3 MB 的图片。这些仍保留在本地工作区。

## 编译论文

```bash
cd paper/overleaf_decision_admissibility
tectonic main.tex
```

`main.tex` 是 Overleaf 的 Main document，它 `\input` 了 `main_wp8_final_20260724.tex`，
新章节通过其中的 `\input{sections/multiround_two_harness.tex}` 引入。
本目录与 Overleaf 项目 `decision_admissibility_overleaf_20260720` 在 2026-09-22 同步一致。
