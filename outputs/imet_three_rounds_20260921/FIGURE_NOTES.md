# iMet 三轮曲线（与 Leaf 图同一方法）

生成脚本：`plot_imet_three_rounds.py`（Python/matplotlib，`/usr/bin/python3`，mpl 3.9.0）
输出：`imet_three_rounds.{png,svg,pdf}`、`source_data_imet.csv`、`summary_imet.json`

## 核心结论

三轮 Host micro-F1 单调上升，且**最佳解出现得越来越早**：

| 轮次 | 记忆范围 | 最佳 Host micro-F1 | 最佳出现时刻 | Host 认证/注册 |
|---|---|---|---|---|
| R1 | 无记忆 | 0.38917 | 98.6 min | 1/3 |
| R2 | 仅 R1 记忆 | 0.44095 | 85.1 min | 2/3 |
| R3 | 累计 R1+R2 记忆 | 0.45859 | 70.5 min | 1/2 |

R1→R2 +0.05178，R2→R3 +0.01764；最佳解到达时间 98.6 → 85.1 → 70.5 min。

## 与 Leaf 图方法的差异

Leaf 图指标是 log-loss（越低越好），阶梯线取**累计最小**。iMet 指标是 3474 属性多标签的
host micro-F1（越高越好），因此阶梯线取**累计最大**，纵轴箭头相应改为 ↑。其余（时间轴含
初始化/规划/训练/评分、rug 泳道、失败候选记 ×、底部汇总卡片）与 Leaf 脚本一致。

## 关于"内部 tune 节点"

iMet 三轮的 Host 认证节点很少（1/2/1，共 4 个），所以按要求加入内部 tune 节点作为第二层：
虚点线 + 实心点，取每个候选**阈值校准后**的 tune-split micro-F1（`worker/artifacts/` 下
R1、R2 为 `metadata.json`，R3 为 `meta.json` 的顶层 `tune_micro_f1`）。

这一层额外补进了两个 Host 评分阶段失败、但内部训练完整的候选：

- `imet-r1-20260920-a1-002`：tune 0.30211，predict1 阶段 rc=137 被杀
- `imet-r3-20260920-a1-001`：tune 0.44040，predict2 阶段 rc=137 被杀

tune 与 Host 分数几乎重合（例如 R3 a-000：tune 0.45849 vs Host 0.45859），说明阈值校准
没有对 tune 划分过拟合。

### 一个没有采用的做法，及原因

`metadata.json` / `meta.json` 里还有逐 epoch 的 `trajectory`，合计约 50 个点，密度接近
Leaf 图。但**三轮定义不一致，不能直接混用**：

- R1 记录的是 `val_micro_f1_at_0.15`（固定阈值 0.15 的训练期监控值），末轮 0.21556，
  而该候选真实校准后的 tune 分是 0.38823；
- R2 / R3 记录的是逐 epoch **校准后**的 `tune_micro_f1`，与最终 tune 分同尺度。

若把三者画在同一纵轴上，R1 会被系统性压低约 0.17，凭空造出"记忆带来巨大收益"的假象。
R1 的逐 epoch 校准值无法补算（只保存了最终 checkpoint）。因此本图只用每个候选的
**最终校准 tune 分**——这是三轮唯一同定义、可比的内部量。

如果需要逐 epoch 的密集曲线，应另做一张 inner-overlay 图，并且只在 R2/R3 之间比较、
R1 单独标注，不与本图混排。

## 读图约定

- 实线阶梯 = Host 认证候选的累计最佳；空心点 = 真实 Host 分数
- 虚点线 + 实心点 = 内部 tune 分数（未经 Host 认证）
- × = 候选生成或训练失败
- **纵轴未从 0 起**（范围 0.285–0.475），以便看清轮次差异
- 认证候选是顺序搜索观测，不是独立重复试验；本图不做显著性或因果记忆结论
- 非 Kaggle 官方成绩

## 数据来源（只读，未改动 iMet 运行目录）

- R1 `experiments/autoresearch_imet_plant_20260920_v001/imet/runs/imet-r1-20260920-a1`
- R2 `experiments/autoresearch_imet_plant_20260920_v001/imet/runs/imet-r2-20260920-b1`
- R3 `experiments/autoresearch_imet_plant_20260920_v001/imet/remote-recovery/r3-scored-artifacts/imet-r3-20260920-a1`

三轮共用：`gpt-5.6-terra` / effort low / 每轮 7200s；
data manifest `ecd6bcb1db4043f2ef4c225bcdb732ba1aca2bc61291e7274c5f94d00172d5ed`，
runtime manifest `60de8e9480c6dd3133248d46704532c8887a70507e9cf888388d068dbc2d7bbc`（三轮一致）。
各文件 SHA256 见 `summary_imet.json` 的 `source_hashes`。

脚本内置断言：注册数 [3,3,2]、认证数 [1,2,1]、每轮最佳必须等于该轮 `RUN_OUTCOME.json`
的 `best_search`；来源目录一旦被替换，脚本会直接报错而不是画出错误数字。

---

# 第二张图：加上内部 tune 轨迹面板

脚本：`plot_imet_three_rounds_tune.py`
输出：`imet_three_rounds_tune.{png,svg,pdf}`、`source_epochs_imet.csv`、`summary_imet_tune.json`

按"在主图下面再加一条内部 tune 轨迹面板、用较浅颜色区分、不与 Host 混成一条曲线"的做法实现。

## 内部 tune 节点清单（共 50 个，不是 30 个）

| 候选 | epoch 数 | 逐 epoch 字段 | 末 epoch | 最终校准 tune | Host |
|---|---|---|---|---|---|
| R1-000 | — | 无 metadata（16s 即失败） | — | — | — |
| R1-001 | 10 | `val_micro_f1_at_0.15` | 0.21556 | 0.38823 | 0.38917 有效 |
| R1-002 | 2 | `val_micro_f1_at_0.15` | 0.15764 | 0.30211 | predict1 rc=137 |
| R2-000 | 14 | `tune_micro_f1` | 0.43860 | 0.43860 | 0.44095 有效 |
| R2-001 | 4 | `tune_micro_f1` | 0.39477 | 0.39477 | 0.39592 有效但被淘汰 |
| R2-002 | — | 无 metadata（无 stage） | — | — | — |
| R3-000 | 12 | `tune_micro_f1` | 0.45849 | 0.45849 | 0.45859 有效 |
| R3-001 | 8 | `tune_micro_f1` | 0.44040 | 0.44040 | predict2 rc=137 |

合计 12 + 18 + 20 = **50**。

## 两处需要更正的说法

1. **R3 本地并非"只有终态收据"**。R3 两个候选都有完整逐 epoch 轨迹，只是文件名是
   `worker/artifacts/meta.json`，而 R1/R2 是 `metadata.json`；按后者去找会漏掉 20 个节点。
   本脚本两种文件名都读。

2. **R1 的逐 epoch 值与 R2/R3 不同定义，不能同轴直接比较**。R1 记的是训练期固定阈值 0.15
   的监控值，R2/R3 记的是逐 epoch 校准值。若直接混画，R1 会被系统性压低约 0.17，凭空造出
   "记忆带来巨大收益"的假象。

图中的处理：R1 轨迹用点线、单独图例项，并用虚线箭头把"训练后阈值校准"的跃升
（0.21556 → 0.38823、0.15764 → 0.30211）显式画出来，而不是隐藏这个尺度差。

## 其他限制

- **epoch 时刻是等分推算**，不是实测时间戳：`train.log` 为空，没有逐 epoch 打点，
  只能把每个候选的 epoch 均匀铺在其 train 阶段内。横轴上 epoch 的**相对次序**可信，
  **绝对时刻**是近似。
- 两个 panel 分属不同指标（Host 复算 vs 候选自报 tune），纵轴均未从 0 起。
- 下 panel 的空心点 = 该候选最终校准 tune 分；R2/R3 的空心点与其轨迹末点重合，
  R1 的空心点明显高于末点，即上面说的校准跃升。

脚本断言：每轮 epoch 节点数 [12,18,20]、合计 50、R1 字段必须是 `val_micro_f1_at_0.15`
而 R2/R3 必须是 `tune_micro_f1`。来源一旦变动会直接报错。
