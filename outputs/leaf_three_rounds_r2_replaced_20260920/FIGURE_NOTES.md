# Leaf 三轮曲线（最新 R2 替换版）

核心结论：新 R2 的最佳 Host search loss 为 0.06356954；图中 R1、R3 保留原记录，R2 完全替换为 2026-09-20 的新 autoresearch R2。

- 指标：固定 search 集 multiclass log-loss，越低越好。
- R1：14 个注册候选，2 个认证，最佳 0.44389385。
- R2：11 个注册候选，6 个认证，最佳 0.06356954；仅 R1 记忆（18 张 search 卡 + 29 张 inner 卡）。
- R3：11 个注册候选，10 个认证，最佳 0.06677362。
- 仅认证且有限的 Host 分数进入曲线；失败候选保留为 ×，不赋值为 0。
- 认证候选是顺序搜索观测，不是独立重复试验；图中不做显著性或因果记忆结论。

数据来源哈希：
- old source_data.csv: 9f3837d24b45dfe63694c278ba0724861a2670cf1d58bdf4f95ee9ca02c8bac2
- current R2 remote snapshot: d407284b90a3cf091161abce7d82ee57e0a22519ad1cb9b347590df5f212328f
- current R2 data manifest: 7bd8ce4887d28eea1bbe26730227ac4e49160332bd087571d7331110b5580a7f

后端：Python/matplotlib；输出为可编辑 SVG/PDF 和 300 dpi PNG 预览。
