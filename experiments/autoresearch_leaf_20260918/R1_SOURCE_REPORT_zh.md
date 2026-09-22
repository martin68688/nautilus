# Leaf 历史 R1 来源与严格两小时边界

选定来源：`20260912_003640_leaf-r6-nomemory-luna-low-round1-2h-seed0`。
原目录名包含 `3h`，实际 `run-2h/control/LAUNCH.json` 的预算为 7200 秒。

- 起算：Unix 1789173084.9179494；截止：1789180284.9179494。
- 实际结束：Unix 1789178709.2408366，5624.32288666 秒，即 93.7387 分钟。
- 不补足剩余时间，不虚构后段测量；所有 14 个已完成节点均在截止前。
- 2 个 Host 认证结果：26.3514 分钟 0.7164004836110863；70.3514 分钟 0.44389385340454646。
- 11 条内部记录是固定训练阶段，不是 Agent 调参提出的 11 个独立方案。
- 1 个实际执行失败，11 个未执行的审查拒绝；均无成绩，不计作 0。

## 零记忆证据

原 LAUNCH 明确 `memory_enabled=false`、`memory_sources=[]`、`search_memory_mode=control`。
20 个实际规划请求均明确 NO historical task memory；`distilled_experience.cards=[]`、
`pack_sha256=null`、`source_run=null`。请求中的 incumbent 代码仅来自本轮已发生节点。

## 已验证来源与适配

原 NRP PVC 只读获取 185 MB 目录，全部普通文件逐一 SHA256 与源端匹配。
仅未复制指向 `/data` 的输入软链接，原数据已另行核验固定 manifest。
成功 checkpoint 的模型文件、预测和提交文件哈希与原 Host 回执逐一相符。

`cutoff_adapter.py` 显式把旧 TRAINING_EVENT 适配到技能要求的 SESSION/outcome 形状：
标注 `training_stage_not_agent_tuning`、`fresh_fit_training_stage`；不捏造参数搜索域。
规范化回执只补充由原 stage 记录导出的 trial ID/parameters，原回执字节保存在只读来源归档。
所有转换及源文件哈希见 `r1-cutoff-adapted-v001/CUTOFF_AUDIT.json`。

未改动技能的提取/构建/验证脚本；由本分析者依据原始节点和成功代码进行条件性综合，
没有教师 API 调用。18 张外层总结，29 张内层卡（18 总结+11 原始阶段）；14 节点全覆盖。
更改分数、删除节点、删除阶段的负测均被拒绝；47 个 reader 项逐字段保持完整。

## 可迁移观察与不可混淆事项

两个成功候选实际使用 CPU CatBoost、拟合集内中位数填补、192 数值特征；
N2 到 N7 的代码差异仅为 stage_schedule 从最高 1200 扩展至 2400。
每阶段都是全新拟合，不是增量续训。历史候选 random_seed=42，数据划分 seed=0。
后续新轮按自身协议 seed=0，不能把这项差异隐藏为相同模型种子。

历史 R1：Luna-low、旧 MLEvolve R6、NRP A10、8 CPU/32 GiB。
新 R2/R3：Terra-low、apizh-ai.com、新 autoresearch 单 Agent、AWS A10G、4 CPU/14 GiB/2 GiB shm。
因此这是历史起点上的累计记忆实验，不是严格的仅记忆因素因果对照。

R2 只用此 R1 新包；R3 只用同一 R1 和正式 R2 截止前的已完成轨迹。
旧 91 卡、旧 R2/R3/R4 后续成绩、工程 smoke 和真实 Agent probe 均不进入实验记忆。
