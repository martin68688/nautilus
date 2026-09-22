# Spooky 累计记忆协议（替代未启动的旧 A/B）

- R1：既有 20260910_011048_spooky-free-linear-r1-fixed-terra-low-seed0。无记忆、仅 novel_exploration，架构不限；linear 是搜索拓扑。原始启动 1789002547.6623847，截止 1789009747.6623847。只纳入截至该时已完成 N1–16；N17–30 全排除。前缀最佳 Host log-loss 0.35028213509959194。
- 原运行曾获19800秒预算，此处是事后截取7200墙钟秒。不能称作原本按两小时规划的实验。
- R2：全新任务模型初始化；只读取上述R1蒸馏；autoresearch单Agent候选编辑/训练/Host保留或丢弃；Terra-low，https://apizh-ai.com/v1；7200秒包含Agent、下载、训练、重载与评分。
- R3：R2终态后，用相同技能完整蒸馏R2，并合并已冻结R1；仅R1+R2。另起新模型，7200秒，R3后不再自动新增轮次。
- R2/R3：AWS A10G单卡，候选7CPU/28Gi。R1 NRP A10、原MLEvolve框架；框架/资源与事后截取规划差异须保留，不单独将成绩变化归因于记忆。
- 运行不拷贝历史候选代码、任务训练权重或预测；公共预训练权重可以独立下载。没有强制复现历史最佳或强制某种模型。
- R1适配：41个legacy publication是脚本阶段，不是互动Agent trial；完整86条telemetry保留，OOF fold误标的tune已标记其实际角色。四个执行前review拒绝保留为审查证据，不当作实际OOM/训练失败。N16较低tune未成为Host选定结果，保持区别。
- 新包：31 search摘要；158 inner卡=同31摘要+127原始记录。旧225卡和旧A/B目录不进入新run。
- 独立GPU/保存/两次重载/Host复算/失败拒绝预检已通过；实际记忆检索与决策需在指定relay再通过，正式分数另行记录。

证据：r1-cutoff-v1/CUTOFF_AUDIT.json、AUDIT.json、distillation/SYNTHESIS.json、SOURCE_MANIFEST.json、memory/BINDING.json；脚本使用原技能extract/build/verify，adapter另行留档。
