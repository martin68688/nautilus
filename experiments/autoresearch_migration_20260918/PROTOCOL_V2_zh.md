# 累计记忆协议 v2

用户于本任务明确替换旧 A/B 协议。任务仍为 Spooky 与 Leaf，参考 iMet/Plant 的累计记忆迭代方法，不改变数据集。

- 新轮次均用 https://apizh-ai.com/v1 的 gpt-5.6-terra，reasoning_effort=low。
- R1 不重跑：选择各任务已有冷启动、novel、无历史记忆记录，仅取启动后 7200 秒内的完整证据。原实际不足 7200 秒则如实记录，不能补齐、插值或加入晚于 cutoff 的结果。
- R2 跑 7200 秒，只接收 R1 的蒸馏记忆。
- R3 跑 7200 秒，接收 R1 与 R2 的累计蒸馏记忆；结束 R3 即收尾。
- 使用 experiment-memory-distillation 原脚本 extract/build/verify。旧格式用可审计适配器，不把旧训练阶段冒称 Agent 调参试验。
- 记忆保留 architecture/tuning/debug、完整试验/失败和配方语义；新轮次不继承旧训练权重、候选代码或晚轮结果。
- 原 A/B 冻结目录保留为作废协议证据；原 225/91 张全历史包不是新 R2 的输入。
- 原模型/硬件/框架差异在 R1 来源中保留，不声称是纯记忆因果对照。
- 所有新轮次要求实际检索采纳检查、训练保存重载及 Host 复算。两台中转真实生成均已返回 HTTP200 与 gpt-5.6-terra。
- 完整终态后 EBS→Mac→版本化 PVC 校验，满足实时停机门后 stop，不 terminate，不删除卷。
