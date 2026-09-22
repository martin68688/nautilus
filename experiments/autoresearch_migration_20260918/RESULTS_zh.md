# Spooky / Leaf 累计记忆实验结果

两项实验均已结束第三轮，不再增加轮次。指标为 Host search logloss，越低越好；它是搜索过程中使用的评估集指标，并非独立最终测试成绩。

| 任务 | R1：历史无记忆 | R2：使用 R1 记忆 | R3：使用 R1+R2 记忆 |
|---|---:|---:|---:|
| Spooky | 0.3502821351 | 0.3511051102 | **0.3491117641** |
| Leaf | 0.4438938534 | 0.0682068513 | **0.0372787323** |

新轮统一使用 `https://apizh-ai.com/v1` 的 `gpt-5.6-terra`、`reasoning_effort=low`，AWS A10G。R2、R3 各有固定 7200 秒的原始截止时间；修复、网络失败和空闲不从预算中扣除，也没有通过重启重置预算。每轮重新训练，仅迁移记忆，没有继承已训练权重。

Spooky 第三轮有 22 次尝试、13 个有效结果。最终最佳为 `spooky-ar-r3-v1-021`，已经通过重载及 Host 复算，较 R1 略有改善。网关故障、中断、恢复和未成功尝试均保留。最终控制器记录经过 7152.69 秒，归档在原始截止时间之后执行。

Leaf 第三轮有 5 个 Host 有效结果，最佳为 `attempt-00005`。后期上游持续返回 503，旧重试逻辑产生大量失败记录，因此其 1214 条历史记录不能解释为 1214 次训练。02:48:10 UTC 停止故障循环，修复后的健康检查仍失败，剩余预算不满足恢复条件，原截止时间 03:15:16 UTC 到达后以 `provider_blocked` 终态收尾；不能声称连续完成了两小时有效训练。

R1 是历史记录截断：只纳入各自起点后 7200 秒以内完成的合法结果。Leaf 的来源实际运行约 93.74 分钟，旧模型为 Luna-low，旧框架、硬件资源和部分种子也有差异。Spooky R1 同为 Terra-low，但仍存在框架及资源差异。按用户协议未设置无记忆对照，因而结果证明了跨轮记忆接入和运行可行，不能把所有分数提升归因于记忆。

这次迁移复用了数据、Host 评估、模型重载验证、历史来源审计及记忆蒸馏流程。autoresearch 提供的主要是单个 Agent 持续提出修改、训练、评估及保留候选的循环；记忆通过受控 query/read 接口接入。实际运行的是经过适配和修复的 harness，并非上游仓库未经修改即可替代 MLEvolve。超时清理、失败分类、返回结构校验、记忆引用校正和退避熔断等工程修复都已记录。

归档与停机状态以同目录 `AWS_COORDINATION.json` 及 `return-20260919-v1/` 下的实时回执为准，不能仅凭此结果表推断实例已经停止。

## 关键证据

- Spooky：`../autoresearch_spooky_20260918/lifecycle-return-v1/bundle/FINAL_CLOSEOUT.json`
- Spooky 最佳：上述 bundle 内 `spooky-autoresearch-chain-r3-v1/code/runs/spooky-ar-r3-v1/attempt-021/HOST_RESULT.json`
- Leaf：`../autoresearch_leaf_20260918/lifecycle-return-v001/transport/R3_FINAL_TERMINAL.json`
- 协议：`PROTOCOL_V2_zh.md`
