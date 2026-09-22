> 历史初始审计：本文件中的 A/B、模型分配和“尚无训练”描述已被后续执行取代。当前协议见 PROTOCOL_V2_zh.md，实际运行状态见 AWS_COORDINATION.json。审计发现仍作为固定上游版本的证据。

# autoresearch 迁移评估与验证协议

结论：可以移植记忆系统，并用 autoresearch 风格的单层修改—执行—比较循环继续 Spooky / Leaf；不能原封不动替换 MLEvolve。当前只完成源码审计和记忆传输原型验证，没有新训练成绩，不能宣称记忆提高成绩。

## 固定审计对象

- https://github.com/ruz048/autoresearch ，commit `1e9849798006a069d17399f0d4ad0178200ee946`。源码快照见 `upstream/`，文件哈希见 `AUDIT_INPUTS.json`。
- 原任务 `spooky`：`01a088ac-054e-7532-9e9f-a2f9f0240623`。
- 原任务 `leaf继续`：`01a08d40-e47a-7411-bdf3-6d72094cd12a`。该任务后续讨论过 Hotel / Insults；默认迁移原来的 Leaf，用户澄清优先。
- 历史 Spooky 的稳定 E 轮为 Terra-low、A10、7200 秒。Leaf 近期原始 R2 是 Luna-low、A10、7200 秒。两任务无需用同一个模型，但每个任务自己的有/无记忆对照必须同模型同 effort。

## 能保留什么，必须改什么

| 部分 | 迁移方式 |
|---|---|
| 记忆蒸馏 | 保留确定性证据抽取、分析者条件总结、architecture/tuning/debug、完整原始轨迹、来源哈希；已有包优先原样复用 |
| 记忆读取 | Agent 提出短字面查询，工具返回完整卡片；在真正的代码修改请求中保留配方、参数域、initial/selected、trial trajectory、continuation、反例与限制 |
| 记忆决策 | 记录实际读取 ID、采用/适配/拒绝理由，再对照代码 diff 和执行参数；文件存在不算采用 |
| 搜索结构 | autoresearch 单层循环；不照搬 MLEvolve 多轮 proposer/judge/review，也不声称内外层算法保持一致 |
| 任务数据 | 继续使用原 fit/tune/search/test 文件、类别顺序、manifest、Host 隐藏标签，不重划数据 |
| 评分 | 将 nanochat `val_bpb` 换为任务原本的 Host log-loss，保存/重载预测、概率校验与 Host 复算保持有效 |
| Agent 模型 | 上游是 Claude Agent SDK，不能只改 `--model` 就认定支持现有 GPT CPA 路由；需要明确 provider adapter，并实测目标模型生成 |
| 时间 | 上游 300 秒训练预算不能替代原 2 小时端到端预算；Agent、下载、训练、验证均计时，保存每次成本 |
| 原始数据安全边界 | Agent 和候选不能访问 Host 标签、其它实验、凭据；上游 Read/Grep/Glob 本身没有路径白名单，候选 Python 也不受编辑工具保护 |

## 已复现的问题

`python3 verify_migration.py` 执行原始源码函数（未安装 SDK、未调用模型）：

1. 训练先输出 `val_bpb: 0.100000` 再退出 17，上游仍返回 `ok=True`。需拒绝非零退出和非有限指标。
2. `1.2e-3` 被解析为 `1.2`，超大数字成为 infinity，未被显式拒绝。使用有 schema 的 Host 回执，不能依赖候选日志正则评分。
3. 静态审计：`--no-setup` 路径把 `best_bpb` 重置 None，第一次成功提案会无条件成为 incumbent；必须从已验证持久状态恢复。
4. `run.log` 逐次覆盖，失败写成 0，缺乏跨轮记忆和完整训练证据。必须逐尝试归档，失败指标用 null；保留失败/丢弃代码、参数和输出。
5. `program.md` 没有被 `main.py` 显式加载，实际合同来自 `SYSTEM_APPEND` / `PROPOSE_PROMPT`。接记忆时要核对真实请求，不能只编辑 program.md。
6. 上游只杀直接训练进程，没有完整进程组回收；没有全会话端到端 deadline。适配时补齐。
7. 上游会 `git add -A`、`git reset --hard`，只能在全新隔离工作目录使用，绝不能在现有脏工作区运行。

## 已完成的记忆传输验证

`memory_bridge.py` 是独立的无损传输原型，不导入 MLEvolve。使用原文件，不重写蒸馏内容：

- Spooky：55 张 search + 170 张 inner 卡片，保留渠道身份，共 225 个条目。
- Leaf：91 张原搜索卡片，含三类经验和原来源元信息。
- 完整 JSON 往返相等；来源全部字段保留；错误包 hash 和错误数据 manifest 被拒绝；查询按字面匹配。
- 原型读取不会截断卡片，但正式接入必须选择相关完整卡片并记录上下文大小，不能把全包盲目塞进每次 prompt。
- Leaf 91 卡旧包含不同性质历史分支；这些分类必须保留。当前只验证可传输，不代表每条经验已重新做科学审核，也不默认它与 Spooky 蒸馏版本相同。
- 没有真实 Agent 的采用验证，没有训练，没有记忆效果数据。详见 `VERIFICATION.json`。

## 正式验证设计

每个任务在同一迁移版 harness 内做两组：A 不读取历史包，B 读取 hash 固定的同任务历史包。两组均保留各自本轮历史，均从相同新初始化/公共预训练权重起步，不带旧拟合权重或旧候选。相同 Agent/model/effort、种子、数据、资源、预算、依赖与候选起始合同。不能用旧 MLEvolve 无记忆结果直接对比新 autoresearch 有记忆结果来归因。

首轮可作为单种子配对试验，不能以一次胜负宣称统计显著。优先记录最终 Host 最佳值、到预注册阈值的端到端时间、有效候选数、错误类型与耗时、模型调用用量、每次实际采用的经验和相应可执行改动。公开结果中明确搜索指标不是独立最终泛化评估；反复使用同一 search 集可能形成适应性偏差。

先通过目标模型真实请求、读取/采用、最小真训练、保存重载、Host 复算、超时清理及断点恢复检查，再按原任务预算启动配对实验。预检不能强制重现历史最优，也不能要求记忆先提分才允许运行。

首阶段判断的是“旧 MLEvolve 蒸馏记忆能否帮助新 harness”；若要证明 autoresearch 自己跨轮积累有效，下一阶段需把新循环逐尝试证据映射到蒸馏输入合同，继续使用既有 distill.py extract/build/verify 与条件总结，另作多轮对照。
