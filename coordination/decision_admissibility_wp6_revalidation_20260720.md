# WP6 Clean Replay / certified-memory Revalidation

日期：2026-07-20  
基于：WP5 v1.1 Result/Adoption/Causal 增量实现  
状态：**PASSED**

## 复验范围

本次没有重新解释或升级旧历史分数；只验证 WP5 新增的 typed positive derived
Claim、Authority snapshot 和 Bundle staging 接线不会削弱 Clean Replay 边界：

- method-preserved replay 仍只创建新 support path；
- method-changing replay 仍创建 Successor Claim，旧 Claim 不恢复；
- untrusted/cross-artifact/human-review material 仍 fail closed；
- 未 replay 的历史 score 仍不可 Rank/Promote；
- Positive Result/Adopted derived Method Claim 仅从当前 typed writeback path 产生，不能反向给历史 Claim 授权。

## 证据

```text
tests/authority/test_method_preserving_replay.py       通过
tests/authority/test_method_changing_fake_replay.py    通过
tests/authority/test_replay_successor_claim.py        通过
tests/authority/test_replay_authority_recovery.py     通过
focused total                                           17 passed
```

联合回归：

```text
tests/authority                                  170 passed
计划 §20.1                                      369 passed
完整 suite（排除冻结 composite benchmark）         512 passed
compileall mlevolve paper-skills tests            passed
```

冻结 composite benchmark 仍为既存 `18 passed, 1 failed` detector-lock hash
不一致，未修改其 lock。

## Stop Gate

- [x] method-preserved replay 可建立新 support path；
- [x] method-changing replay 不恢复旧 Claim；
- [x] 未 replay 历史 score 仍无法 Rank/Promote；
- [x] ProtocolSpec repair surface 仍由 immutable v2 声明，v1 未被静默扩宽；
- [x] WP5 typed positive derived Claim 不改变 Clean Replay 的 lineage/authority 语义；
- [x] parent Base、旧 Claim、用户 dirty assets 未被修改或覆盖。

**WP6 Stop Gate：PASSED。允许准备 corrected WP7，但 WP8 仍禁止。**
