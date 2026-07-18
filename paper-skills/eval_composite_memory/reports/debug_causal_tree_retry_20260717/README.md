# Debug Causal-Tree Retrieval Retry

## Change under test

Debug retrieval no longer ranks successful RunNodes as generic history. It now:

1. extracts a failure signature from the current error;
2. retains only task- and stage-compatible `debug_fixed` transitions;
3. ranks the complete parent failure -> code change -> successful child transition;
4. projects only SOPs causally attached to that transition;
5. computes the Tree weight from transition confidence, with SOP-only fallback below 0.50;
6. caps the Debug Tree contribution at 0.60 instead of applying the configured 0.75 unconditionally.

The production prompt also expands the parent failure, proven code change, successful child result, and causally supported SOP IDs. The default runtime uses `flat_twin`; `F11` retains Poincare only as an ablation.

## Frozen offline benchmark

The frozen test split contains 120 episodes, of which 70 have silver relevance labels and are score-applicable.

| Condition | nDCG@10 | Adoption AP@10 | Debug nDCG@10 | Unsafe escapes |
|---|---:|---:|---:|---:|
| D2 Tree-only | 0.2371 | 0.1913 | 0.1647 | 0 |
| D3 SOP-only | 0.5222 | 0.4256 | 0.5027 | 0 |
| D6 Flat Hybrid | 0.5208 | 0.4248 | **0.5095** | 0 |
| F11 Poincare Hybrid | 0.5113 | 0.4192 | 0.4954 | 0 |

Relative to the prior D6 implementation, overall nDCG@10 increased from 0.4431 to 0.5208 (+0.0777), while Debug nDCG@10 increased from 0.3584 to 0.5095 (+0.1511).

D6 trails D3 by only -0.00141 mean nDCG@10. Across the 70 paired scored episodes, the win/tie/loss count is 31/16/23 and the 20,000-sample paired bootstrap 95% interval is [-0.0368, 0.0360]. This is not statistically significant evidence that either method is globally better.

## Routing behavior

Among 36 Debug episodes:

- 6 used causal Tree evidence;
- 30 fell back to SOP-only because no explicit compatible failure signature had sufficient causal Tree confidence;
- observed Tree weights ranged from 0.0000 to 0.3884, mean 0.0594;
- no unsafe candidate escaped the clean evidence gate.

The lower D2 Tree-only score is expected under the stricter contract: unrelated successful nodes are no longer used to fill the ranking. Tree-only therefore exposes evidence-coverage gaps, while Hybrid remains usable through its SOP fallback.

## Claim boundary

This result supports the narrow claim that failure-matched causal transitions and dynamic fallback remove the previous Debug degradation. It does not establish downstream MLEBench improvement, statistical superiority over SOP-only, or superiority of Poincare geometry. Labels are silver, blind annotation is incomplete, and only 70 episodes are scored.

Raw evidence is in `offline_test_report_v1.json` and `offline_test_receipts_v1.jsonl`.
