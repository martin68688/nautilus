# RunForest Composite Benchmark v1

- Phase 0: **FAIL CLOSED**
- Mechanism claim allowed: **false**
- Downstream claim allowed: **false**
- Normal episodes: 70
- Coverage gaps: 50
- Frozen-fixture replay recall: 1.0
- Independent held-out replay recall: 0.125
- Independent held-out pre-execution block rate: 0.1875
- Evidence status: **DIAGNOSTIC ONLY** (silver labels, incomplete Agent/runtime tiers).

## Offline retrieval

F/P portfolio rows are retrieval-identical at T1 because no Agent generation occurs; they must not be used to infer portfolio effects.

| Condition | nDCG@10 | AP@10 | Unsafe escapes |
|---|---:|---:|---:|
| B1 | 0.2448 | 0.1572 | 0 |
| B2 | 0.2017 | 0.1515 | 0 |
| D1 | 0.1466 | 0.1149 | 0 |
| D2 | 0.3142 | 0.2416 | 0 |
| D3 | 0.5222 | 0.4256 | 0 |
| D4 | 0.3125 | 0.2424 | 0 |
| D5 | 0.3115 | 0.2132 | 0 |
| D6 | 0.4431 | 0.3508 | 0 |
| D7 | 0.3895 | 0.3206 | 24 |
| D8 | 0.3967 | 0.3262 | 0 |
| F00 | 0.2017 | 0.1515 | 0 |
| F01 | 0.4382 | 0.3519 | 0 |
| F10 | 0.2017 | 0.1515 | 0 |
| F11 | 0.4382 | 0.3519 | 0 |
| O1 | 1.0000 | 1.0000 | 0 |
| P0 | 0.4382 | 0.3519 | 0 |
| P1 | 0.4382 | 0.3519 | 0 |
| P2 | 0.4382 | 0.3519 | 0 |
| P3 | 0.4382 | 0.3519 | 0 |

## Closed gates

- `coverage_gap_count_zero`
- `two_blind_annotators`
- `krippendorff_alpha_at_least_0_67`
- `replay_heldout_expected_issue_recall_one`
- `replay_heldout_all_sources_blocked`
- `replay_eligible_tasks_below_8`
- `non_mock_agent_adoption_below_60`
- `five_stage_replay_repairs_not_completed`
- `T4_external_holdout_not_completed`

## Diagnostic interpretation

- Stage Hybrid - flat clean nDCG: `0.2365`.
- Stage Hybrid - SOP-only nDCG: `-0.0840`.
- Poincare - Flat-Twin nDCG: `-0.0049`.
- Disabling safety admitted `24` unsafe candidates.
- T1 cannot identify portfolio effects because no Agent generation occurs at this tier.

## Negative results that block positive interpretation

- **Stage Hybrid does not beat SOP-only:** the nDCG difference is `-0.0840`.
- **Poincare does not beat Flat-Twin:** the nDCG difference is `-0.0049`.
- **Static replay safety does not generalize on the independently authored challenge:** issue recall is `0.1250` and pre-execution block rate is `0.1875`. The frozen-fixture `1.0` is not an independent recall estimate.
- These are falsifying diagnostics, not evidence for the composite mechanism.

All silver-label and incomplete-runtime results are diagnostic only.
