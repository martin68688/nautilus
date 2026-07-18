# RunForest Composite Benchmark: Terminal Report

## Verdict

**COMPLETED, STOPPED FAIL-CLOSED. All primary scientific claims remain closed.**

The preregistered independent replay-safety gate failed, so claim-bearing T4 was not started.
This is the required stopping behavior, not a missing positive result.

## Evidence

- Test decision episodes: `120`.
- Offline receipts: `2640`.
- Coverage gaps: `50`; complete rate `0.583`.
- Independent replay issue recall: `0.1250`.
- Independent replay pre-execution block rate: `0.1875`.
- T4 completed runs: `0`.

## Phase Status

- `phase_A_protocol_and_manifests`: `completed`
- `T0_integrity`: `completed`
- `T1_offline_retrieval`: `completed_diagnostic_only`
- `phase_0_gate`: `failed`
- `T2_agent_adoption`: `bounded_pilot_only_then_stopped`
- `T3_replay`: `bounded_generation_pilot_and_independent_safety_challenge_completed`
- `T4_micro_execution`: `not_started_by_preregistered_stop_rule`

## Required Before v2

- Add genuinely new clean L1/L2 evidence for uncovered task-family/stage cells and freeze snapshot v2.
- Use two independent blind annotators plus adjudication before relevance claims.
- Improve the detector without tuning on the frozen held-out challenge, then preregister a new challenge v2.
- Run full T2/T3 only after the new safety gate passes; run T4 only after all mechanism gates pass.

No superiority, universal safety, or downstream claim is licensed by this run.
