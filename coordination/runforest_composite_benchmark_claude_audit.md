# RunForest Composite Benchmark Independent Audit

## Scope

Final read-only ClaudeAgent audit session:
`e26fd7f0-ca9f-479f-b6b2-c7ee72c4baab`.

ClaudeAgent inspected the frozen plan, builders, T1 receipts, claim scorer,
independently authored replay challenge, R0-R3 matrix, T2/T3 pilots, generated
reports, and regression tests. It verified files rather than relying on the
implementation summary.

## Verdict

- **P0:** none. Every scientific claim remains closed; no leakage,
  pseudo-ablation, score inflation, or route that lets Agent self-report open a
  runtime claim was found.
- **Framework:** T0/T1 and fail-closed T2/T3/T4 interfaces are implemented.
- **Scientific benchmark:** incomplete. T1 still uses silver labels with 50/120
  coverage gaps; T2 has one pilot episode; T3 has one pilot defect; T4 has no
  trusted execution.
- **Current evidence:** a falsifying preregistration, not a positive result.

## Independently Confirmed Results

1. Frozen-fixture replay recall/blocking is `1.0`, but this is a closed-loop
   fixture result and not an independent generalization estimate.
2. The detector-blind 16-case challenge is a distinct artifact. Its issue
   recall is `0.125` and its pre-execution blocking rate is `0.1875`; both are
   printed in the main report and close Phase 0.
3. R0-R3 are distinct. Agent-declared runtime provenance is recorded separately
   and can never set `runtime_provenance_verified_clean` or `clean_repair`.
4. The T2 first pass remains visible: `1/4` completed. Retries produced four
   candidates, but the pilot claim stays false.
5. T1 negative results are not hidden: Stage Hybrid is below SOP-only by
   `-0.0840`, and Poincare is below Flat-Twin by `-0.0049`.

## Claude Findings And Disposition

| Severity | Finding | Disposition |
|---|---|---|
| P1 | T2 pilot consumed one test episode before blind freeze | Accepted. Report now records `test_split_consumed_by_pilot=true`; future runner defaults to dev and requires explicit `--confirm-frozen-test`. The consumed episode cannot support the eventual frozen test. |
| P1 | Pilot could silently drop a key that never succeeds | Fixed. Report now emits `never_successful_count`, per-key attempts, and failure reasons. |
| P1 | R0-R3 evidence is only 1/48 cases | Open blocker. No comparative R3 claim is allowed. |
| P1 | 50/120 T1 episodes are coverage gaps | Open blocker. Memory must expand or scope must be preregistered narrower; no detail SOP backfill is allowed. |
| P2 | Portfolio rows are retrieval-identical at T1 and easy to misquote | Fixed in presentation. Main report now warns immediately above the table that T1 cannot identify portfolio effects. |
| P2 | Provenance completeness ignored token fields | Fixed. Non-null input/output token counts are now required. |
| P2 | Missing artifacts could soft-skip tests | Fixed for canonical T1 and pilot artifacts; tests now fail when those deliverables disappear. |
| P2 | Held-out tuning prohibition was process-only metadata | Strengthened. `replay_heldout_lock_v1.json` freezes both challenge SHA and detector-source SHA; v1 refuses reevaluation after either changes. |

## Required Next Actions

1. Obtain two independent blind annotations and adjudication; require ordinal
   Krippendorff alpha at least `0.67`.
2. Resolve the 50 coverage gaps without substituting wrong-abstraction memories.
3. Re-freeze the T2 test set because one episode was consumed by the pilot.
4. Run all 48 defect cases across R1-R3 and add evaluator-owned isolated runtime
   provenance.
5. Do not run claim-bearing T4 while the independent replay safety gate fails.
   Any next detector version needs a newly authored detector-blind v2 challenge.

## Validation

The relevant benchmark, retrieval, ablation, stage-hybrid, and leakage suites
pass in separate processes: `84 passed`. Separate execution avoids the local
NumPy/Pandas ABI crash observed when all suites share one interpreter; no test
assertion failed.

## Terminal Ledger Re-Audit

Final terminal-ledger review session:
`f2633956-509a-4e1e-947a-8a065b7738c0`.

After the coverage root-cause report and fail-closed terminal ledger were added,
ClaudeAgent performed another read-only audit and returned **PASS, no P0**. It
independently confirmed that `completed_stopped_fail_closed` means completion of
the negative stop procedure, not completion of T4; the report explicitly records
`T4_completed_count=0` and opens no scientific claim.

The reviewer manually reconciled all 50 gaps: 30 require additional clean SOP
evidence and 20 have sufficient evidence only in source-split held-out runs. It
found no hidden negative result, post-hoc gate relaxation, primary-condition
leakage, pseudo-ablation, or unsupported claim. Its P1 items are the scientific
limits already exposed by the ledger: one replay-eligible task, 50/120 coverage
gaps, only four source-level test runs, and held-out detector recall `0.125`.

P2 recommendations for v2 are to preregister the exact held-out challenge before
authoring, explain low R1-R3 Agent yield, replace the consumed T2 test episode,
and keep silver-label T1 results diagnostic until two-person blind adjudication.
