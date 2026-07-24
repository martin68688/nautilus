# Decision Admissibility WP0–WP8 consolidated implementation report

Date: 2026-07-24  
Branch at closeout: `codex/dual-time-procedural-memory`  
Baseline checkpoint: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`

## Outcome

WP0 through WP8 engineering implementation is complete. Every work package
passed its Stop Gate before the next phase. The final engineering Gate reports
20/20 prerequisites, 6/6 kill gates, and 47/47 acceptance checks. An independent
Claudeagent MCP session using `glm-5.2[1m]` returned `PLAN_APPROVED`.

This is not a positive performance result:

- Full Decision Admissibility completion: 4/9;
- No Memory completion: 6/9;
- score-estimable Full/No-Memory pairs: 4/9;
- available-pair wins/ties/losses: 4/0/0 (diagnostic only);
- raw exact one-sided sign-flip p: 0.0625;
- Holm-adjusted p: 0.25;
- Taxi scored pairs: 0;
- Full superiority: rejected;
- experience causality: pending without formal L4.

## Work-package closure

| WP | Delivered capability | Stop-Gate evidence |
|---|---|---|
| WP0 | baseline, dirty-worktree and asset-preservation audit | baseline remote checkpoint and WP0 report |
| WP1 | canonical Stage ontology, Claim/Operation/Protocol authority | WP1 report and focused tests |
| WP2 | mixed Claim decomposition and trusted Receipt boundary | WP2 report and forgery/trust tests |
| WP3 | clause-level pre-Prompt Visibility Gateway, bypass closure | WP3 report and deterministic exposure tests |
| WP4 | read-only corpus, audit sidecars, Full/seed/task raw Bundles | WP4 report and checksum ledger |
| WP5 | Base/Overlay, Result/Adoption/Causal paths, atomic publication | WP5 report and crash/idempotence tests |
| WP6 | method-preserving Clean Replay and certified child Bundle | WP6 report and replay verification |
| WP7 | off→shadow→review→canary→enforce/rollback | corrected canary report and verification |
| WP8 | controlled, multi-generation, same-domain online evaluation, statistics and Ledger | joint inventory, statistics, Evidence Ledger and Final Gate |

Rejected/partial attempts remain preserved. In particular, the WP7 r11
infrastructure abort and WP8 final-test attempts r1–r3 are not used as success
evidence; only r4 is bound.

## Implemented architecture

### Pre-use authority

- canonical Generation and Governance stage axes;
- typed Claims and explicit Claim/Operation compatibility;
- immutable ProtocolSpec registry and compiled evidence obligations;
- host-owned trusted collectors and hash-chained Receipts;
- clause-level visibility before embedding/ranking/token/Prompt;
- task/domain/protocol transfer scope with same-domain provisional generation;
- Dynamic Hybrid / GlobalMemory consuming only authorized packs.

### Post-use authority

- independent Result Fact publication from the current target evidence;
- Adoption Edge only after matching static/runtime actuation;
- Causal Edge only after matching counterfactual evidence;
- Positive Result and Positive Adopted SOP separation;
- fixed-holdout exactly-once terminal Result Fact finalizer;
- immutable Base, append-only Overlay, atomic child publication and rollback;
- Clean Replay new/Successor Claim recovery without historical mutation.

## Corpus and Bundle evidence

The formal non-Spooky corpus has 79 complete runs across 16 tasks, 1,577 audited
code nodes and 1,545 bound clauses. Eleven incomplete directories are excluded
with reasons. Full, seed-heldout and task-heldout manifests are independent and
hash-bound; source/test overlap and heldout references are zero. Raw-audited
Bundles intentionally do not inherit source-task answers or scores.

The later task-specific certified method Bundles are same-domain and
target-history-excluded. Cross-domain non-general methods are blocked. General
guardrails may transfer, but target-task scores never do.

## Formal WP8 experiment

The preregistered population contains:

- 3 protocol/task families: image, audio and tabular;
- 3 agent seeds per task;
- 5 online systems per block;
- 9 host-only Oracle dispositions;
- 45 assigned online outcomes;
- 22 protocol-legal selected scores and 23 retained failures;
- 0 imputed scores and 0 post-assignment exclusions.

Every successful condition emitted one Result Fact; failed conditions emitted
none. The first four completed blocks bind source snapshot `d7fed754...`; the
five result-blind retry blocks bind `c674364d...`. The r9 preregistration
amendment, frozen before analysis, limits the difference to schema/receipt
plumbing and declares candidate generation, retrieval, condition behavior,
terminal evaluation, and writeback semantics unchanged.

Post-result safety fixes are covered by closeout regression only and are not
credited to the formal effect estimate.

## Final regression

Authoritative root:
`coordination/decision_admissibility_wp8_final_tests_20260724_r4/`

- §20.1: 410 passed;
- §20.1-A: 45 passed;
- §20.2: 69 passed;
- §20.3: 58 passed;
- Tier-2/final targeted: 93 passed;
- full suite: 760 passed;
- frozen composite: 19 passed;
- seven green runs / 1,454 tests total;
- zero failure, error, or skip;
- 579-entry source/test-dependency closure unchanged before/after.

Receipt hash:
`e46eeaa4a7db62f69aa22ba73a2c9ba110babc74ae8dee5e05666ff5a17beb78`.

Final regression receipt hash:
`e006ed0f0817d7372efc699ed9f320a8d2fe3b990d1d610c867764e531ea89fd`.

## Final Gate and independent review

Final Gate report hash:
`de73a72b8e55db72a54e8cbcd38fa1d3ce4df580158c1e4449692c0675bac5e2`.

Independent Gate verification hash:
`dcf2e62c47d9f22c03a7adb05d325f588367284aee8f7309d2d0831116718778`.

Claudeagent audit:

- session `b0d62720-cdc7-4afe-9d3a-e543414a0214`;
- actual model `glm-5.2[1m]`;
- 48 turns;
- only Read/Glob/Grep observed (47 calls);
- no web calls, write calls, subagents, or permission denials;
- verdict `PLAN_APPROVED`;
- no blocking findings.

The reviewer authorized WP8 engineering completion only. It explicitly preserved
rejected superiority, diagnostic conditional utility, and pending causality.

## Asset and source-boundary preservation

No historical formal outcome, Bundle, failed receipt, or user presentation was
overwritten. New documentation, paper, PDF and presentation outputs are
versioned. The r4 source closure does not include these append-only coordination,
paper-version, or output-version artifacts, so documentation closeout does not
retroactively change the tested implementation.

## Remaining scientific work (not an implementation blocker)

A fresh preregistered replication may test a narrower performance hypothesis
after reducing authority denials and candidate-runtime failures. It must retain
the current 45 outcomes, use additional same-domain tasks/seeds, and bind claimed
experience effects to L2/L3/L4. Until then, no “memory system is better than no
memory” or experience-causality statement is authorized.

