# Decision Admissibility WP0–WP8 Independent Claude Audit Packet

Packet purpose: a bounded, self-contained input for a read-only Claude Fable 5
Advisor audit. This packet is not itself evidence of approval. The reviewer must
look for contradictions, missing evidence, claim laundering, post-hoc changes, and
failure suppression rather than defer to the root agent's conclusions.

## Required reviewer output contract

The first line must be exactly one of:

- `PLAN_APPROVED` — only if the evidence presented is sufficient to authorize
  WP8 engineering completion while preserving every negative claim boundary.
- `PLAN_REVISE` — if any material evidence, implementation, statistical,
  provenance, leakage, or authorization issue remains.

Then provide:

1. a concise verdict;
2. a findings ledger with stable IDs, severity, supporting evidence, and required
   correction;
3. an explicit answer for each audit question in the final section;
4. a separate statement on whether any performance-superiority or
   experience-causality claim is authorized.

Approval must not authorize a performance headline. The formal effect gate is
false by construction and by observed results.

## Scope and immutability boundary

- Repository: `/Users/haoming/Downloads/nautilus`
- Branch: `codex/dual-time-procedural-memory`
- HEAD: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`
- Worktree is intentionally dirty and preserves user assets. No final commit or
  push has been performed.
- Execution plan:
  `coordination/decision_admissibility_complete_execution_plan_20260719.md`
- Final Stop Gate:
  `coordination/decision_admissibility_wp8_final_stop_gate_20260724_r1/stop_gate_report.json`
  - report hash:
    `de73a72b8e55db72a54e8cbcd38fa1d3ce4df580158c1e4449692c0675bac5e2`
  - independently recomputed verification hash:
    `dcf2e62c47d9f22c03a7adb05d325f588367284aee8f7309d2d0831116718778`
  - status: `engineering_complete_effect_claim_not_authorized`
  - `wp8_engineering_complete=true`
  - `wp8_stop_gate_passed=true`
  - `effect_claim_authorized=false`
  - `goal_completion_authorized=false`
  - next authorized phase: `Independent Claude audit`
  - 20/20 prerequisites, 6/6 kill gates, and 47/47 acceptance checks passed.

The Stop Gate root, final regression package, and accepted host test receipt root
are mode `0555`; contained files are mode `0444`. Their verifiers reject writable
roots, symlinks, non-regular or extra entries, source drift, command drift, JUnit
count laundering, skips, and self-rehashed tampering.

## System semantics that must remain distinct

The implementation and paper claims distinguish three writeback objects:

1. `PROMOTE_RESULT`: the current target node may publish its own Result Fact from
   target execution, protocol, data, evaluator, and score evidence. It does not
   claim that historical experience affected the node and fixes
   `derived_from_refs=[]` for an independent result.
2. `PUBLISH_ADOPTION`: historical experience → current artifact adoption edge;
   requires static and runtime actuation.
3. `PUBLISH_CAUSAL`: historical experience → current artifact causal edge;
   additionally requires a counterfactual (L4) receipt.

Therefore an unexposed or cold-start successful target node can enter memory as a
Result Fact without historical actuation. Missing adoption/causal evidence must
block only the corresponding edge and descendant authority, not the target result.

The engineering implementation also claims the following completed invariants:

- one explicit dual-axis Stage Ontology;
- mixed experiences decomposed into Claim-level permissions;
- host-owned trusted receipts cannot be supplied by Agent prose;
- GlobalMemory checks outcome, operation scope, policy, protocol, and stages;
- clause visibility is enforced before ranking and Prompt construction;
- no `attached_sop_ids`, projection, cache, or legacy-memory bypass;
- empty authorized packs cause traced abstention, not legacy fallback;
- Base Bundle + Overlay + crash-safe atomic publication;
- fixed-holdout positive writeback occurs exactly once only after sealed terminal
  scoring;
- production paths do not call legacy `Operation.PROMOTE`;
- Positive Result SOP and Positive Adopted SOP are distinct;
- Clean Replay may create a new evidence path, but method-changing replay cannot
  restore the old Claim;
- descendant paraphrase or distillation cannot expand authority.

## Same-domain, cross-task transfer boundary

Formal tasks and domains were frozen before formal execution:

- `aerial-cactus-identification` — image, macro-F1, maximize;
- `mlsp-2013-birds` — audio, macro-F1, maximize;
- `new-york-city-taxi-fare-prediction` — tabular, RMSE, minimize.

The hard domain rule is:

- online systems may receive only same-domain method content or explicitly
  domain-general guardrails;
- cross-domain and unknown-source method content is forbidden;
- target-task history, answers, labels, scores, code, and audit-derived target
  text are forbidden from the memory Bundle;
- a clean, manifest-bound same-domain source execution may provide a provisional
  `METHOD_HYPOTHESIS` for candidate generation, but cannot transfer source scores,
  authorize Rank/Select/Promote, publish a target Result Fact, or establish target
  success;
- every generated target candidate must be re-executed and evaluated on the target
  protocol.

The original preregistration recorded
`task_heldout_target_history_exposure=0`. Immutable child Bundles were separately
published and validated for image, audio, and tabular domains. The staging Gate
asserted child Bundle identity and validation, holdout isolation, evaluator and
train hashes, copied physical views, condition universes, and empty output roots
for every block before launch.

Semantic-purity evidence:

- method report hash:
  `649b1f3ae6c02301775bb9ad9a51227bb78a67c374afd5e284f3c31526c4d92d`
- `passed=true`
- `source_outcome_assertion_count=0`
- `raw_text_embedded=false`
- formal and independent Bundle validators both returned `valid=true`, no errors,
  zero heldout references, zero Spooky nodes, and resolvable clause sources.

## Formal design and preregistration chain

Formal design:

- 3 tasks × 3 agent seeds (`104729`, `130363`, `155921`) = 9 blocks;
- 5 online systems per block:
  `full_decision_admissibility`, `no_memory`, `flat_relevance_memory`,
  `global_validity_bit`, `authority_only`;
- one host-only Oracle disposition per block;
- 45 assigned online outcomes and 9 Oracle dispositions;
- primary contrast: Full − No Memory;
- secondary contrasts: Full − Flat, Full − Global Bit, Full − Authority Only;
- identical task data, holdout, model environment, candidate contract, budget,
  condition-order control, and host-owned terminal evaluator within each block.

Preregistration consists of r1–r9 plus five independent verification receipts.
The final r9 amendment states:

- tasks, systems, seeds, conditions, holdouts, memory Bundles, memory permissions,
  search budgets, Oracle algorithm, primary contrast, target-history exclusion,
  source-score inheritance, and statistics did not change;
- only formal execution revision/root, host failure receipt, and runtime schema
  compatibility changed for a result-blind precontract recovery;
- `terminal_score_value_used_to_choose_fix=false`.

The analysis policy was frozen after structural dispositions but before score
reveal:

- policy hash:
  `26d4fe7e7645fe8f1911e757df2e15a9cf85966158e0d567e6d7e0dd3705670d`
- status: `frozen_after_structural_dispositions_before_score_reveal`
- terminal and Oracle score values were neither included nor inspected;
- source scores were not used to choose the policy;
- all 45 assignments remain in the ITT population;
- score imputation, post-assignment exclusion, and rerunning to replace failure are
  forbidden;
- an unscored pair has undefined continuous delta but remains a failed assignment
  in ITT and completion endpoints;
- raw native metrics/deltas cannot be pooled across tasks;
- exact sign-flip tests enumerate all `2^n` signs;
- Holm correction covers the frozen family of four contrasts;
- not-estimable analyses fail the effect-claim criterion;
- all eight frozen effect criteria must pass to authorize the headline.

## Formal execution and missingness

Observed disposition counts:

- 9/9 blocks;
- 45/45 online assignments retained;
- 9/9 Oracle dispositions;
- 22 scored selected results;
- 23 retained pre-terminal failures;
- 22 Result Facts, one for each successful condition;
- zero failed conditions with a Result Fact;
- zero fixed-holdout Result-Fact orphans;
- zero imputed scores;
- zero post-assignment exclusions.

Completion by system (denominator 9 each):

- Full: 4/9;
- No Memory: 6/9;
- Flat: 4/9;
- Global Bit: 4/9;
- Authority Only: 4/9.

Completion by task across all five systems:

- aerial: 15 scored, 0 failed;
- birds: 6 scored, 9 failed;
- taxi: 1 scored, 14 failed.

In particular, Taxi has no Full/No-Memory scored pair. Taxi seed 155921 has No
Memory scored while Full failed. These facts are retained as negative evidence and
are not repaired, excluded, or imputed.

Joint inventory evidence was hash-only before score analysis:

- report hash:
  `19a2835857945d56980b8dd0a6ae7d9a51e718045acd6753e9f7db8af05cdba5`
- verification hash:
  `bc1ea054326304e4cdb935e4df6ace0df7a1e70eed3349aec2a593a35e117f62`
- `score_policy=hash_only`
- `score_values_included=false`
- `score_values_inspected=false`
- `score_bearing_artifacts_parsed=false`.

## Statistical result and permitted interpretation

Primary Full − No Memory completion endpoint:

- Full 4/9 versus No Memory 6/9;
- mean completion difference `-2/9`;
- both complete: 4;
- Full-only complete: 0;
- No-Memory-only complete: 2;
- neither complete: 3;
- exact one-sided discordant completion p = `1.0`.

Continuous primary contrast, only where both systems legally scored:

- availability: 4/9;
- 4 wins, 0 ties, 0 losses for Full;
- exact one-sided sign-flip raw p = `0.0625`;
- Holm-adjusted p = `0.25`;
- aerial: 3 available pairs, mean macro-F1 delta
  `+0.014373260193018914`;
- birds: 1 available pair, macro-F1 delta
  `+0.05979378455983453`;
- taxi: 0 available pairs;
- task-macro bootstrap: not estimable because one task has no scored pair;
- mixed-effects sensitivity: not estimable because a contributing task has fewer
  than two pairs.

Consequently:

- Full superiority over No Memory is rejected;
- positive direction in the four available pairs is diagnostic only;
- poorer Full completion and Authority false-denial/runnability loss offset the
  conditional gains;
- no target performance headline is authorized;
- no source-task score is inherited or used as a target score.

Statistics report hash:
`5480e2f5c39ae8d8bde1e5d279877959c62874ca9c28e3ee1a2bf4da587369c8`.
Independent verification hash:
`81078aeb542cd160c4dda87fb308a56baeee189d85bfc7cc0307633293f6f360`.

## Evidence Ledger claim states

Ledger hash:
`c9e8f5b2cedee1072baae0d7f5c5a505de4980e59223a0678ef19dcc86174570`.
Independent verification hash:
`a21af8c27420447f88fc45f38be77cac37be20e8ef2f901062dc8fd9dadd2b2f`.

- `WP8-C1-FORMAL-EXECUTION`: supported.
- `WP8-C2-RESULT-WRITEBACK`: supported; Result retention only, not Adoption or
  Causal edges.
- `WP8-C3-FULL-SUPERIORITY`: rejected.
- `WP8-C4-CONDITIONAL-UTILITY`: diagnostic only.
- `WP8-C5-NO-IMPUTATION`: supported.
- `WP8-C6-EXPERIENCE-CAUSALITY`: pending; L2 static + L3 runtime + L4
  counterfactual evidence is required and was not established for formal gains.
- `WP8-C7-PRIOR-KILL-GATES`: supported for mechanism/safety evidence only; it does
  not override the formal downstream result.

`headline_effect_claim_authorized=false`.

## Test and source-version evidence

Accepted host-owned test receipt:

- root:
  `coordination/decision_admissibility_wp8_final_tests_20260724_r4/`
- receipt hash:
  `e46eeaa4a7db62f69aa22ba73a2c9ba110babc74ae8dee5e05666ff5a17beb78`
- manifest hash:
  `ed9adc6485d371d1ab7fef89dda8a12558771791f9b2f57a934b928cf163901e`
- source/test-dependency snapshot: 579 files, about 115 MB;
- snapshot hash:
  `5415e61d198e952f304490a963780a8be843c44d8232e0d56ec36bf1d856d81e`
- `source_unchanged_during_tests=true`;
- exact commands, working directory, controlled pytest environment, repository
  `.venv`, Python executable, and key dependency versions are bound;
- pytest plugin autoload is disabled and test-control environment is fixed;
- section JUnits must be subsets of the current full suite;
- skips, duplicate test identities, declared/observed count mismatch, JUnit reuse,
  writable roots, extra/special entries, and self-rehashed command tampering fail
  verification.

Passing test counts, all with zero failures, errors, and skips:

- §20.1: 410;
- §20.1-A: 45;
- §20.2: 69;
- §20.3: 58;
- Tier-2/final targeted: 93;
- full suite: 760;
- frozen composite module: 19.

The current full suite explicitly includes malicious re-signed split overlap,
candidate-build crash isolation, empty-pack abstention, re-signed trace tampering,
P4-B latency/token/empty-pack reporting, and receipt command/skip laundering tests.

Final regression package:

- root:
  `coordination/decision_admissibility_wp8_final_regression_20260724_r1/`
- receipt hash:
  `e006ed0f0817d7372efc699ed9f320a8d2fe3b990d1d610c867764e531ea89fd`
- file SHA256:
  `20ba0d027e43555cf39f7de7c8629cc2dc77dc0676d6322ac86a4bca92b199ae`
- final source inventory: 390 implementation/test files;
- source inventory hash:
  `bfb01b79dbb6021931165dfce78a5a420dee8d3deb9b0670aa956f5c1664f9b8`;
- verifier independently recomputed the receipt with no errors.

Two real historical regression chains remain preserved rather than deleted:

1. Tier-2 revision-chain verifier: 2 failures in the original 90-test JUnit,
   followed by a 92-test clean revision-aware run.
2. Semantic-provenance loader trigger: 1 failure in the 410-test r2 baseline,
   followed by a clean 410-test r3 after narrowing the modern-Bundle trigger while
   retaining malicious overlap rejection.

## Invalid final-test attempts that must not be laundered

These roots are retained but are not valid Gate inputs:

1. `decision_admissibility_wp8_final_host_test_receipt_20260723_r1`:
   interrupted partial root; only a 463-row source inventory exists; no receipt or
   manifest. It was stopped because the dependency closure was known incomplete.
2. `decision_admissibility_wp8_final_tests_20260724_r2`:
   tests were green (including a 758-test full suite), but receipt status is
   `failed` and `source_unchanged_during_tests=false` because seven closeout source
   and test files changed concurrently. It is not used by the Gate.
3. `decision_admissibility_wp8_final_tests_20260724_r3`:
   interrupted partial root; only a 579-row source inventory exists; no receipt or
   manifest. It overlapped a final test-source edit.
4. r4 is the only accepted host receipt. It began after all subagents were stopped,
   uses the expanded dependency closure, completed with a byte-identical before/
   after snapshot, and is the root bound by final regression and Final Gate.

No formal Tier-2 training outcome, score, seed, task, or condition was rerun or
reselected during closeout. r1–r4 above are local regression-test attempts, not
formal experiment reruns.

## Post-result change boundary

The formal outcomes bind the frozen execution sources used during online runs.
Post-result changes were restricted to safety/verification closeout, including:

- semantic overlap validation at Bundle load;
- candidate publication crash isolation;
- explicit empty-pack abstention;
- stronger factorial trace-value and re-sign tamper checks;
- P4-B overhead reporting;
- host-owned regression receipts and Final Gate verification.

These changes were regression-tested but were not used to alter, rerun, exclude,
impute, or reinterpret formal outcomes. The Final Gate records:

- `post_result_safety_fixes_not_used_to_rewrite_outcomes=true`;
- `post_result_safety_fixes_not_evaluated_for_formal_effect=true`;
- `rerun_or_seed_selection_performed=false`.

This boundary means engineering completion may include the closeout safety fixes,
but the formal effect estimate cannot be credited to those post-result sources.

## Audit questions

Answer every item explicitly:

1. Does the r1–r9 preregistration/amendment chain preserve the scientific design,
   or is any retry/recovery a post-outcome method change or unauthorized rerun?
2. Are 23 failures correctly retained without imputation or post-assignment
   exclusion, including the Taxi Full failures?
3. Is the statistical conclusion correct given Full 4/9 versus No Memory 6/9,
   only 4/9 scored pairs, raw p `0.0625`, Holm p `0.25`, and a non-estimable
   three-task bootstrap?
4. Does the same-domain child-Bundle rule test cross-task method transfer without
   target-task history, answer, score, or cross-domain method leakage?
5. Are Result Fact, Adoption Edge, and Causal Edge permissions kept separate in
   both engineering and claims?
6. Is `WP8-C4` appropriately limited to a diagnostic available-pair statement?
7. Must `WP8-C6` remain pending without L4 counterfactual evidence?
8. Does the Evidence Ledger reject rather than launder the failed superiority
   claim?
9. Do the host test receipt, preserved failure chains, source inventory, immutable
   package modes, and malicious tests sufficiently support engineering completion?
10. Is the post-result source-version boundary honest and adequately prevents
    using closeout fixes as formal-effect evidence?
11. Do the invalid r1/r2/r3 regression-test attempts remain visibly excluded from
    the accepted r4 evidence chain?
12. Is it defensible to approve WP8 engineering completion while explicitly
    rejecting performance superiority and leaving experience causality pending?

If the packet lacks enough raw evidence to support any answer, return
`PLAN_REVISE` and identify the exact missing artifact or invariant. Do not infer
approval from the existence of hash fields alone.
