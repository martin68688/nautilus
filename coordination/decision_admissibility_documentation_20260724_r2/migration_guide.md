# Migration guide: legacy memory to Decision Admissibility

This guide migrates a deployment without rewriting historical runs or Bundles.
Use a new branch/config/output root and preserve every rejected or partial
artifact.

## 1. Preconditions

1. Record `git status --short`, branch, HEAD, remotes, and the current Bundle
   pointer.
2. Create and remotely verify a selected baseline checkpoint. Never use
   `git add -A` in a dirty research worktree.
3. Freeze protocol, policy, collector, source, and Base-Bundle hashes.
4. Run the legacy baseline tests and retain their JUnit/log outputs.
5. Mount historical corpus roots read-only.

## 2. Semantic mapping

| Legacy concept | New concept | Migration rule |
|---|---|---|
| whole SOP/run valid bit | clause-level Claim authority | decompose; never union scopes |
| one stage enum | GenerationStage + GovernanceStage | map deterministically; unknown fails |
| relevance hit | proposal only | no permission implied |
| execution success | Result Fact candidate | requires target protocol evidence |
| injected/exposed memory | visibility/exposure record | never an Adoption Edge |
| `PROMOTE` | one of three explicit operations | choose Result, Adoption, or Causal |
| positive SOP | Positive Result or Positive Adopted SOP | separate authority paths |
| mutable memory directory | immutable Base + Session Overlay | publish a new child and CAS CURRENT |
| protocol repair in place | Clean Replay | new Claim or Successor Claim |

## 3. Register immutable protocols

Create `ProtocolSpec` entries for every supported protocol family and pin the
canonical hash in runtime config. Include split, preprocessing, evaluator,
metric, selection, seed, holdout, promotion, and compatibility policies.
Unknown versions must require replay rather than falling back to a nearby score.

Validate at least grouped classification, random classification, and
chronological regression without changing Authority kernel code.

## 4. Replace stage plumbing

At node creation, resolve the runtime stage through
`runtime_stage_axes(...)`. Governance callers pass an explicit governance axis
for rank/select/writeback/distillation/replay while retaining the node's
generation provenance.

During one compatibility cycle, legacy `DecisionStage` values may be supplied
to `resolve_stage_axes(legacy_stage=...)`. Log the resolved pair. Remove
callers that invent a fallback for unknown stages.

## 5. Decompose mixed memory

Run Claim decomposition before embedding or Prompt construction. Convert every
SOP to `SOPClauseV1` with stable clause, source, Claim, task/domain, protocol,
operation, stage, publication, Authority and Receipt refs.

Do not copy a container-level valid bit to all clauses. Audit/debug findings may
remain visible while score/rank clauses are denied.

## 6. Move trust to host collectors

Replace Agent-authored evidence dictionaries with trusted collector output.
Ingest through the Receipt bridge and validate collector capability, artifact
ID, run ID, protocol hash, payload hash, event chain, and Claim-type support.

Keep legacy static audit as `legacy_static_only`; never relabel it as an
observed runtime Receipt.

## 7. Put visibility before influence

Instantiate `SOPVisibilityGateway` in shadow mode first. Ensure every consumer
uses `VisibleSOPPack` before embedding, RRF, token allocation and Prompt
rendering. Remove or gate all alternate paths:

- `attached_sop_ids`;
- Tree/SOP projection helpers;
- legacy navigation edges;
- cached decisions;
- direct GlobalMemory lookups.

An empty pack must produce explicit abstention, not legacy fallback.

## 8. Split writeback

Replace every production `Operation.PROMOTE` call:

- current legal node/result → `PROMOTE_RESULT`;
- claimed historical use → `PUBLISH_ADOPTION`;
- claimed causal influence → `PUBLISH_CAUSAL`.

Do not require static/runtime actuation for an independent Result Fact. Do
require static plus runtime actuation for Adoption. Require a prior Adoption and
counterfactual actuation for Causal. Make each event idempotent and append-only.

## 9. Split positive distillation

Replace ambiguous positive distillation with:

- `DISTILL_POSITIVE_RESULT` for a method distilled from the target result's own
  legal evidence;
- `DISTILL_POSITIVE_ADOPTED` for a method attributed to historical experience.

The second path requires the matching L3 edge; causal wording requires L4.
Diagnostic failure knowledge remains a separate distillation class.

## 10. Install fixed-holdout terminal finalization

Disable positive writeback in planning, candidate generation, result parsing and
search-time scoring. After the host evaluator seals the terminal result, call
`finalize_result_writeback(...)` exactly once. Persist the explicit incomplete
status on any validation or tamper failure.

Verify orphan rate and duplicate writeback rate are zero under crash/restart.

## 11. Migrate storage

1. Publish the historical material as a verified immutable Base Bundle.
2. Start a new hash-chained Session Overlay.
3. Run the sleep-time pipeline into a fresh staging root.
4. Validate all stage reports and hashes.
5. Compare-and-swap `CURRENT.json` against the expected parent hash.
6. Preserve both parent and failed candidate roots.

Never edit a published Bundle to repair a manifest.

## 12. Clean Replay and recovery

Queue only fully bound method/debug candidates. Re-execute under the active
ProtocolSpec. Verify method fingerprint:

- preserved → register a new Claim with new trusted Receipts;
- changed → register a Successor Claim;
- unknown → require human review.

Do not restore the predecessor's score, rank authority, or task scope.

## 13. Rollout sequence

1. `off`: preserve the legacy output and establish observability.
2. `shadow`: compute both legacy and full decisions; do not change behavior.
3. independent review: sample disagreements from the verified ledger.
4. corrected canary: enforce only frozen operations/stages and evaluate IIR/VKR,
   unauthorized allow, and false denial.
5. staged `enforce`: expand only after kill gates pass.
6. rollback: CAS CURRENT to a verified parent; append rollback events; delete
   nothing.

Do not treat an infrastructure-aborted shadow as a completed Gate.

## 14. Minimum verification

Run the plan's §20.1, §20.1-A, §20.2 and §20.3 suites plus the full test suite.
Require zero failures, errors, and skips; bind executed argv, cwd, interpreter,
environment, testcase IDs, logs, JUnit files, and the source/test-dependency
closure.

Also verify:

- unauthorized Prompt exposure and activation are zero;
- clean unexposed Result Facts publish with empty derivation refs;
- exposure alone creates no Adoption/Causal edge;
- receipt types cannot substitute for each other;
- re-signed split/trace tampering remains rejected;
- crash before CURRENT swap preserves the old pointer;
- three-to-five-generation derivation never expands authority.

## 15. Rollback and incident handling

On internal error, stale policy, unknown protocol, missing Claim ref, forged
Receipt, or manifest mismatch:

1. fail the high-risk operation closed;
2. keep Inspect/Debug warning visibility only where policy permits;
3. preserve the invalid artifact and Authority trace;
4. roll back the active pointer through CAS if publication changed;
5. create a new repair/replay/output root;
6. rerun every invalidated Gate. Never rewrite the evidence that failed.

## 16. Completion criterion

Migration is complete only when the new host receipt proves an unchanged source
closure and all final tests pass, the immutable final Gate independently
recomputes, and the Evidence Ledger preserves negative scientific outcomes.
Engineering completion does not authorize performance superiority or
experience causality.

