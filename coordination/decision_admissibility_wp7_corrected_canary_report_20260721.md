# Decision Admissibility WP7 Corrected Canary and Stop-Gate Report

Date: 2026-07-21  
Branch: `codex/dual-time-procedural-memory`  
HEAD / upstream: `b47dab63b7861f3ea0871094d6dd07b77e6b81a4`  
Work package: WP7 — Shadow → Enforce  
Verdict: **PASSED for the pinned r7 rollout; WP8 was not started by this report**

## Scope and preservation boundary

- The run used the user-authorized Dev Pod
  `ecepxie/decision-admissibility-wp7-canary-a100x1-r7`; no Kubernetes Job was
  created.
- No commit, push, branch rewrite, cleanup, or broad staging was performed.
- HEAD and upstream remained the remote-verified baseline above. The dirty
  worktree and unrelated tracked/untracked user assets were not overwritten.
- The canary used a fresh immutable source snapshot and fresh output roots; no
  r25 output was reused as success evidence.

Pinned source:

```text
/work/decision-admissibility-wp7-corrected-r7d-source
source_sha256 = b3b8e718d11d8cad2f10c81cf0625650a580943557311b4650b70b68553ed3f2
parent_source_sha256 = 22ba164efe6bf4513e0c31c255e248c8d68d9e86c968d8fdabc25ca0e10e7725
```

Fresh run:

```text
/workspace/decision-admissibility-wp7-canary-corrected-aerial-r7
/workspace/decision-admissibility-wp7-canary-corrected-aerial-r7/runs/20260720_183851_wp7-canary-corrected-aerial-r7
```

The source preflight and post-run verification both reported no added, removed,
changed, writable, or `__pycache__` paths. The formal Bundle also revalidated
unchanged after the run.

## Same-domain task-heldout design

Target task: `aerial-cactus-identification`  
Target domain: `image`  
Certified source task: `leaf-classification`  
Transfer design: `same_domain_different_task_task_heldout`

Pinned Bundle:

```text
bundle_id = mlevolve-be034ec-image-aerial-task-heldout-certified-replay-v4
manifest_sha256 = 8c9bac59636e53488ab25dc4b361229a10dd9dc86bbe3bf15444e2649cc2eb18
certified_clause = clause::certified-replay::75ddd37fe7d7d115be468c72
```

The exposure audit found exactly two certified image→image exposures, one
unique contract, zero target-task historical sources, and zero cross-domain,
unscoped, or invalid exposure.

## Execution result

The pinned launcher completed with `EXIT_CODE=0` and
`LAUNCHER_EXIT_CODE=0`. Synthetic enforce preflight completed with:

```text
147 passed
```

The online search executed ten candidate nodes. Nine candidates failed for
ordinary workload reasons such as DataLoader worker failure, unavailable local
model material, incompatible loss/autocast, or execution timeout. Those nodes
were preserved in RunForest for Debug and correctly denied positive Result
writeback because no trusted successful `code_execution` receipt existed.

One cold-start Debug descendant completed cleanly:

```text
artifact_id = 1c9032936d864b5bae89c6d320ffaa25
validation AUC = 0.9968613459259859
code_sha256 = 38108263651860f5eff2b4d984bd28a09fe26105684632be93b5c7311ab901f6
```

Its exact code is present in the final immutable Journal and resolves to the
same hash stored in the Result Fact. It has trusted method identity,
`code_execution`, split-lineage, fit-scope, prediction-scope, evaluator, and
selection-freeze receipts under the pinned ProtocolRef.

## Result / Adoption / Causal online separation

The successful cold-start node was not exposed to the certified leaf clause.
Its enforced memory-writeback decision was:

```text
operation = PROMOTE_RESULT
outcome = ALLOW
missing_obligations = []
```

The frozen Session Overlay classifies as:

```text
Result Fact count = 1
Adoption Edge count = 0
Causal Edge count = 0
derived_from_refs = []
adoption_status = not_exposed
```

The two historical-experience exposures belonged to failed Draft nodes, not to
the successful cold-start result. No `experience_link_appended` event exists.
This is the required online evidence that:

1. current-node execution/protocol evidence controls `PROMOTE_RESULT`;
2. missing historical actuation does not block an independent clean Result
   Fact;
3. exposure alone does not mint Adoption or Causal lineage;
4. static/runtime/counterfactual actuation remains evidence for the historical
   experience→current-node edge, not for whether the training program ran.

## Independent oracle and official canary gate

The launcher generated a blank 31-decision oracle packet. Independent reviewer
`dalton-wp7-r7-independent-oracle-reviewer` checked the packet against the
complete 256-event ledger and authored all labels after the run:

```text
oracle allow = 13
oracle deny = 18
requires_fix = []
```

The official verifier accepted all 31 labels and the official canary evaluator
passed with the strict thresholds:

```text
minimum decisions = 20
observed / enforced decisions = 31 / 31
unauthorized effective allow = 0
effective false denial = 0
effective IIR = 0.0
effective VKR = 1.0
legacy IIR = 0.5
```

Operation population:

```text
PROMOTE_RESULT = 10  (1 allow, 9 deny)
RANK = 13            (4 allow, 9 deny)
SELECT = 8           (8 allow)
```

The ledger hash chain, every decision record hash, oracle evidence hash, and
review packet binding were independently verified. Authority internal-error
taxonomy count and launcher runtime-exception count were both zero.

## Local regression after the canary

```bash
PYTHONPATH=mlevolve .venv/bin/python -m pytest -q \
  tests/authority/test_canary_gate.py \
  tests/authority/test_canary_launcher_static.py \
  tests/authority/test_result_adoption_causal_writeback.py \
  tests/authority/test_actuation_pipeline.py \
  tests/authority/test_runtime_protocol_observer.py \
  tests/authority/test_enforce_rollout.py \
  tests/authority/test_domain_transfer_scope.py
# 51 passed in 19.26s

PYTHONPATH=mlevolve .venv/bin/python -m pytest -q tests/authority
# 170 passed in 23.23s
```

The previously documented frozen composite-benchmark detector-lock mismatch was
not changed to manufacture a pass.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Authority ledger | `b8e5bae977b90dcf91f26a1f5097cf0a19a78cee3624892a088266a0e13d89f7` |
| Authority rollout report | `d8875312341e50f5ec9d65799a4b12040434701a5e57f16d9aef3ceb09f42360` |
| Final Journal | `46952e6f698f1bde837c9cf136f43c0e1e4ed1db646b1d46d87ad364e0028db4` |
| Session Overlay events | `0898dd26a11ae446babbbd15d0a45ee8f372b545ae563d51dcff6a8e68edc13f` |
| Session Overlay manifest | `6f0e6e447277fe48de7b3ca809e2f257291d8c7cb9c4b5b16eb4e49815941c36` |
| Reviewed oracle packet | `cff9bdfa7f3c89c1c484b58494b9d8a62686f6c4fe72373d5142f385a079cdc6` |
| Verified oracle | `4e519d24182dd1efa707a019788f3b7454539ad229f3aaf3588a7fc2d2820618` |
| Oracle review report | `abe101bfafe2e685f24cf947a1de119487aa0617a63a766ea484f943cb2fbf95` |
| Canary gate report | `0881f9a3bc1a18fcb84288fd9863db414b1c57bdbf719953953c818176b54c3b` |
| Exposure-scope audit | `10ed7295d59ed8d59e217553a629118a78cbdce6ec614ea91c563ee60a565948` |
| Runtime-exception audit | `50006e369027ca302b9e469b744ecf3e72f2320285cbef8661c7da7646e54af7` |
| Source post-run verification | `d05f78fe9607cb2e1c3715344c340d4f6bb0bd1ca0e0d48c40d9d8b1911d7166` |
| Canary run summary | `d95ef0d870cbc380143336c0843464fb27b5b2ab9064581a0c8d1c092073b03b` |

## Known boundary before WP8

The v4 certified replay clause is correctly restricted to same-domain
`GENERATE_CANDIDATE` use and its historical metric is not a trusted Rank,
Select, or Promote path. Its prose nevertheless contains the source-task phrase
“validation log loss of 0.1458”. This does not alter the r7 Result/Adoption/
Causal verdict, does not expose target-task history, and did not authorize any
historical score operation. It is still a semantic-purity risk for paper effect
experiments because a method-only Prompt should not carry an outcome assertion.

Therefore WP8 must begin with a fail-closed semantic-purity audit and a new
immutable method-only Bundle (or a deterministic projection that removes the
score-bearing proposition) before any transfer-effect or Tier-2 result is used.
The v4 Bundle must not be presented as final evidence that method and source
outcome text are fully atomized.

## Stop-gate decision

- [x] Frozen policy/protocol/collector/Bundle/source pins verified.
- [x] Shadow disagreement population independently reviewed before enforce.
- [x] Synthetic high-risk and visibility enforce passed.
- [x] Fresh same-domain task-heldout online canary completed.
- [x] Minimum enforced-decision population exceeded.
- [x] Independent oracle verification and strict effective-IIR/VKR gate passed.
- [x] Runtime and Authority internal exceptions are zero.
- [x] Cross-domain/unscoped/target-history exposure is zero.
- [x] A clean unexposed node independently produced a Result Fact.
- [x] That Result Fact has no Adoption/Causal lineage.
- [x] Failed/exposed nodes produced no positive Result/Adoption/Causal writeback.
- [x] Source and formal Bundle remained immutable.

**WP7 Stop Gate: PASSED for the pinned r7 enforcement rollout.** WP8 may enter
its deterministic/Tier-0 entry work, but the semantic-purity remediation above
is a mandatory precondition before using same-domain transfer content in paper
effect or Tier-2 experiments.
