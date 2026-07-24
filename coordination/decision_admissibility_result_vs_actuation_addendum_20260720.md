# Decision Admissibility: Result Memory vs Experience Actuation Addendum

Date: 2026-07-20  
Applies to: `decision_admissibility_complete_execution_plan_20260719.md` WP5–WP8  
Status: implementation requirement discovered during WP7 r25 review

## Clarification

The memory subject and the experience-influence edge are different objects:

```text
historical SOP/Clause --exposed/adopted/causal--> current executed node
                                               ├── code/run facts
                                               ├── protocol-legal score facts
                                               └── newly distilled Claims
```

`static_actuation` and `runtime_actuation` prove the arrow. They do not prove
that the current training program executed; that fact is proved by
`code_execution` and the runtime protocol Receipts.

The current node may therefore enter RunForest/Session Overlay as an executed
result without an adoption edge. Exposure remains recorded, but
`derived_from_refs` must remain empty until a separately authorized adoption
publication exists.

## Split operations

| Operation | Subject | Minimum evidence | What it may assert |
|---|---|---|---|
| `PROMOTE_RESULT` | current executed node/result | method identity, code execution, clean protocol/data/evaluator Receipts for score-bearing Claims | the current node/result is a real, protocol-legal memory fact |
| `PUBLISH_ADOPTION` | historical experience → current node edge | source Claim + contract-bound static and runtime actuation (L3) | the historical experience was realized in the current program |
| `PUBLISH_CAUSAL` | historical experience → current node causal edge | all L3 evidence + contract-bound counterfactual actuation (L4) | removing/replacing the experience changed action or code |
| effective causal Claim | causal edge + outcome | all L4 evidence + protocol-legal positive outcome evidence (L5) | the experience caused a beneficial legal outcome |

Legacy `PROMOTE` retains its conservative L3 behavior so existing ledgers and
configs are not reinterpreted. New ordinary result writeback uses
`PROMOTE_RESULT`.

## Non-laundering rules

1. Exposure alone never creates `derived_from` lineage.
2. `PROMOTE_RESULT` records exposure report references but writes
   `derived_from_refs=[]`.
3. Each adoption/causal edge is a separate immutable Claim bound to one
   `ExperienceContract` hash and one target artifact.
4. Static/runtime/counterfactual Receipts from one contract cannot authorize a
   different contract's edge.
5. A current node's score may be stored without claiming that an exposed
   experience caused it.
6. Historical scores are never inherited by the current node, and current
   scores never retroactively certify the historical experience.

## WP7 r25 interpretation

r25 remains immutable evidence for the pre-addendum narrow canary:

- 42 enforced decisions;
- 2 certified same-domain exposures from `leaf-classification` to
  `aerial-cactus-identification`;
- zero invalid/cross-domain exposure and zero Authority/runtime exception;
- independent oracle: 24 allow, 18 deny, 42/42 agreement.

Its successful cactus nodes had protocol-legal scores but no verified
leaf-to-cactus actuation. Under this addendum, the correct interpretation is:

```text
PROMOTE_RESULT: allow for protocol-legal successful nodes
PUBLISH_ADOPTION: deny without L3
PUBLISH_CAUSAL: deny without L4
```

r25 does not by itself prove the corrected result-writeback path because its
frozen source used legacy `PROMOTE`. The corrected path must pass synthetic
tests and a fresh dev-Pod canary before WP7 is closed.

