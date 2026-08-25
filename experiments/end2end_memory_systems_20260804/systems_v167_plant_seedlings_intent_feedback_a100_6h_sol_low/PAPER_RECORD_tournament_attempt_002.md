# v167 Plant Seedlings four-seat Transfer tournament — attempt-002

## Paper-use status

This run is an **incomplete, right-censored four-seat tournament**. It must not be reported as a completed six-hour tournament, a completed four-seat ranking, a Top-2 result, or a champion result.

The defensible paper claim is:

> During an incomplete four-seat Transfer tournament on Plant Seedlings, three of four seats produced valid target-task baselines before an isolated EfficientNet-B0 checkpoint-discovery failure blocked baseline qualification. The best observed partial baseline achieved internal macro-F1 0.9863712473 and Kaggle public/private score 0.98236. The run was subsequently censored by the live Pod's six-hour `DeadlineExceeded` limit before the experiment's own six-hour budget expired.

`paper_grade_eligible=false` for the full tournament. The best baseline may be used only as a partial/censored observation with the failure and stopping mechanism disclosed.

## Identity and reproducibility

| Field | Value |
|---|---|
| Target task | `plant-seedlings-classification` |
| Transfer source task | `leaf-classification` |
| System | `dynamic_transfer_portfolio_tournament_v1` |
| Attempt | `attempt-002` |
| Run ID | `20260825_034153_e2e-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1-attempt-002` |
| LLM | `gpt-5.6-sol`, low reasoning effort |
| Behavior | `memory_transfer` only |
| Workers | 1 GPU / 1 worker / 16 CPU |
| Internal budget | 21,600 seconds |
| Tournament implementation | `9ca3b51722d28e8d92d631c8bbfe67983412a97a` |
| Seat-identity fix | `9ecc6d295e1108d8fc0e7b3555642dee4318af9c` |
| Pod manifest commit | `2623499a706e534bd59f1475ae0b554cabff4e08` |
| Immutable runtime | `/workspace/nautilus-exp-end2end-agent-v167-transfer-tournament-sol-low-r14` |
| Runtime receipt SHA-256 | `24506e4a272255ba8584f3a79c395a579cc5928daf4edb880548a1a184cb535e` |
| Frozen portfolio SHA-256 | `7ae852a9934964ea5e6e653fff0d919f36306545cd0aba130458d069b5e7ac9b` |
| Selection binding | `false` |

This attempt predates and **did not use** the later four-blind-ballot implementation in commit `c722a3f5`. Its four seats came from the earlier frozen Search/Judge portfolio.

The run identity registers 96 historical memory-source runs in the global snapshot, while the actual frozen Transfer candidates used here are projected from `leaf-classification`. Memory snapshot SHA-256: `cd84a7e76721139eb1068183c7e92bf2620ee8f2000b7d86e9f52e4f1ef6cd27`; index SHA-256: `2488813025311b1081daf414b6ff7b5df59247fe9de77868721ce4dbe43c15b2`.

## Compute and timing

| Field | Value |
|---|---|
| Pod | `mlevolve-plant-v167-tournament-a100-r1-dev` |
| Namespace | `ecepxie` |
| Node | `gp-engine.beocat.ksu.edu` |
| GPU | NVIDIA A100-SXM4-80GB |
| Resources | 1 A100, 16 CPU, 32 GiB RAM, 64 GiB ephemeral storage |
| Pod start | 2026-08-25 02:28:31Z |
| Container start | 2026-08-25 02:32:39Z |
| `run.py` start | 2026-08-25 03:41:41Z |
| Expected internal deadline | 2026-08-25 09:41:41Z |
| Kubernetes deadline signal | 2026-08-25 08:28:31Z |
| Container finish | 2026-08-25 08:33:32Z |
| Live `activeDeadlineSeconds` | 21,600 |
| Manifest `activeDeadlineSeconds` | 50,400 |
| Restart count | 0 |
| Terminal state | `Failed / DeadlineExceeded / exit 137` |

The formal attempt ran 17,210 seconds before the deadline signal and 17,511 seconds before container exit. It missed 4,390 seconds of its requested internal budget. The five-minute difference between deadline signal and container finish matches `terminationGracePeriodSeconds=300`; there is no evidence of OOM, NodeNotReady, or restart.

## Tournament protocol and observed progress

The intended protocol was:

1. Obtain four valid baselines.
2. Give each seat one valid Improve.
3. Give each Top-2 seat one additional valid Improve.
4. Continue only the champion.

The terminal state remained `baseline_qualification`: four distinct roots were created, but only three valid baselines were obtained. No round-one Improve, Top-2 ranking, finalist set, or champion was produced.

| Seat | Representation | Root outcome | Valid baseline node | Internal macro-F1 | Official score |
|---|---|---|---|---:|---:|
| `comprehensive_best` | B3 + target descriptors + self-attention | `weights=None` gate rejection; mandatory Debug | `a1c0044cba42441f97057af77b0ecec3` | 0.9147905355 | not submitted |
| `backbone_diversity_1` | B3 backbone-isolation variant | `weights=None` gate rejection; mandatory Debug | `4457027e05544659a64d8a84299d1a30` | 0.9600101100 | not submitted |
| `backbone_diversity_2` | independent B0 variant | `weights=None` gate rejection; three Debug executions failed | none | — | — |
| `descriptor_fusion_diversity` | B3 + target descriptors + gated fusion | valid initial Draft | `2fa93df8144240188a1c77227ddb5753` | **0.9863712473** | **0.98236 public / 0.98236 private** |

### Seat receipts

- `comprehensive_best`: candidate `transfer::comprehensive_best::5843e2a11615`, branch 1, root `81386978aad648e2a714fe2ab9056a2d`.
- `backbone_diversity_1`: candidate `transfer::backbone_diversity_1::33f6f45eeb05`, branch 2, root `ce07d322db8f4c379dc81ccb5ff68c44`.
- `backbone_diversity_2`: candidate `transfer::backbone_diversity_2::4dc76486e52c`, branch 3, root `cdf7051534ae4865923e032c468ab946`.
- `descriptor_fusion_diversity`: candidate `transfer::descriptor_fusion_diversity::73f1758184d2`, branch 4, root `2fa93df8144240188a1c77227ddb5753`.

## Best partial result

| Field | Value |
|---|---|
| Seat | `descriptor_fusion_diversity` |
| Node | `2fa93df8144240188a1c77227ddb5753` |
| Stage | Draft baseline |
| Internal macro-F1 | `0.9863712473009653` |
| Kaggle public | `0.98236` |
| Kaggle private | `0.98236` |
| Kaggle submission ref | `55766426` |
| Solution SHA-256 | `637643cbb4abbc58d5c5fa28f2fa5b091ee6a46e798c68dc4e2f8069de205c61` |
| Submission SHA-256 | `0908310bb337da7df9dee251a38007c9864844e6586db40d7f55a6bbb902909e` |
| Prediction signature SHA-256 | `8e08d3d930df98fc4a40c64b3924b14dbdbed68b3f2a95b8191d7ed468d6acd1` |

PVC artifacts:

- Solution: `/workspace/experiment-end2end-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1/runs/attempt-002/20260825_034153_e2e-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1-attempt-002/workspace/best_solution/solution.py`
- Submission: `/workspace/experiment-end2end-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1/runs/attempt-002/20260825_034153_e2e-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1-attempt-002/workspace/best_submission/submission.csv`
- Local official-score evidence: `coordination/plant_v167_tournament_attempt002_official_score_20260825/`

Internal macro-F1 and Kaggle accuracy-like leaderboard score are different metrics and must not be compared as though they were the same quantity.

## Failure analysis

The B0 seat failed for a reproducible software reason. All three mandatory Debug candidates passed the representation adoption gate, then failed inside isolated Candidate execution with:

```text
FileNotFoundError: Local public-pretrained EfficientNet-B0 weights not found:
efficientnet_b0_rwightman-7f5810bc.pth
```

Failed nodes:

| Node | Execution seconds | Code SHA-256 |
|---|---:|---|
| `bd546a034b714cde8824fb9247511f32` | 244.29 | `3dc0cfaf2773d14b2e5e11be26a9826fea3b6ff1129837ee7a0687a4582156f0d` |
| `20e02dcc46b34b648c2c12da5f6b6676` | 666.76 | `731f94fb9de8bf3cb8262159577ff703b683e3a71e5c8c8c2f49af927c3e89cc` |
| `40f63c678a834660bd157b010db174f1` | 251.33 | `3401ab590091216ce6187ef17544d900f7a124037c5ab369113c76b32b721571` |

This exposes a preflight coverage gap: the UID-65534 offline B0/B3 construction smoke passed, but it did not execute the exact checkpoint-discovery path used by generated isolated Candidate code.

After the third failure at 06:55:49Z, the tournament still required branch 3 to complete baseline qualification. The scheduler repeatedly logged that the role had no selectable node, returned `None`, and kept ordinary score-driven selection gated. It made no further real candidate progress for 5,562 seconds before the Kubernetes deadline signal.

Thus the incomplete outcome has two layers:

1. **Software failure:** Candidate-visible B0 checkpoint discovery failed, and the tournament lacked a terminal skip/fail transition after repair exhaustion.
2. **Infrastructure censoring:** the live Pod deadline was six hours from Pod start, so Kubernetes terminated the still-running process before its internal deadline.

## Communication and authority evidence

- Four distinct role contracts and four independent branch/root identities are present.
- All nine candidate nodes have an empty `strategy_alignment` payload.
- All nine candidate nodes record `host_plan_execution_gate.enabled=false` and `valid=true`.
- Therefore this run does **not** provide positive evidence for a Strategy→Planner alignment feedback loop or a Host→Planner recoverable-return loop.
- The authority ledger contains 44 events: 21 claims, 9 receipts, 6 decisions, 6 shadow comparisons, one protocol registration, and one frozen rollout-version event.
- Authority ran in shadow mode (`enforced=false`). The final failed B0 candidate received `require_replay` because the `code_execution` receipt was missing; this was marked repairable and assigned to the evidence collector, not demonstrated as a Planner return.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `run_identity.json` | `12cf992aeb93a34db8dc4278780e87ed26687c98fb670df00e0cb45c05e08fad` |
| `config.yaml` | `698a0b6b7e2b3d2196fa19a9cda20983663a38086a602e8e8d40475ab61e311d` |
| `TRANSFER_PORTFOLIO_TOURNAMENT.json` | `ca8670599c7a2bd20000e3b34e33bed43faf74eb2b12153dffffbb40e501cbbc` |
| `journal.json` | `e0bbc60417ce8b4d3644452c25659507248655ba8ec2001daf7718e023ba5556` |
| `filtered_journal.json` | `c2a592b0fe334e588e533d889ed7a131578dc8711ff51f80eb9178f468e0fc11` |
| Official-score ledger file | `fd5736395b11f52535b87bfd83d20fffbc8a8ccedf3e16215ebb9de9858eb9d9` |
| Official-score ledger content receipt | `248649d0896589e2ff413fb20ee20fc02509f7ebffb3814105da80904f49fe4c` |

The companion machine-readable record is `PAPER_RECORD_tournament_attempt_002.json`.
