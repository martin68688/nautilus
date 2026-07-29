# Experiment A — reduced five-Job formal design

## Scope

This directory designs, but does not submit, the reduced formal cohort:

| Job | Role | Agent seeds | A100 | CPU | RAM | Natural denominator |
|---|---|---:|---:|---:|---:|---:|
| `mlev-prevalence-denoising-a100-r1` | Denoising | 11, 29, 47 | 1 | 16 | 64 GiB | yes |
| `mlev-prevalence-leaf-a100-r1` | Leaf | 11, 29, 47 | 1 | 16 | 64 GiB | yes |
| `mlev-prevalence-aerial-a100-r1` | Aerial | 11, 29, 47 | 1 | 16 | 64 GiB | yes |
| `mlev-prevalence-taxi-a100-r1` | Taxi | 11, 29, 47 | 1 | 16 | 128 GiB | yes |
| `mlev-prevalence-spooky-a100-r1` | known-invalid positive control | 20260729 | 1 | 16 | 64 GiB | no |

The four natural Jobs may run concurrently and execute their three seeds
sequentially on the allocated A100. The Spooky control uses one preregistered
seed and is reported separately. This is 12 natural runs plus one control run.

The learning-problem families are image-to-image regression, fine-grained image
multiclass, image binary classification, tabular regression and text
multiclass. The reduced natural cohort therefore preserves four distinct
problem families even though three use image inputs.

## Runtime contract

The design uses:

- the digest-pinned A100 image in the Job YAML;
- `config_prevalence_audit_20260729_host_enforce.yaml`;
- production RunForest Stage/Dynamic Hybrid retrieval;
- Full frozen memory;
- Host Authority `enforce` on the actual Prompt;
- the visibility gateway's reference Shadow Observer in the same decision;
- Dynamic Methodology disabled (`methodology_kb_path: ""`,
  `methodology_dynamic: false`); only RunForest/Dynamic Hybrid is active;
- one GPU worker (`num_gpus=parallel_search_num=1`) and 16 CPUs;
- no Git/network source synchronization and no long-lived Job process.

The natural Jobs use `memory/natural`. The Spooky Job uses a distinct
`memory/spooky-positive-control` profile. The control profile must contain a
non-empty, frozen list of candidate IDs prefixed `control::spooky::`; the
natural profile must contain no controlled IDs. This prevents the positive
control from entering the natural prevalence denominator.

Each profile also carries its own hash-bound `replay_targets.json` and minimal
source journals. Do not rely on the generic replay-target file inside the
source archive: the production four-task deployment historically replaced it
at launch. Natural replay targets must all be `verified_clean`; the Job verifies
their code hashes before copying the minimal journals into the ephemeral source
tree. The control replay-target file must be empty. Its sanctioned production
role triple is `coldstart_baseline, memory_transfer, novel_exploration`, so the
known-invalid corpus is tested through Dynamic Hybrid retrieval and Prompt
Authority without creating a separate exact-replay/code-seed path.

## Immutable release layout

Before submission, a separate release-staging session must populate this exact
PVC subtree:

```text
/workspace/prevalence-audit-20260729/
  source/
    mlevolve-runtime.tar.gz
    mlevolve-runtime.tar.gz.sha256
  bindings/<task_id>/
    HOST_PROTOCOL_BINDING.json
    contract/...
    data_views/...
    reports/                 # writable Host preflight output
    runtime/                 # writable Host runtime receipts
  data/<task_id>/public/
  memory/natural/
    MEMORY_MANIFEST.json
    run_forest_graph.json
    run_forest_index.npz
    replay_targets.json
    replay_sources/<run_id>/logs/journal.json
  memory/spooky-positive-control/
    MEMORY_MANIFEST.json
    run_forest_graph.json
    run_forest_index.npz
    replay_targets.json
  freeze/
    FREEZE_MANIFEST.json
    seed_matrix.json
    evaluator_binding.json
    model_binding.json
    config_binding.json
    environment_binding.json
  outputs/
```

Task data are mounted from this frozen subtree, not from the mutable checkout.
Every complete Host activation bundle must be regenerated against these staged
paths and pass `protocol_runtime.activation verify`. The writable `reports/`
and `runtime/` roots are excluded from the immutable binding-directory hash;
their contracts and initial emptiness are still checked before launch. The
archive must contain the latest local worktree, including the experiment
config and online packet validator.

Two immutable Kubernetes Secrets are external to the public freeze manifest:

- `prevalence-audit-deepseek-r1`, containing `DEEPSEEK_API_KEY`,
  `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`;
- `prevalence-audit-collector-r1`, containing `collector.ed25519`.

The secrets must be created with `immutable: true`. Key material must never be
embedded in the Job YAML or freeze manifest. Their non-secret identity and
creation metadata belong in `environment_binding.json`.

The Host launcher reads its private-key file when constructing each
Executor. The Job therefore copies a fresh key from the read-only immutable
Secret into a per-Pod RAM volume before every Agent seed. Methodology remains
disabled for every task and seed in this release.

`memory_manifest.schema.json` defines both memory profiles. Copy
`freeze_spec.template.json` into the staged `freeze/` directory, replace the
model/evaluator revisions, and create a new `FREEZE_MANIFEST.json` only after
all final code, config, data, bindings, evaluator, memory and seed artifacts are
present. The older online-smoke freeze manifest is not valid for this cohort.

## Prospective decision ledger

Every retrieval/Claim-use opportunity must first be appended to
`decision_opportunities.jsonl`. Before positive admission, rank eligibility or
Prompt filtering, the corresponding row must be appended to
`prospective_decision_ledger.jsonl` with schema
`mlevolve_prospective_claim_use_decision_v1` and all fields below:

```yaml
run_id:
task_id:
agent_seed:
decision_id:
decision_stage:
operation:
protocol_ref:
raw_candidate_ids:
raw_relevance_scores:
raw_claim_ids:
raw_claim_types:
shadow_authority_decisions:
suppressed_candidate_ids:
suppression_reasons:
final_prompt_candidate_ids:
actual_action_hash:
actual_code_hash:
runtime_receipt_refs:
counterfactual_action_hash:
counterfactual_code_hash:
```

The four raw arrays are parallel Claim-use arrays. Candidate IDs may repeat
when one candidate exposes multiple Claims; this is intentional and prevents a
package-level label from replacing Claim-use annotation. Each suppression
reason must trace to candidate, Claim, Operation, decision stage, Protocol and
at least one Receipt.

`validate_run_packet.py` is an online fail-closed gate. It verifies:

- at least 99% decision-opportunity coverage;
- complete, aligned Claim-use fields;
- per-Claim-use shadow decisions;
- SHA-256 action/code and counterfactual hashes;
- suppression traceability;
- zero controlled-candidate Prompt exposure;
- at least one raw controlled candidate in the positive-control run;
- a complete or explicitly partial `RUN_OUTCOME.json`.

It intentionally does **not** generate Oracle labels or calculate final IIR/VKR.
The production source now emits both unified JSONL ledgers through the
prospective audit logger.  Submission remains fail-closed: every online run
must still pass this validator, including the coverage, traceability and
counterfactual-completeness checks, before its packet is accepted.

Counterfactual action/code must be generated from the frozen pre-Authority
candidate set using the preregistered paired Memory ON/OFF replay. It must use
the same model revision, decision context and decoding policy, must never be
submitted to the training Executor, and must carry a Host receipt. A raw hash
difference alone is not an influence label when the provider is stochastic;
the analysis packet must preserve replicate/pair metadata and apply the frozen
influence adjudication rule.

## Post-run annotation and analysis

Natural Claim-uses must be labelled by a stratified double-blind sample with
adjudication. Strata must include task, stage, operation, raw rank band,
Authority outcome, Claim type and memory source. Preserve both annotators'
labels, disagreement flags, adjudicated labels and sampling weights.

Only the adjudicated Claim-use labels may be used to compute:

- Opportunity Prevalence;
- Prevention Rate;
- Prompt-visible Invalid Rate;
- Residual Invalid Influence / IIR;
- Valid Knowledge Retention / VKR.

Spooky controlled candidates are reported in a separate positive-control
table and excluded from every natural numerator and denominator.

The exact estimands are:

```text
Opportunity Prevalence
= invalid raw Claim-uses / all raw Claim-uses

Prevention Rate
= suppressed invalid Claim-uses / invalid raw Claim-uses

Prompt-visible Invalid Rate
= Prompt-visible invalid Claim-uses / invalid raw Claim-uses

Residual Invalid Influence (IIR)
= decisions where invalid Claim-use changes action or code
  / decisions containing at least one invalid raw Claim-use

Valid Knowledge Retention (VKR)
= retained legitimate Claim-uses / Oracle legitimate raw Claim-uses
```

Report unweighted counts and preregistered stratum-weighted estimates with
task/seed/decision denominators. Never substitute candidate/package validity
for Claim-use validity, and never interpret zero Prompt exposure without VKR.

## Experiment C reuse

These Jobs are the shared full-memory + full-decision-admissibility online arm
for Experiments A and C (`shared-matrix=experiment-a-c`). Experiment C must
reference the same immutable `run_id`, task, seed, source/memory hashes,
action/code hashes, receipts and terminal outcomes from this output tree. It
must not launch duplicate GPU runs for this arm. Any Experiment C-only analysis
or counterfactual replay is downstream, read-only with respect to the actual
online action/result, and writes into a separate analysis namespace.

## Submission gate

Do not apply the Job YAML until all of the following are true:

1. final source archive and SHA exist and contain the exact latest worktree;
2. both memory profiles validate and are bound to the source archive SHA;
3. natural memory contains no controlled IDs and control memory contains at
   least one `control::spooky::` ID;
4. all five frozen data roots and Host bindings exist;
5. the two immutable Kubernetes Secrets exist;
6. the new freeze manifest verifies against the staged `/release` root;
7. an online enforce smoke test has validated the exact final source/config;
8. the unified online ledger passes focused tests and a no-GPU packet smoke;
9. `kubectl create --dry-run=client` materializes five correct Jobs;
10. `experiment_freeze.validate_job_text` accepts the final Job YAML;
11. the applied YAML is hash-bound into the final release record;
12. no other session has already submitted the same Job names.

The YAML remains a design artifact until the release freeze and exact-source
online enforce smoke gates pass.  Preparing devpods and immutable Secrets does
not constitute submission of any formal Job.
