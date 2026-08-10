# Fixed Holdout Evaluation

This mode separates search from final comparison:

1. `prepare` copies the public task files into `train_view/input` and writes
   hidden labels into a different `evaluator_view`.
2. The MLEvolve training container mounts only `train_view`. Its generated
   programs can use K-fold or ordinary validation inside the labeled training
   rows, but cannot read the fixed holdout labels.
3. Internal metrics are `search_only`. They may guide tree exploration, but
   they are not reported as the final comparison.
4. Before terminal evaluation, the solver freezes one selected node using only
   its internal search metric and writes a v3 evaluation request.
5. After the run stops, a separate evaluator scores the frozen submissions for
   diagnostics, but the fixed-holdout oracle is not allowed to replace the
   preselected system node.

This prevents holdout-label leakage. It does not prohibit using the unlabeled
holdout features to produce predictions, which is required by the competition
submission protocol.

## Prepare immutable views

```bash
cd mlevolve
python -m fixed_holdout.prepare \
  --dataset-root /workspace/nautilus/mlevolve/data \
  --output-root /workspace/fixed_holdout \
  --task spooky-author-identification \
  --task leaf-classification \
  --task aerial-cactus-identification
```

The training process must use `config/config_run_forest_fixed_holdout.yaml` and
pass the generated `train_view/input` and `train_view/fixed_holdout_manifest.json`
paths. Its container must not mount the split root or `evaluator_view`.

## Prepared split inventory

The July 2026 preparation run produced these immutable splits under the PVC root
`/workspace/runforest-fixed-holdout-v2`. The matching local inspection copy is
under `/private/tmp/runforest-fixed-holdout-built`; that local path is temporary
and is not the durable source of record.

| Task | Holdout rows | Metric | Split ID | Public train-view SHA-256 |
| --- | ---: | --- | --- | --- |
| Aerial Cactus | 3,325 | binary ROC AUC | `aerial-cactus-identification-6217a0196966-2089629209a8` | `6217a0196966b2b35ed415dbbf9bf82f958dc6f134bb744a119521479b00868c` |
| Leaf Classification | 99 | multiclass log loss | `leaf-classification-73e8353188dc-c33202487f61` | `73e8353188dc456f7560ec1a98e3393e658f86fcff87c7aa257ef43c7dc7d7fd` |
| Spooky Author | 1,958 | multiclass log loss | `spooky-author-identification-7edc8feae7e5-d0f035fed1ac` | `7edc8feae7e58fc6ab6700fe3a235bd950692251b114d03c475f638df7be5f70` |

Only each split's `train_view` subpath is mounted into a training Pod. The
evaluator receives `evaluator_view` separately. New York Taxi has a generic RMSE
adapter and synthetic coverage, but no durable real split was materialized in
this preparation run.

## Score a completed run

Run this in a separate evaluator container that can read hidden labels:

```bash
python -m fixed_holdout.score_run \
  --manifest /workspace/fixed_holdout/TASK/SPLIT/evaluator_view/fixed_holdout_manifest.json \
  --submission-dir /workspace/runs/RUN/workspace/submission \
  --journal /workspace/runs/RUN/logs/journal.json \
  --output /workspace/runs/RUN/logs/fixed_holdout_scores.json
```

Do not expose `fixed_holdout_scores.json` to an active search. Reusing the same
holdout score to choose further changes turns it into a tuning set. For a final
paper claim, keep a second untouched test set or score only frozen runs.

## Optional official Kaggle terminal score

`fixed_holdout.kaggle_terminal` integrates an official competition score as a
post-run authority. It submits exactly the node already frozen in
`fixed_holdout_evaluation_request.json`; it never submits every candidate and
never returns the leaderboard score to the search loop. This preserves the
competition test set as a terminal test and avoids spending the submission
quota on evolutionary tuning.

The evaluator spec is immutable and self-hashed:

```json
{
  "schema": "mlevolve_kaggle_terminal_evaluator_v1",
  "task_id": "leaf-classification",
  "competition": "leaf-classification",
  "metric": "multiclass_log_loss",
  "maximize": false,
  "sample_submission": "/workspace/data/leaf/sample_submission.csv",
  "id_column": "id",
  "prediction_kind": "probability",
  "score_field_preference": ["privateScore", "publicScore"],
  "poll_seconds": 15,
  "poll_timeout_seconds": 1800,
  "spec_hash": "SHA256_OF_ALL_OTHER_FIELDS"
}
```

Run the scorer in a separate CPU environment after the GPU training Job exits.
The Kaggle CLI obtains credentials from its normal environment; credentials are
never stored in the spec or experiment manifest.

```bash
python -m fixed_holdout.kaggle_terminal score \
  --spec /workspace/evaluators/leaf-kaggle.json \
  --request /workspace/runs/RUN/logs/fixed_holdout_evaluation_request.json \
  --work-dir /workspace/runs/RUN/official-kaggle \
  --output /workspace/runs/RUN/OFFICIAL_SCORE_REPORT.json

python -m fixed_holdout.kaggle_terminal measurement \
  --base /workspace/runs/RUN/MEASUREMENT.json \
  --report /workspace/runs/RUN/OFFICIAL_SCORE_REPORT.json \
  --output /workspace/runs/RUN/OFFICIAL_MEASUREMENT.json
```

The original `MEASUREMENT.json` remains immutable. Analysis opts into official
authority explicitly with `analyze_results.py --score-authority official`.
An official score is required for official-memory promotion, but is not by
itself sufficient: leakage and safety audits still apply.

## Native official-test inference (all tasks)

Historical runs may retain only code or predictions for an internal holdout;
those runs must be reproduced once on the competition's public train/test
files.  New runs should enable `official_submission` instead.  The task release
must expose its complete public train set, complete unlabeled official test set,
and exact `sample_submission.csv` to every candidate execution.

The framework never hard-codes a task's row count or target columns.  For each
candidate it streams the task-specific sample and output together, requiring
identical columns, ID order, and row count.  It also checks finite numeric
values, probability bounds, and multiclass row sums when applicable.  The
resulting Host receipt binds candidate code SHA, submission SHA, sample SHA,
official ID-set SHA, and the observed shape into `journal.json`.

```yaml
fixed_holdout:
  enabled: false
official_submission:
  enabled: true
  provider: kaggle
  competition: leaf-classification
  metric: multiclass_log_loss
  maximize: false
  sample_submission_path: /frozen/task/input/sample_submission.csv
  id_column: id
  prediction_kind: multiclass_probability
```

The same contract supports regression (`prediction_kind: numeric`), binary
probability, and multiclass probability tasks.  The candidate computes its OOF
or validation metric and official-test predictions in the same process, so the
frozen selected submission requires no post-search training or inference.

End2End task releases use terminal evaluator kind
`deferred_official_kaggle_v1`.  The GPU Job exits after writing an immutable
`official_evaluation_request.json`; a separate CPU scorer then runs one
idempotent command:

```bash
python -m fixed_holdout.kaggle_terminal finalize \
  --spec /frozen/evaluators/task-kaggle.json \
  --request RUN/logs/official_evaluation_request.json \
  --base CONDITION/MEASUREMENT.json \
  --report CONDITION/OFFICIAL_SCORE_REPORT.json \
  --measurement CONDITION/OFFICIAL_MEASUREMENT.json \
  --work-dir CONDITION/official-kaggle
```

Only the submission selected using the internal search metric is uploaded.
The official score remains invisible to the search and becomes the terminal
reporting authority through the immutable overlay.

For an End2End manifest, `run_official_assignment.py` applies this finalizer to
one condition index.  Its CPU Indexed Job should use `parallelism: 1` (Kaggle
submission quotas are task/account scoped) even if the preceding GPU training
Job used several parallel workers.  The task's frozen `RUNTIME_SPEC.json` pins
the evaluator as `official_kaggle_evaluator_spec`; credentials remain outside
all manifests and artifacts.
