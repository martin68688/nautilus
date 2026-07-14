# Fixed Holdout Evaluation

This mode separates search from final comparison:

1. `prepare` copies the public task files into `train_view/input` and writes
   hidden labels into a different `evaluator_view`.
2. The MLEvolve training container mounts only `train_view`. Its generated
   programs can use K-fold or ordinary validation inside the labeled training
   rows, but cannot read the fixed holdout labels.
3. Internal metrics are `search_only`. They may guide tree exploration, but
   they are not reported as the final comparison.
4. After the run stops, a separate evaluator mounts `evaluator_view` and scores
   every `submission_<node_id>.csv` once. The best node is selected from these
   fixed scores.

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
