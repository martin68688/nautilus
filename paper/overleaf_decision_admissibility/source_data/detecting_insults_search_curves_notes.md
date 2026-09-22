# Detecting Insults search-curve source notes

## Figure claim

The figure describes how the best internal validation ROC–AUC found by each
search run changed with the number of search steps. It compares strict
zero-memory Novel search with cross-task memory-transfer search.

## Records included

- Novel: all five runs that entered candidate search — original attempt-004,
  rerun2 attempt-001, rerun3 attempt-000, AWS r5, and AWS r6. The partial
  rerun2 and AWS r5 trajectories are retained rather than discarded.
- Transfer: the five user-selected records — AWS runtime-bound r5 attempt-001,
  AWS runtime-bound r6 attempt-002, NRP v172 direct attempt-003, NRP v174 Sol
  attempt-001, and NRP v174 Luna attempt-004. The earlier NRP v172
  attempt-002 is intentionally excluded.
- Task `01a04413-6f86-7582-b5db-3b74c6076dd0` is the dispatch/monitoring entry
  for the Novel task and is not counted as a separate experiment.
- The figure uses neutral `Run 1`, `Run 2`, … labels. Infrastructure provider,
  GPU, runtime version, and model names are retained only in this provenance
  note and do not appear in the plotted panel.

## Processing

- A point is retained only when the journal reports a numeric metric,
  `is_valid == true`, and the node is not buggy. AWS r5/r6 additionally require
  `leakage_audit.rank_eligible == true`; seven protocol-biased AWS candidates
  are therefore excluded from 128 scored Novel candidates, leaving 121.
- The apparent AWS r6 value 0.938483 is excluded because the final authority
  audit marked it `protocol_biased` and ineligible for promotion/ranking. The
  certified AWS r6 maximum is 0.893969.
- A marker-free terminal anchor extends a running-best line to the journal's
  final search step when no later rank-eligible candidate exists. It does not
  invent or display an additional candidate score.
- The shared y-axis extends to 0.60 so the retained AWS r5 candidate at
  0.625531 is visible rather than silently clipped.
- Dots are the retained candidate scores and use one color per experimental
  condition; lines use restrained run-specific colors. `Final best` endpoint
  values are collected in compact keys below the dense score region rather
  than attached with crossing callouts. Lines are a shape-preserving PCHIP
  interpolation of the observed running maximum. The interpolation is for
  presentation only and cannot introduce a score above the observed running
  maximum.
- AWS r6 attempt-002 is read from its complete local journal and is shown with
  a solid line. The other four trajectories were reconstructed from archived
  journal dumps or timestamped task-audit snapshots; dashed lines and
  asterisks identify these records. No intermediate score is imputed.
- Infrastructure-only attempts without a valid scored candidate are excluded.

## Transfer endpoints shown

| Run | Best internal validation ROC–AUC |
|---|---:|
| AWS runtime-bound r5 attempt-001 | 0.954680 |
| AWS runtime-bound r6 attempt-002 | 0.943834 |
| NRP v172 direct attempt-003 | 0.930514 |
| NRP v174 Sol attempt-001 | 0.953293 |
| NRP v174 Luna attempt-004 | 0.954319 |

## Novel endpoints shown

| Run | Best admissible internal validation ROC–AUC |
|---|---:|
| Original attempt-004 | 0.957143 |
| Rerun2 attempt-001 | 0.941743 |
| Rerun3 attempt-000 | 0.952824 |
| AWS r5 | 0.884912 |
| AWS r6 | 0.893969 |

## Interpretation limits

These are internal validation ROC–AUC values, not official leaderboard scores.
Validation partitions and some runtime configurations differ between runs, so
the figure is descriptive and must not be interpreted as a controlled estimate
of the causal effect of transferred memory.

## Descriptive comparison used in the manuscript

- The first rank-eligible candidate appears at median global step 2 (range
  1--3) for transfer and step 4 (range 1--48) for Novel. All five transfer runs
  have a rank-eligible candidate by step 3, compared with two of five Novel
  runs.
- Using internal ROC--AUC 0.93 as a descriptive viable-candidate threshold, all
  five transfer runs reach the threshold by step 3. Three of five Novel runs
  reach it at steps 1, 4, and 5; the other two remain below it through terminal
  steps 44 and 137.
- Across retained candidate points, transfer has median 0.930514 and IQR
  0.025532 (59 points), versus median 0.921204 and IQR 0.041355 for Novel
  (121 points). Candidate points within a run are dependent and the unequal
  counts reflect different trajectory lengths, so no candidate-level
  inferential test is used.
- Across the five final running-best values per condition, transfer has median
  0.953293 and IQR 0.010484, versus median 0.941743 and IQR 0.058856 for Novel.
- Earlier threshold crossing is a search-step proxy for potential GPU and LLM
  savings under an early-stopping policy. Per-step GPU time, LLM token use, and
  execution contracts are not harmonized, so the experiment does not provide
  a measured cost ratio.

## Statistical metadata

- n definition: one search trajectory per substantive run; five Novel and
  five transfer records are shown.
- biological replicates: not applicable.
- technical replicates: independent search runs, but not matched seeds/folds.
- center statistic: none; individual runs are shown.
- spread/interval: none.
- test: none.
- multiple-comparison correction: not applicable.
- p-value display: none.
- metric: internal validation ROC–AUC, higher is better.
- train/validation/test split: run-specific; not harmonized across all records.
- number of seeds or folds: no matched seed/fold design across the displayed runs.
- confidence interval or variability definition: none; individual trajectories are shown.
- baseline definition: Novel search is the strict no-memory condition.
- source-data file: `detecting_insults_search_curves.csv`.
