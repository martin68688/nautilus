# RunForest Strict Decision-Point Benchmark

## Purpose

This benchmark evaluates retrieval for independently stated ML decisions. It
does not use a historical parent as the query or its actual child as gold. The
current revision incorporates an independent ClaudeAgent audit of the original
60-query diagnostic.

## Audit-Driven Corrections

The strict set contains 29 of 60 deterministic convenience-seed decision
points rather than preserving an artificial target. This is not a random or
representative task sample. Seed points are excluded when any of these
conditions holds:

- any other retained point has the same unordered gold set;
- a gold SOP is not explicitly compatible with the query task family;
- a gold SOP lacks evidence satisfying the runtime positive-memory predicate;
- five blocked RunNodes from the same task are unavailable.

The retained counts are `10/5/5/3/3/3` across six task-family labels. The
largest stratum therefore contributes 34% of the micro average, while four
strata contain at most five points. Family and stage summaries are descriptive
only. The full 286-candidate results are not directly comparable to older
benchmarks that used smaller, preselected pools.

Each query contains the full 281-SOP inventory plus five deterministically
selected blocked RunNodes from that query's own task. Candidate pools are not
preselected by BM25, TF-IDF, dense similarity, or graph retrieval. The query
contains no run, node, transition, parent, child, local-best, or trajectory
coordinate.

## Labels

Each query has three graded SOP seeds (3/2/1). Every seed is backed by at least
one transition whose child is clean, positive-eligible, paper-grade-eligible,
rank-eligible, valid, non-buggy, non-quarantined, non-protocol-biased, and has a
numeric metric. Gold sets are globally unique. Gold requires the query's
explicit task-family tag; a generic `general` tag is not accepted as an escape
hatch.

These remain silver labels. The blind packet contains no relevance or safety
answer, but two independent annotators and adjudication are still required.
Therefore `offline_retrieval_claim_allowed=false`.

## Systems

The lexical and dense controls are evaluated in symmetric pairs:

- `bm25_unfiltered` and `bm25_safety_filtered`;
- `tfidf_unfiltered` and `tfidf_safety_filtered`;
- `lsa_dense_unfiltered` and `lsa_dense_safety_filtered`;
- `minilm_dense_unfiltered` and `minilm_dense_safety_filtered`;
- `tree_only_mapped_no_task` and `tree_only_mapped_task_aware`;
- `legacy_stage_gateway`, which preserves the old field-aware lexical scorer
  plus the clean-evidence gate for an exact historical comparison;
- `stage_hybrid_sop`, which calls the shared production v2 implementation with
  the SOP and Tree channels, stage/task gates, geometry, weighted RRF, and the
  final clean-evidence gate;
- deterministic random and safe oracle controls.

The safety-filtered text variants use the same positive-evidence gate. None is
given a target abstraction level. Their zero non-admissible rate is a property
of that explicit gate, not evidence that the underlying scorer learned safety.

## Metrics And Statistics

- graded nDCG@10 with exponential gains for labels 3/2/1;
- Adoption AP@10, where only labels 2 or 3 count as strongly relevant and form
  the AP denominator;
- blocked RunNode rate@10;
- unsupported SOP rate@10;
- overall non-admissible rate@10;
- method-family diversity and latency.

Paired nonparametric bootstrap with 10,000 samples estimates 95% confidence
intervals. Two-sided p-values use a paired random sign-flip null test, followed
by Holm correction. Per-family and per-stage tables are observational only.

## Current Silver Diagnostic

| Method | nDCG@10 | AP@10 | blocked RunNode | unsupported SOP |
|---|---:|---:|---:|---:|
| BM25 unfiltered | 0.2333 | 0.1851 | 0.2828 | 0.2000 |
| BM25 + safety gate | 0.3200 | 0.2723 | 0.0000 | 0.0000 |
| TF-IDF unfiltered | 0.3050 | 0.2487 | 0.1897 | 0.2000 |
| TF-IDF + safety gate | 0.3454 | 0.2886 | 0.0000 | 0.0000 |
| LSA unfiltered | 0.3145 | 0.2309 | 0.2586 | 0.2000 |
| LSA + safety gate | 0.3995 | 0.3273 | 0.0000 | 0.0000 |
| MiniLM unfiltered | 0.2912 | 0.2126 | 0.1690 | 0.2241 |
| MiniLM + safety gate | 0.4347 | 0.3476 | 0.0000 | 0.0000 |
| Tree mapped, no explicit task bonus | 0.2253 | 0.1330 | 0.0000 | 0.0793 |
| Tree mapped, task aware | 0.2925 | 0.1862 | 0.0000 | 0.0448 |
| Legacy field-aware gateway | 0.3543 | 0.2905 | 0.0000 | 0.0000 |
| Production Stage Hybrid v2 | 0.4522 | 0.3861 | 0.0000 | 0.0000 |

The safety-gate effect is positive for all four text scorers in this silver
diagnostic. Production Stage Hybrid v2 exceeds the legacy gateway by 0.0980
nDCG@10 in point estimate, but its 95% interval [-0.0093, 0.2056] crosses zero
and the Holm-adjusted p-value is 0.2640. It exceeds MiniLM plus safety filtering
by 0.0175 in point estimate, also without statistical support (95% interval
[-0.0941, 0.1248], Holm-adjusted p=0.7605). The diagnostic verifies that the
production row now exercises the intended dual-channel mechanism; it does not
establish a relevance or downstream winner.

## Remaining Limitations

1. The same author wrote the decision descriptions and seeded the silver labels,
   so construction alignment remains possible.
2. Only 29 of 60 convenience seeds survive, with an uneven `10/5/5/3/3/3`
   distribution; task-family and stage claims are disabled.
3. The safety-filtered rows test a deterministic gate, not learned safety.
4. Tree retrieval still has access to graph structure; task-aware and no-task
   variants expose, but do not eliminate, that additional information channel.
5. Offline retrieval does not measure Agent adoption, generated code, or
   downstream model quality.
6. Human blind annotation and concurrent online controls remain necessary.
7. `image_classification` and `tabular_multiclass` are distinct decision-family
   labels but share the underlying `leaf-classification` task domain. Global
   gold deduplication prevents duplicated targets, but these strata are not
   independent task domains.

## Reproduction

```bash
python paper-skills/eval_skill_memory/build_decision_point_benchmark.py
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python paper-skills/eval_skill_memory/evaluate_decision_point_benchmark.py
pytest -q tests/test_decision_point_benchmark.py
```

Use `requirements-decision-point-benchmark.txt` in an isolated environment for
the MiniLM baseline. The report records the actual Python, NumPy, platform, and
sentence-transformer availability.
