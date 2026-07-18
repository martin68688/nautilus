# RunForest Causal Granularity Benchmark v2

## Purpose

The previous composite benchmark primarily scored the final SOP ID ranking.
That is useful for checking taxonomy relevance, but it cannot answer the main
Debug question: did the retriever find a historically successful repair for
the same failure mechanism, and did it expose the complete causal evidence to
the Agent?

Version 2 therefore separates two claims:

1. **Stage granularity:** Draft, Model Design, Improve, and Debug should receive
   L1, L2, L2, and L3 memory respectively.
2. **Causal Debug transfer:** when cross-run evidence exists, retrieve the
   complete parent failure -> code change -> successful child transition;
   when it does not exist, fall back to SOP-only.

## Leakage-resistant construction

- All Debug episodes come from real `debug_fixed` transitions in the current
  RunForest graph.
- Both the transition and its successful child must be clean, valid,
  rank-eligible, metric-bearing, and causally attached to at least one
  stage/task-compatible SOP.
- For every episode, the complete source run is removed from the candidate
  memory. Exact replay therefore cannot earn credit.
- Gold transitions must come from another run, share a benchmark-side
  silver-labeled primary failure mechanism, and have a compatible task family.
- Source runs are deterministically partitioned into dev and test. Primary
  numbers below use the 25 test episodes. This remains retrospective and is
  not a blind held-out test; the dev partition was not used for tuning.

The benchmark-side failure labels use a separate code path from the production
retriever's `FAILURE_SIGNATURES`, but they are semantically aligned silver
labels rather than independent expert annotations. They cover resource exhaustion, tensor
alignment, API incompatibility, definition order, path I/O, numerical
instability, fit-scope leakage, gradient lifecycle, and evaluation reuse.

## Data

| Item | Count |
|---|---:|
| Real causal Debug episodes | 38 |
| Source runs | 12 |
| Dev episodes | 13 |
| Test episodes | 25 |
| Episodes with cross-run causal evidence | 20 |
| Coverage-gap episodes requiring fallback | 18 |
| Stage-granularity episodes | 120 |

## Metrics

- `Granularity Precision@5`: fraction of Top-5 SOPs at the correct L1/L2/L3 level.
- `Detail Intrusion@5`: fraction of Top-5 SOPs from the wrong abstraction level.
- `Route Accuracy`: whether the method chose causal Tree or SOP-only fallback correctly.
- `Transition Hit@1/MRR`: whether a cross-run repair with the same failure mechanism was retrieved.
- `Fallback Accuracy`: whether the method abstained when no cross-run causal evidence existed.
- `Selective Decision Accuracy@1`: one combined operational metric. Covered
  episodes require a correct Top-1 transition; coverage gaps require fallback.

## Results

### Stage granularity, 120 episodes

| Method | Granularity Precision@5 | Detail Intrusion@5 | Empty result rate |
|---|---:|---:|---:|
| Ungated flat retrieval | 0.4467 | 0.5533 | 0.0000 |
| Tree-only | 0.7583 | 0.0000 | 0.2417 |
| SOP-only with taxonomy gate | 1.0000 | 0.0000 | 0.0000 |
| Stage Hybrid dynamic | **1.0000** | **0.0000** | **0.0000** |

The stage gate, not Tree by itself, removes wrong-granularity memories. SOP-only
already passes this narrow test, so this track does not claim a Hybrid advantage
over SOP-only; it verifies that adding Tree does not break the granularity gate.

### Causal Debug transfer, fixed source-run test partition

| Method | Route Acc. | Selective Decision Acc.@1 | Hit@1 | MRR | Fallback Acc. |
|---|---:|---:|---:|---:|---:|
| SOP-only | 0.5200 | 0.5200 | 0.0000 | 0.0000 | **1.0000** |
| Random transition | 0.4800 | 0.0000 | 0.0000 | 0.0167 | 0.0000 |
| Task-only transition | 0.4800 | 0.0800 | 0.1667 | 0.2639 | 0.0000 |
| Lexical transition | 0.4800 | 0.1200 | 0.2500 | 0.3472 | 0.0000 |
| Legacy successful-node Tree | 0.4800 | 0.1600 | 0.3333 | 0.4028 | 0.0000 |
| Causal Tree, fixed 0.75 weight | 0.7600 | 0.7200 | **0.7500** | **0.7500** | 0.6923 |
| Causal Tree, dynamic fallback | **0.8000** | **0.7600** | 0.5833 | 0.5833 | 0.9231 |
| Oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The primary retrieval comparison is against the legacy successful-node Tree,
not SOP-only: dynamic Hybrid improves route accuracy by 0.32, Selective
Decision Accuracy@1 by 0.60, and transition MRR by 0.1806. The random,
task-only, and pure lexical baselines establish that this is above a simple
chance, task-family-only, or wording-only effect.

Against fixed 75% causal Tree, dynamic fallback raises route accuracy and the
combined decision score by 0.04 and fallback accuracy by 0.2308, while reducing
covered-case MRR by 0.1667. This is an explicit precision/abstention tradeoff,
not an across-the-board win.

All evaluated production methods had zero unsafe transition escapes and zero
source-run escapes. These are conformance invariants enforced by candidate
filtering, not independent empirical safety discoveries.

## What this supports

The results support two bounded diagnostic claims:

1. Stage-aware taxonomy gates eliminate wrong-granularity memory injection on
   this frozen retrospective set.
2. Failure-matched causal transitions plus confidence-based fallback are more
   useful than SOP-only or generic successful-node Tree retrieval for deciding
   whether to inject a historical Debug repair.

## What this does not support

- It is retrospective; the graph was visible during system development.
- Failure mechanisms are deterministic, semantically aligned silver labels,
  not blinded expert gold.
- A retrieved successful transition is only an execution-relevant proxy. It
  does not prove that a new Agent adopts the repair or that downstream metric
  improves.
- Several failure mechanisms have only one run and therefore correctly become
  coverage gaps. More cross-run causal transitions are still needed.

The next claim-opening experiment should freeze a newly collected run-level
test set before any retriever tuning, then execute the generated repair code
under the existing leakage and provenance harness.

## Reproduction

```bash
PYTHONPATH=mlevolve:paper-skills/eval_composite_memory \
  python paper-skills/eval_composite_memory/run_causal_granularity_benchmark_v2.py

pytest -q tests/test_causal_granularity_benchmark_v2.py
```

Machine-readable results are in
`paper-skills/eval_composite_memory/reports/causal_granularity_report_v2.json`.
