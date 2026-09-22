# Spooky iterative-memory source-data notes

## Claim supported by the figure

The sequential case study supports a descriptive claim: the final audited
all-history replay achieved the best external score of the four displayed
stages, while the two bounded-memory stages were non-monotone. It does not
identify a causal effect of memory quantity because memory coverage, selected
replay target, runtime, and replay implementation changed together.

## Stage definitions and selected records

| Stage | Memory condition | Historical journals | Node | Public | Private | Selection basis |
|---|---|---:|---|---:|---:|---|
| Cold start | No memory | 0 | `78dc92019f5c...` | 0.34318 | 0.32388 | Clean full-code reproduction |
| Round 1 | One-run memory | 1 | `c5f8bebcf884...` | 0.31505 | 0.30095 | Clean synchronized inference |
| Round 2 | Two-run memory | 2 | `676a1b783714...` | 0.33494 | 0.30940 | Clean exact replay |
| All history | Audited memory plus replay fix | 56 | `591e9b4b2e10...` | 0.27986 | 0.25613 | Passed isolation audit; official replay |

The plotted metric is Kaggle multiclass log-loss, for which lower is better.
Public and private leaderboard values are external test evidence and are kept
separate from the heterogeneous internal validation metrics retained in the
CSV. No rows are missing from the four-stage plotting table. Each stage is one
selected run (`n = 1`), so no variance estimate, confidence interval, or
hypothesis test is reported.

## Final package composition

- 56 unique historical journals admitted from 88 discovered paths after 31
  duplicate paths were reconciled.
- 1,914 run nodes and 1,681 evidence items in the RunForest graph.
- 1,858 audited code nodes: 730 clean, 638 blocked, 79 protocol-biased, 209
  warning, and 202 audit-unavailable.
- 22 stratified SOPs in the graph; the final recipe bundle contains 20 distilled
  nodes (5 recipe, 5 tactic, 10 repair).
- The isolation audit and retrieval smoke test passed, and the clean DeBERTa
  replay target was retrieved.

## Reviewer-risk controls

- The figure explicitly labels the observations as single selected runs.
- The manuscript states that the comparison is sequential and descriptive.
- The two-round regression is retained rather than hidden.
- Blocked and warning history is counted but not described as positive memory.
- The 20.9% reduction is `(0.32388 - 0.25613) / 0.32388`; it is an external
  score comparison, not a randomized treatment effect.
