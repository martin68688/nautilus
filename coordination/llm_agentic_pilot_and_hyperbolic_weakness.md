# LLM Agentic Navigator Pilot and Hyperbolic Weakness Notes

Date: 2026-07-07

## What Changed

- `paper-skills/eval_skill_memory/run_hyperbolic_retrieval_benchmark.py` now supports `--navigator-mode llm`.
- Deterministic mode remains the default for clean geometry ablations.
- LLM mode loads `mlevolve/config/config_hyperbolic_agentic.yaml` and passes cfg into `ExternalSkillMemoryLayer`, so `_llm_choose_navigation_action` calls the configured DeepSeek/OpenAI-compatible backend.
- Results now record `navigator_mode`, `agentic_pack_mode`, `llm_tool_calls`, `navigation_error`, and incremental JSONL rows.
- `mlevolve/llm/openai.py` now honors `OPENAI_COMPAT_TIMEOUT`; default remains 1200 seconds, but LLM retrieval pilots can use a shorter timeout.

## Run Notes

The local environment had proxy variables pointing to `127.0.0.1:7897`, but that proxy was not running. DeepSeek calls required running with proxy variables unset:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
    -u https_proxy -u http_proxy -u all_proxy \
    NO_PROXY='*' OPENAI_COMPAT_TIMEOUT=20 \
    python3 paper-skills/eval_skill_memory/run_hyperbolic_retrieval_benchmark.py \
      --navigator-mode llm \
      --benchmark paper-skills/eval_skill_memory/benchmarks/hyperbolic_sop_benchmark_llm_pilot8.jsonl \
      --systems agentic_lexical agentic_euclidean agentic_poincare agentic_flat_twin \
      --geometry-params '{"geometry_distance_norm":"minmax","geometry_distance_weight":0.35,"geometry_semantic_weight":0.8,"geometry_constraint_weight":0.5,"geometry_query_radius_quantile":0.5}' \
      --output paper-skills/eval_skill_memory/reports/hyperbolic_retrieval_results_llm_pilot8.jsonl
```

## LLM Pilot Result

Pilot benchmark: 8 hard test queries, 2 each from `rare_partial_clue`, `abstract_failure`, `minimal_context`, and `hard_method_set`.

DeepSeek execution status:

- Rows: 32
- Rows with successful LLM tool calls: 19
- Total LLM tool calls: 38
- Fallback rows due to connection errors: 13
- Error rows: 13

This proves the true LLM MemoryNavigator path works, but the API path was unstable enough that this pilot should be treated as behavioral smoke evidence, not a paper metric.

LLM pilot metrics:

| system | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| agentic_euclidean | 0.000 | 0.500 | 0.160 |
| agentic_flat_twin | 0.000 | 0.500 | 0.129 |
| agentic_lexical | 0.000 | 0.375 | 0.108 |
| agentic_poincare | 0.000 | 0.250 | 0.067 |

Same 8-query deterministic control:

| system | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| agentic_euclidean | 0.250 | 0.625 | 0.331 |
| agentic_flat_twin | 0.250 | 0.500 | 0.323 |
| agentic_lexical | 0.250 | 0.625 | 0.365 |
| agentic_poincare | 0.250 | 0.500 | 0.323 |

Interpretation: LLM navigation is wired and genuinely calls DeepSeek, but this first pilot is too small and too fallback-contaminated for a performance claim. It also did not rescue Poincare.

## Why Poincare Is Currently Weaker

Use the full 160-query deterministic hard benchmark for geometry diagnosis.

Key diagnostic numbers:

- Poincare loses while at least one other agentic system hits@5: 14 queries.
- Poincare unique hit@5 over lexical/Euclidean/Flat-Twin: 0 queries.
- Of those 14 loss cases, 11 have edge-band gold SOPs.
- Loss cases concentrate in low-specificity queries:
  - `minimal_context`: 6
  - `abstract_failure`: 5
  - `hard_method_set`: 3

The main failure mode looks like radial mismatch:

- Builder radius encodes a proxy reliability/core prior:
  `radius = 0.08 + 0.84 * (1 - core_score)`,
  where `core_score` depends mostly on `p_hat`, `n_use`, `level`, and a general bonus.
- Many useful rare SOPs have weak or missing success evidence, so they sit near the edge.
- Hard benchmark queries often provide abstract symptoms and little exact condition text.
- With weak angular signal, Poincare distance amplifies radial/edge effects. If the query radius is not well inferred, edge gold SOPs become easy to push out of top-5.
- Euclidean memory uses independent unit direction coordinates and does not carry this reliability-radius penalty. Flat-Twin uses the same Poincare coordinates but only Euclidean distance, so it is less aggressive about boundary/radial distortion.

Observed selected-radius means on the full hard benchmark:

- Gold radius mean: about 0.650
- Agentic lexical selected radius mean: 0.624
- Agentic Euclidean selected radius mean: 0.626
- Agentic Flat-Twin selected radius mean: 0.616
- Agentic Poincare selected radius mean: 0.585

Poincare is selecting slightly more central SOPs than the gold distribution, while most of its misses are edge-band gold SOPs. This supports the radial-mismatch diagnosis.

## Practical Next Fixes

1. Separate reliability from geometry radius.
   Keep confidence/support as a feature in scoring, but do not make it the Poincare radius unless the evaluation specifically wants reliability hierarchy.

2. Learn or infer query radius.
   Current query radius is a quantile heuristic. For abstract failure and minimal-context queries, the query should probably search edge-band SOPs more aggressively.

3. Improve angular embeddings.
   TF-IDF-SVD directions pass the quality gate, but masked/abstract queries still weaken angular alignment. Sentence embeddings or task-trained contrastive projections are likely needed.

4. Fix LLM navigator protocol before larger runs.
   The LLM path works, but needs retry/backoff, shorter timeout, and maybe a cheaper first action policy. Also pass radius-band hints through the final known-SOP sorting step.

5. Keep the claims separate.
   Current evidence supports agentic SOP memory as useful. It does not support a claim that Poincare geometry is better than same-coordinate Flat-Twin or independent Euclidean memory.
