# Codex Implementation Sync — Hyperbolic SOP Memory Fixes

## Status

Implemented locally on 2026-07-06. Not committed or pushed.

## Key Fixes

- Flat-Twin main control is now same-coordinate: `flat_twin == poincare`; runtime switches only the distance function.
- Runtime supports `scoring_mode=lexical | poincare | flat_twin`.
- Poincare scoring uses the standard Poincare ball distance; Flat-Twin scoring uses Euclidean distance over the exact same coordinates.
- Builder now defaults to `TF-IDF-SVD d=16`, saves `hyper_text_model.joblib`, writes `coordinate_quality_report.json`, and records coordinate quality gate results.
- Builder records provenance status. Current artifact is `uncertified_bootstrap` because the input graph lacks `source_runs / allowlist / leak_verified` and per-node source evidence.
- Dedicated experiment profile added: `mlevolve/config/config_hyperbolic_agentic.yaml`.
- `load_cfg` supports `extends:` profiles and the `MLEVOLVE_CONFIG` env var.
- Safety fixes ported: forced-return guard, leakage check on every metric-bearing node, and cross-fold/OOF embedding leakage prompt.
- Paired-bootstrap evaluation helper added for the pre-registered geometry claim gate.

## Current Rebuilt Artifact

- `paper-skills/hyper_memory/hyper_graph.json`
- `paper-skills/hyper_memory/hyper_index.npz`
- `paper-skills/hyper_memory/hyper_text_model.joblib`
- `paper-skills/hyper_memory/coordinate_quality_report.json`
- `paper-skills/hyper_memory/graph_builder_report.json`

Rebuild output:

- nodes: 865
- edges: 3580
- Poincare shape: `(281, 16)`
- `flat_twin == poincare`: true
- coordinate quality: passed
- provenance: `uncertified_bootstrap`
- paper-grade provenance: false

## Claim Gate

Hyperbolic geometry can only be claimed if all three hold:

- Agentic Poincare Rare Recall@5 beats same-coordinate Flat-Twin by at least 5 percentage points.
- Paired bootstrap over queries gives `p < 0.05`.
- Condition Precision does not decline.

## Verification Run

```bash
python3 -m py_compile paper-skills/hyper_memory/build_hyperbolic_memory.py paper-skills/hyper_memory/evaluate_hyperbolic_ablation.py mlevolve/agents/memory/external_skill_memory.py mlevolve/config/__init__.py mlevolve/engine/agent_search.py mlevolve/analysis/adoption_tracker.py mlevolve/agents/data_leakage_agent.py mlevolve/agents/triggers.py mlevolve/engine/node_selection.py mlevolve/agents/result_parse_agent.py tests/test_hyperbolic_memory.py
python3 -m pytest tests/test_hyperbolic_memory.py -q
python3 paper-skills/hyper_memory/build_hyperbolic_memory.py
```

Result: 5 tests passed; rebuilt artifact remains bootstrap-only due to missing source provenance.
