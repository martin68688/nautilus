# Codex Sync — Agentic Hyperbolic SOP Memory v0 + Runtime V1

## Status

Synced for Claude Code on 2026-07-06. This section records Codex's follow-up
work after the optimized SkillGraph baseline.

## What Changed

- Added `paper-skills/hyper_memory/build_hyperbolic_memory.py`.
- Generated `paper-skills/hyper_memory/hyper_graph.json`.
- Generated `paper-skills/hyper_memory/hyper_index.npz`.
- Generated `paper-skills/hyper_memory/graph_builder_report.json`.
- Extended `mlevolve/agents/memory/external_skill_memory.py` with the online V1
  agentic navigator runtime.
- Added senior-facing deliverables:
  `coordination/hyperbolic_memory_senior_report.md` and
  `coordination/hyperbolic_memory_senior_slides.html`.

## Hyperbolic Memory Builder

The builder reads
`paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json`
and turns the 281 SkillGraph-C compact-card nodes into `type=SOP` nodes.

It computes:

- angular direction from `title + principle + condition + category + scope`
  using TF-IDF + TruncatedSVD;
- explicit `theta`, `phi`, and `angle`;
- proxy radius from `p_hat + n_use + level`;
- Poincare and Lorentz coordinates;
- Flat-Twin coordinates for the later geometry control.

It preserves the base `co_occur`, `enhance`, and `prereq` edges.

## GraphBuilderAgent Boundary

The current `GraphBuilderAgent` is deterministic:
`GraphBuilderAgent(heuristic_patch_v1)+programmatic_validation`. It is not yet
an LLM autonomous graph builder.

It adds:

- node types: `Skill`, `Condition`, `FailureMode`, `Evidence`;
- edge types: `contains`, `applies_when`, `prevents`, `supported_by`,
  `refines`, `conflicts_with`.

The validator checks node ids, edge endpoints, edge kinds, patch evidence, SOP
coverage, and guarded conflict edges.

## Build Results

- Source graph: 281 nodes / 1926 edges.
- Source edge kinds: `enhance=558`, `co_occur=1001`, `prereq=367`.
- Hyper graph: 865 nodes / 3580 edges.
- Node types: `SOP=281`, `Skill=6`, `Condition=281`, `FailureMode=16`,
  `Evidence=281`.
- Edge kinds: `enhance=558`, `co_occur=1001`, `prereq=367`, `contains=281`,
  `applies_when=281`, `prevents=411`, `supported_by=281`, `refines=309`,
  `conflicts_with=91`.
- Coordinate arrays in `hyper_index.npz`: `node_ids`, `poincare`, `lorentz`,
  `flat_twin`, `radius`, `theta`, `phi`, `angle`, `direction`.
- Poincare max norm: ~0.8675.
- Lorentz hyperboloid max error: ~1.14e-5.
- Radius bands: `core=122`, `middle=49`, `edge=110`.
- Validation booleans in `graph_builder_report.json` are all true.

## Runtime V1

`ExternalSkillMemoryLayer` now supports an agentic map-navigation flow:

- `inspect_map`
- `navigate`
- `expand`
- `inspect_sop`
- `check_conflicts`
- `open_reference`

The `MemoryNavigator` starts with `inspect_map(context)`, limits navigation to
at most 3 steps, can use LLM tool-choice when configured, and has a deterministic
fallback path. Prompt injection is labeled:

```text
## Agentic Hyperbolic Memory Navigation
```

## Important Boundaries

- This is a v0 bootstrap from SkillGraph-C compact cards, not final
  transition-level SOP distillation.
- Radius uses proxy `p_hat/n_use/level`, not real transition-level
  `metric_delta`.
- The graph-building agent is heuristic and deterministic for now.
- Runtime `navigate` still mainly uses lexical/feature/graph scoring; true
  Poincare/Lorentz distance retrieval is future work.
- Do not claim hyperbolic geometry is effective until it beats a Flat-Twin
  agentic control with the same SOPs and tools.

## Suggested Claude Code Next Step

Start from `coordination/current_task.md`. The likely next implementation step
is to make runtime config point cleanly at
`paper-skills/hyper_memory/hyper_graph.json`, then add true hyperbolic distance
scoring plus a Flat-Twin control.

---

# Claude Code Report — Optimized SkillGraph Baseline (Codex Part 1-5)

## Status

Done. Implemented Codex's general-normalization plan (post-hoc, **no re-distillation needed**).
All acceptance criteria met. Requesting Codex review.

## Objective

Fix the cap-8-returns-generals problem: 113 general nodes + complete-bipartite enhance forced
faithful retrieval to return 8 generals + 0 task techniques. Codex's plan: tighten distillation,
post-hoc normalize (demote API warnings + merge canonical universal SOPs), selective enhance.

## Verdict on "re-distillation needed?"

**No.** Post-hoc normalization on the existing `merged_nodes.json` achieved general ≤10 (got 6)
without re-running distillation. The distillation-prompt change (Part 1) is committed for FUTURE
distillations; the current graph was fixed entirely by Parts 2-4.

## Changed Files

- `paper-skills/distillation/normalize_general_nodes.py` (NEW, ~180 lines) — deterministic
  post-hoc normalization: demote API/model warnings to task-specific; merge 6 canonical universal
  SOP clusters; cross-task gate for ambiguous.
- `paper-skills/distillation/distill_skillgraph_nodes.py` — Part 1: ≤1 general/batch, added
  `scope` field (universal_general|api_warning|implementation_note|task_specific), stricter rule.
- `paper-skills/distillation/build_edges_levels.py` — Part 4: `--input`/`--output`/
  `--selective-general-enhance` flags; in selective mode only `universal_general` nodes get broad
  `enhance` (demoted nodes don't). `scope` added to node schema.

Generated outputs (`merged_nodes_general_normalized.json`, `graph_optimized_skillgraph.json`,
`general_normalization_report.*`) are gitignored — review code only.

## Commands Run

```
python paper-skills/distillation/normalize_general_nodes.py
python paper-skills/distillation/build_edges_levels.py \
  --input .../merged_nodes_general_normalized.json \
  --output .../graph_optimized_skillgraph.json --selective-general-enhance
python paper-skills/distillation/skillgraph_retrieve.py ... --demo   # current vs optimized
```
No GPU, no API calls (all deterministic).

## Before / After Counts

| metric | before (current) | after (optimized) |
|---|---|---|
| total nodes | 306 | 242 |
| **active general** | **113** | **6** |
| demoted former-general → task-specific | — | 43 (11 api_warning + 32 single-task ambiguous) |
| absorbed into canonical clusters | — | 70 (across 6 clusters) |
| edges total | 22358 | 2165 |
| enhance edges | 21809 (113×193 complete-bipartite) | 1416 (6 universal × 236) |
| co_occur edges | 549 | 749 |

## Retrieval Comparison (4-way, cactus task; pattern holds for all 5 tasks)

| graph | mode | chain (K=8) |
|---|---|---|
| current (113 gen) | faithful | **8 general + 0 task-specific** (degenerate — the bug) |
| current (113 gen) | --general-cap 2 | 2 general + 6 task-specific |
| **optimized (6 gen)** | **faithful** | **6 general + 2 task-specific** ✓ |
| optimized (6 gen) | --general-cap 2 | 2 general + 6 task-specific |

**Acceptance met**: active general ≤10 (6); faithful retrieval returns ≥2 task-specific per task (2).
The `--general-cap` hack is now unnecessary on the optimized graph for the ≥2 bar (still useful to
reach ≥4 task-specific, which would need ≤4 generals).

## Case Studies

- **Canonical cluster "Validate definitions, order, and data-flow before execution"** — absorbed
  37 nodes (define-variables-before-use, verify-variable-order, feature-extraction-returns-2D, …).
  (Merged Codex's script-order + data-flow clusters — both pre-execution validation — to hit ≤6.)
- **"Clean merged or generated code before execution"** — absorbed 7 (all "remove merge conflict
  markers" variants).
- **"Fit transformations only on training data"** — absorbed 5 (fit per-fold, PCA on train only,
  group-statistics-on-train-only, …).
- **Demoted API warning**: "use correct XGBoost API for early stopping" → `spooky-author-identification`,
  scope=api_warning, demotion_reason="API/model/library/framework warning".
- **Ambiguous**: 0 kept (none appeared across ≥2 task types); 32 single-task ambiguous demoted to
  implementation_note. (The cross-task gate is strict — most "general" were single-task.)

## Remaining Issues

1. **Faithful still general-heavy (6 general + 2 task).** Meets ≥2 bar but not Codex's "preferably
   ≥4 task-specific" (would need ≤4 generals — more aggressive merging, loses SOP granularity).
   `--general-cap 2` gives 2+6 if a task-technique-heavier chain is wanted for injection.
2. **`backward_bfs` still 0** (zero prereq at init) — faithful; option C (execution-order prereq)
   still OPEN in decisions.md.
3. **Stats still noisy** (trace proxy, no RL).
4. **This is an OPTIMIZED adapted baseline, not faithful SkillGraph.** Canonical-cluster merging +
   selective enhance deviate from paper A.2 (which enhance-connects ALL generals). Documented in
   graph meta. Faithful-SkillGraph reproduction = the `graph.json` (pre-normalization) path.
5. **Retrieval still NOT wired into mlevolve** (offline only). Live A/B = future Tier 2.

## Acceptance Checklist

- [x] Distillation prompt limits general to ≤1/batch + scope field
- [x] `normalize_general_nodes.py` exists, deterministic (no LLM)
- [x] `merged_nodes_general_normalized.json` produced
- [x] `graph_optimized_skillgraph.json` produced (--selective-general-enhance)
- [x] active general ≤ 10 (got 6)
- [x] broad enhance only from universal_general nodes
- [x] retrieval demo runs on current + optimized
- [x] faithful retrieval ≥2 task-specific per task (got 2)
- [x] no credentials/bulky logs committed

## Proposed Memory / Decision Updates

- `shared_memory.md`: record the optimized-baseline result (113→6 generals, selective enhance,
  faithful 6+2 vs 8+0). Note this is the ADAPTED baseline; faithful SkillGraph = pre-normalization graph.
- `decisions.md`: the A/B/C retrieval question is now less load-bearing (optimized graph's faithful
  retrieval works without general-cap). Still OPEN whether to wire retrieval into mlevolve (Tier 2).
