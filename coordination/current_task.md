# Current Task

## Status

Handoff / next implementation task for Claude Code.

The previous optimized SkillGraph baseline task is complete and recorded in
`coordination/claude_report.md` plus `coordination/shared_memory.md`. Codex has
since implemented the v0 Agentic Hyperbolic SOP Memory bootstrap and online V1
MemoryNavigator runtime, then created a senior-facing report and slide deck.

## Objective

Review the new hyperbolic SOP memory artifacts and continue with the next
implementation step: wire runtime configuration to use the v0 hyper graph in
agentic mode, then implement/evaluate true hyperbolic-distance scoring with a
Flat-Twin control.

## Recently Completed By Codex

- `paper-skills/hyper_memory/build_hyperbolic_memory.py` builds
  `hyperbolic-sop-memory-v0` from
  `paper-skills/distillation/graph_build/graph_skillgraph_c_trace_prereq.json`.
- Generated artifacts:
  - `paper-skills/hyper_memory/hyper_graph.json`
  - `paper-skills/hyper_memory/hyper_index.npz`
  - `paper-skills/hyper_memory/graph_builder_report.json`
- Build report: source graph 281 nodes / 1926 edges; hyper graph 865 nodes /
  3580 edges; validation booleans all true.
- `mlevolve/agents/memory/external_skill_memory.py` now supports the online V1
  agentic navigator tools: `inspect_map`, `navigate`, `expand`, `inspect_sop`,
  `check_conflicts`, `open_reference`.
- Navigator loop starts from `inspect_map(context)`, clamps
  `navigator_max_steps` to <=3, supports LLM tool-choice when configured, and
  has a deterministic fallback.
- Prompt injection section is now `## Agentic Hyperbolic Memory Navigation`.
- Senior-facing deliverables:
  - `coordination/hyperbolic_memory_senior_report.md`
  - `coordination/hyperbolic_memory_senior_slides.html`

## Boundaries To Preserve

- This is a v0 bootstrap from SkillGraph-C compact cards, not final
  transition-level SOP distillation.
- Radius currently uses proxy `p_hat/n_use/level`, not real `metric_delta`.
- `GraphBuilderAgent` is currently deterministic heuristic patching with
  programmatic validation, not an autonomous LLM builder.
- Runtime navigation scoring is still mostly lexical/feature/graph based; do
  not claim it is already true Poincare/Lorentz retrieval.
- Any paper claim that hyperbolic geometry itself helps requires a Flat-Twin
  agentic control using the same SOPs and tools.

## Suggested Next Steps

1. Add/verify config plumbing so `ExternalSkillMemoryLayer` can explicitly load
   `paper-skills/hyper_memory/hyper_graph.json` for
   `hyperbolic_agentic_memory` mode without hardcoded local assumptions.
2. Make `navigate` optionally score candidates by Poincare or Lorentz distance
   from `hyper_index.npz`, while preserving the current lexical/graph fallback.
3. Add a Flat-Twin scoring mode that uses the same navigator tools and SOP graph
   but replaces hyperbolic distance with Euclidean/feature distance.
4. Add focused tests for max 3 navigator steps, empty/fallback map behavior,
   SOP-only final retrieval, conflict classification, and reference budget
   enforcement.
5. Run the offline comparison: Flat retrieval, SkillGraph-C,
   hyperbolic automatic top-k, Agentic Hyperbolic Navigator, and Flat-Twin
   Agentic Navigator.

## Files To Inspect First

- `coordination/hyperbolic_memory_senior_report.md`
- `coordination/hyperbolic_memory_senior_slides.html`
- `paper-skills/hyper_memory/build_hyperbolic_memory.py`
- `paper-skills/hyper_memory/graph_builder_report.json`
- `paper-skills/hyper_memory/hyper_graph.json`
- `paper-skills/hyper_memory/hyper_index.npz`
- `mlevolve/agents/memory/external_skill_memory.py`
