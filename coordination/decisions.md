# Decisions

This file records durable project decisions that Codex and Claude Code should preserve across tasks.

## Confirmed Decisions

- 2026-07-01: Use `coordination/` as a lightweight shared workspace between Codex and Claude Code.
- 2026-07-01: Keep active task/report/review files small and reusable; do not treat them as a permanent chat log.
- 2026-07-01: Keep archived task snapshots out of git by default to avoid repository bloat.
- 2026-07-03: Build and evaluate **SkillGraph-C / adapted-ordering** as a stronger baseline by adding MLE trace-order `prereq` edges. Keep it explicitly separate from faithful `SkillGraph-static`; do not describe trace-order `prereq` as part of the paper's static InitGraph.
- 2026-07-06: Treat the current hyperbolic artifact as **v0 bootstrap memory** from SkillGraph-C compact cards, not as final transition-level SOP distillation. Its radius uses `p_hat/n_use/level` proxy stats, not real `metric_delta`.
- 2026-07-06: Agentic hyperbolic runtime proceeds in two stages: V1 uses a separate `MemoryNavigator` that explores the map before code generation and injects a memory pack; V2 may later expose map tools directly to draft/improve/debug agents.
- 2026-07-06: Any claim that hyperbolic geometry itself improves retrieval requires a Flat-Twin agentic control with the same SOPs and tools. Current runtime scoring is not yet true Poincare/Lorentz distance retrieval.

## Proposed Decisions

Carried over from prior Claude Code work (in private auto-memory) — pending Codex review + user ratification before moving to Confirmed.

- 2026-07-01 (proposed): **Canonical base for skill-graph work = `beta2-skillgraph`** (current). For any future full-MLE-bench / multi-task branch, start from `beta2-skillgraph` and graft beta1's safety commits (forced_return guard + leak-check-on-every-metric) plus the quarantine `.gitignore`. Also: commit the 4 currently-untracked distillation graph builders so a fresh clone keeps the pipeline. Blocked by the open scope question below.
- 2026-07-01 (proposed): **Distillation / memory source gate = the 17 verified-clean runs only.** Selection is per-run INDEX_BUG + train-on-val code-signature verification, NOT a date cutoff. (Already enforced in practice via quarantine; record as a durable rule.)
- 2026-07-01 (proposed): **Any deterministic LLM labeler/judge = DeepSeek + judge-all (no keyword gate).** Cross-the-semantic-gap accuracy dominates the cost saved by filtering; cost-saving variants have been measured and rejected. (Adoption tracker is the reference implementation.)
- 2026-07-01 (proposed): **P1 go/no-go gate = recall ≥10/15** against `small-data-transformer-finetuning/insight.md` (15 SOPs = 15 table rows, NOT 6 section headers). Do not start P2/P3 or write the paper if the gate fails.
- 2026-07-01 (proposed): **Two memory layers (trace memory vs Skill Graph) stay separate** — an ablation may swap global_memory→graph, but never both layers at once (confounds the variable).
- 2026-07-02 (proposed): **Static SkillGraph baseline scope = "w/o Graph Evolution" ablation only.** Build the init graph from the 17 clean runs; explicitly exclude online evolution (Insert/Merge/Split/Deprecate, edge reinforce/decay/prune, progressive-unlock, GRPO). The static graph tests "does structure help" only — do not claim evolution-driven gains.
- 2026-07-02 (proposed): **Build-time dedup = text-similarity + teacher-merge**, NOT the paper's neighbor-Jaccard Merge. Neighbor-Jaccard is degenerate on our single-task enhance-all graph (all generals share identical neighborhoods → would collapse to 1). Flagged as a documented deviation.
- 2026-07-02 (proposed): **Distillation teacher = DeepSeek v4-flash** (`api.deepseek.com`; `deepseek-chat` aliases to `deepseek-v4-flash`). Paper used OpenAI o3; substituted due to no o3 access. Recorded in `graph_build/graph.json` meta.
- 2026-07-02 (proposed, OPEN — blocks the injection layer): **Single-task graph cannot faithfully run the paper's retrieval.** Pick the injection path: (A) curated static chain [non-paper, fast], (B) expand to multi-task traces [faithful retrieval, wider scope — reverses the single-task narrowing], (C) build-time execution-order prereq [restores dependency-ordering value, not multi-task filtering]. Awaiting user direction.
- 2026-07-02 (proposed, OPEN — blocks Phase A): **Do not start the Phase A offline harness until (i) the 15-SOP gold standard passes a deterministic INDEX_BUG audit and the recall denominator is restricted to the leakage-independent subset, and (ii) a prevents/conflicts_with edge-extraction pass exists.** graph.json currently has 0 such edges and 0 nodes with `failure_modes`, so Experiment 2 would evaluate a non-existent capability. Verified findings in `claude_report.md` / `shared_memory.md` "Phase-A Plan Review".
- 2026-07-02 (proposed): **Ratify retrieval-injection path = C (build-time execution-order prereq) in decisions.md BEFORE building SkillGraph-C.** The experiment plan's B2 already commits to C, but the A/B/C item above remains OPEN — building the baseline before ratifying the choice risks a full redo if A or B is later chosen.
- 2026-07-02 (proposed, OPEN — user decision): **Paper title / thesis fork.** Either (a) keep "hyperbolic" load-bearing → add B5 (same features, flat/Euclidean space) and B6 (true Poincare/Lorentz embedding ranked by hyperbolic distance) so geometry is isolated and falsifiable (pre-register: if B6 does not beat B5 by >=X Rare Recall@5 with paired-bootstrap p<0.05 on clean-main, drop "hyperbolic" from the title); or (b) retitle to conflict-aware procedural skill memory and make the benchmark + conflict-edge annotation the contribution. The current pilot (linear feature scorer) cannot support a geometry claim.
