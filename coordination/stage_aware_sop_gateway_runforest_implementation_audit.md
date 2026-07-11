# Stage-Aware SOP Gateway + RunForest Implementation Audit

## Status

The approved implementation plan is complete at the engineering and offline-evaluation level. The resulting evidence does **not** permit a paper-grade retrieval, hyperbolic-geometry, adoption, or downstream-performance claim.

## Delivered Components

### Research specification

- `coordination/stage_aware_sop_gateway_runforest_research_note.md`
- Separates current deterministic geometry from the proposed hybrid system.
- Preserves `coldstart_baseline`, `memory_reproduction`, and `novel_exploration` contracts.
- Defines stage routes, safety gates, controls, metrics, and unsupported claims.

### Opt-in retrieval core

- `mlevolve/agents/memory/stage_aware_hybrid_memory.py`
- `mlevolve/config/config_run_forest_stage_hybrid.yaml`
- `run_forest_stage_hybrid` is explicit and fail-closed.
- Existing `run_forest_agentic` remains on its previous constructor path.
- Exact quotas are implemented for Draft, Improve, Debug, Evolution, and Fusion.
- Real `distills_to` edges build the SOP-to-Transition reverse index.
- Formal gateways require code-audited clean supporting execution.
- Blocked/quarantined/protocol-biased support is warning/repair evidence only.
- Agentic selection uses one structured call, validates IDs, and falls back deterministically.
- SOP-derived and Tree-derived rankings fuse over execution IDs using weighted RRF (`k=60`).

### Role isolation and repair non-regression

- Baseline and reproduction roles bypass hybrid retrieval in the shared helper.
- Improve, Debug, Evolution, Fusion, and aggregation calls propagate the parent role.
- Aggregation creates an explicit `novel_exploration` branch.
- Existing exact replay, blocked repair seed, preservation contract, fresh audit, FIFO deduplication, and max-two-attempt tests remain passing.

### Prompt and adoption audit

- Candidate, verified evidence, SOP-only reference, and risk-warning wording is separated.
- SearchNode stores thread-isolated `memory_navigation_trace`.
- Every hybrid injection record has retrieval channel, candidate class, gateway SOP, supporting transitions, reason, and lifecycle state.
- Post-run outcomes support `fully_adopted`, `partially_adopted`, `adopted_with_constraints`, `rejected_after_inspection`, and `not_adopted`.
- Retrieval lifecycle does not itself claim adoption.

### Held-out benchmark and controls

- 240 natural-language queries from 21 real runs.
- Split counts: train 120, dev 64, test 56.
- Splits are grouped by run ID; gold execution IDs and coordinates are absent from query text.
- Controls: no memory, SOP-only, Tree-only, naive concat, stage hybrid, Flat-Twin hybrid, and independently built Euclidean memory.
- Stage-level paired bootstrap gates and geometry-specific gates are reported.

### Online-control readiness

- `run_runforest_online_matrix.py` accepts all seven same-batch conditions.
- Default remains one `stage_hybrid` condition to avoid accidentally multiplying a production job by seven.
- `--dry-run` emits one manifest row and distinct log path per task/condition.
- No online training was launched by this implementation audit.

## Verification Evidence

- Full local test suite: `91 passed`.
- Python compilation passed for all changed Python entry points.
- No-GPU preflight checks passed:
  - structured config;
  - original cold-start SHA256 `5ecbdc00023227e75840f59104c9f5be58ae9efd403beb3d6c5cff894d49b0ff`;
  - source-allowlisted/code-audited graph provenance;
  - 20 runtime route cases (four controls by five stages);
  - zero blocked positive candidates;
  - held-out benchmark validation;
  - offline claim-gate computation.
- Seven-condition online matrix dry run emitted seven planned rows with condition-specific logs.

## Held-Out Results

Test split contains 56 queries.

| Control | Gateway MRR | Transition R@5 | Execution MRR | Evidence precision | Blocked positive |
|---|---:|---:|---:|---:|---:|
| no_memory | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| sop_only | 0.3899 | 0.1250 | 0.0500 | 0.0438 | 0.00 |
| tree_only | 0.0000 | 0.0000 | 0.3741 | 0.0000 | 0.00 |
| naive_concat | 0.3899 | 0.1250 | 0.1151 | 0.0438 | 0.00 |
| stage_hybrid | 0.3899 | 0.1250 | 0.3670 | 0.0438 | 0.00 |
| flat_twin_hybrid | 0.3899 | 0.1250 | 0.3709 | 0.0438 | 0.00 |
| independent_euclidean | 0.3899 | 0.1250 | 0.3380 | 0.0438 | 0.00 |

Stage Hybrid does not beat Tree-only overall. It also does not beat Flat-Twin. Improve has a positive MRR delta over its best single channel, but it is not statistically significant (`p=0.1489`). Draft, Debug, and Evolution do not pass their stage gates.

The original Draft stage comparison is now retained only as a transition-level
diagnostic. It used mutually exclusive sibling-child gold labels and retrieval
memory containing evaluated runs, so it cannot evaluate the three-role Draft
protocol. The replacement fixes `coldstart_baseline` and `memory_reproduction`,
changes only `novel_exploration`, uses multi-gold method families, and removes
held-out runs from both Tree and SOP-gateway retrieval. Its small test split is
not claim-grade; see `coordination/three_role_draft_tree_vs_hybrid_report.md`.

## Claim Decision

- Offline stage-aware retrieval claim: **rejected**.
- Hyperbolic geometry claim: **rejected**.
- Online downstream improvement claim: **unavailable/rejected** because no same-batch online run has completed.
- Adoption precision claim: **unavailable** offline.
- Safety result: zero blocked positive candidates in the held-out evaluator and no-GPU route preflight.

The honest interpretation is that the architecture and auditability are now testable, but the current retrieval weights and execution expansion do not outperform the strongest Tree-only control. Future work should improve routing using dev-only tuning, then rerun the unchanged test and online controls. Test results must not be optimized directly.
