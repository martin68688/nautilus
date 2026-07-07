# Hyperbolic SkillGraph Experiment Plan

This document is the implementation-facing experiment plan for proving that a
hyperbolic procedural skill graph adds value beyond a flat skill library and the
static SkillGraph baseline.

Codex role: research design and review.
Claude Code role: implement benchmark builders, retrievers, evaluation scripts,
and run the first offline experiments.

## Core Claim

SkillGraph proves that skills need structure. Our claim is narrower and sharper:
MLE procedural skills need condition, failure, and conflict structure, and a
hyperbolic layout gives those structures a useful geometry.

The experiments must not merely show that "hyperbolic is elegant". They must
show measurable gains on four properties:

1. Low-frequency but condition-matched skill recall.
2. Surface-conflict resolution.
3. Coherent method-set retrieval.
4. Better downstream agent behavior.

## Existing Repo Assets

- Historical spooky runs: `mlevolve/runs/`
- Clean-run warning and scope notes: `coordination/shared_memory.md`
- Static SkillGraph output: `paper-skills/distillation/graph_build/graph.json`
- SkillGraph builder scripts: `paper-skills/distillation/{distill_skillgraph_nodes.py,merge_nodes.py,build_edges_levels.py,build_references.py}`
- Branch traces: `paper-skills/distillation/traces/` currently has 93 branch markdown files.
- Hand-written 15 SOP ground truth: `paper-skills/experience_kb/small-data-transformer-finetuning/insight.md`
- Trace2Skill recall evaluator: `paper-skills/trace2skill_baseline/evaluate_recall.py`
- Existing flat retriever: `mlevolve/agents/memory/retriever.py`
- Existing coldstart methodology injection path: `mlevolve/engine/coldstart/knowledge.py`

Important: the 15-SOP ground truth has contamination risk in shared memory.
Before using it as a paper claim, audit it for INDEX_BUG / train-on-val leakage.
Until then, use it as a pilot benchmark and label results accordingly.

## Data Hygiene

Use three data tiers:

- `pilot`: use existing `graph_build/graph.json` and the 15 SOPs to quickly
  validate scripts.
- `clean-main`: use only the verified-clean run allowlist from
  `coordination/shared_memory.md`; exclude quarantined runs and contaminated KBs.
- `heldout`: leave at least one clean run out of graph construction and use it
  only for evaluation.

For paper-grade results, do not report numbers from `pilot` as final. The final
tables should use clean-main or leave-one-run-out.

Recommended split:

- Train: 12 clean runs for graph construction.
- Dev: 2 clean runs for hyperparameters and prompt templates.
- Test: 3 clean runs, frozen before final reporting.

If exact clean-run allowlist cannot be reconstructed from files, stop and report
the blocker instead of guessing.

## Compared Systems

Implement all retrievers behind one common interface:

```python
retrieve(query: str, k: int, context: dict | None = None) -> list[RetrievedSkill]
```

Each result should include:

- `skill_id`
- `title`
- `text`
- `rank`
- `score`
- `source_system`
- `evidence_refs`
- `condition`
- `failure_modes`
- `debug_info`

### B0 Flat Retrieval

Baseline equivalent to the current flat memory / RAG style:

- BM25 only
- vector only, if local embedding model is available
- BM25 + vector RRF, matching `mlevolve/agents/memory/retriever.py`

Corpus:

- `graph.json` node text plus references, or
- skill documents from `paper-skills/experience_kb/`

### B1 Static SkillGraph

Use `graph_build/graph.json` exactly as the static SkillGraph init baseline:

- nodes: title, principle, condition, category
- edges: `enhance`, `co_occur`
- no online evolution
- no invented prereq edges

Retrieval:

- seed by lexical/vector match over node text
- expand over `enhance` and `co_occur`
- rank by seed score plus edge weights

Record this as "SkillGraph-static". It is expected to degenerate on single-task
spooky; that is a finding, not a bug.

### B2 SkillGraph-C

A stronger adapted baseline:

- start from Static SkillGraph
- add build-time execution-order or trace-order dependency edges
- do not call this faithful to the paper

Name it "SkillGraph-C" or "SkillGraph-adapted-ordering".

Purpose: avoid beating a strawman. If Hyper-Skill beats this, the result is more
credible.

### B3 Hyper-Skill

Implement a lightweight hyperbolic procedural retriever first. It does not need
full training in the first pass.

Use:

- radius: metric/evidence confidence and support count; low support but matched
  condition should remain reachable, not discarded
- direction: semantic/topic direction from text embedding or deterministic
  keyword sectors for pilot
- condition match: boosts skills whose `condition` matches query/context
- failure match: boosts skills linked to the same failure mode
- conflict check: retrieves opposite or tension skills as warnings when the
  query mentions a risky method

For pilot, it is acceptable to approximate hyperbolic scoring with explicit
features:

```text
score = semantic_score
      + condition_match_bonus
      + failure_mode_bonus
      + low_frequency_preservation_bonus
      + evidence_bonus
      - redundancy_penalty
```

But the script must preserve fields needed for a later real Poincare or Lorentz
implementation.

### B4 Ablations

At minimum:

- Hyper-Skill without condition matching
- Hyper-Skill without conflict/opposite check
- Hyper-Skill without metric/support radius
- Hyper-Skill without evidence bonus

These are necessary to prove which part of the method matters.

## Experiment 1: Low-Frequency Skill Recall

### Research Question

Can the retriever find low-frequency but condition-matched skills that flat
retrieval or static SkillGraph misses?

### Benchmark File

Create:

`paper-skills/eval_skill_memory/benchmarks/rare_skill_bench.jsonl`

Each row:

```json
{
  "id": "rare_001",
  "query": "小数据上 DeBERTa 过拟合怎么办？",
  "context": {
    "task": "spooky-author-identification",
    "data_size": "small",
    "model": "DeBERTa-v3-large",
    "symptoms": ["validation loss rebounds", "overfitting"]
  },
  "gold_skill_ids": ["insight_1", "insight_8"],
  "gold_titles": ["部分解冻优于全参数微调", "全参数微调存在硬天花板"],
  "required_conditions": ["small data", "large transformer", "overfitting"],
  "failure_modes": ["overfitting", "full finetune instability"],
  "evidence": ["insight.md#1", "insight.md#8"],
  "rarity": "low_frequency_or_hard_recall"
}
```

Seed examples from current 15 SOPs:

- #1 partial unfreezing beats full finetuning
- #2 simple linear head beats complex head
- #8 full finetuning hits ceiling
- #9 complex attention head overfits
- #11 feature extraction dead-code bug
- #13 worst node can yield breakthrough

Add graph-derived rare skills:

- train-fold-only scaler/vectorizer to avoid leakage
- OOM-specific reductions
- AMP / DeBERTa attention mask overflow
- gradient checkpointing plus accumulation conflict
- num_workers=0 shared memory failures

### Metrics

- Recall@3, Recall@5, Recall@8
- MRR
- Rare Recall@k for skills with `n_use <= 2`
- Condition Precision: percent of retrieved skills whose condition matches the query
- Evidence Coverage: percent of retrieved skills with resolved evidence refs

### Paper-Grade Requirement

Report mean with 95% bootstrap confidence intervals. Use paired bootstrap or
permutation test when comparing Hyper-Skill against SkillGraph-C.

## Experiment 2: Conflict Identification

### Research Question

Can the system distinguish true conflicts from condition branches and avoid
deleting useful but condition-specific skills?

### Benchmark File

Create:

`paper-skills/eval_skill_memory/benchmarks/conflict_bench.jsonl`

Each row:

```json
{
  "id": "conflict_001",
  "left": "Use DeBERTa-v3-large for best performance",
  "right": "Use a smaller model when large models overfit or cause OOM",
  "context": {
    "data_size": "small",
    "gpu_memory": "tight",
    "symptoms": ["OOM", "overfitting"]
  },
  "label": "condition_branch",
  "preferred_under_context": "right",
  "keep_as_warning": "left",
  "reason": "Large model is useful when resources and regularization are adequate; under tight memory and overfitting, prefer smaller or frozen alternatives."
}
```

Conflict pairs to include:

- large model vs smaller model
- label smoothing vs no label smoothing / focal loss
- full finetuning vs partial unfreezing
- complex attention head vs simple linear head
- more handcrafted features vs avoid noisy/dead-code features
- AMP for speed vs disable AMP for DeBERTa overflow
- gradient checkpointing vs gradient accumulation

Labels:

- `true_conflict`
- `condition_branch`
- `complementary`
- `risk_warning`

### Metrics

- 4-way relation accuracy
- Conditional-branch F1
- Correct Action Rate
- False Deletion Rate
- Grounded Explanation Rate

False Deletion Rate is central: if a system says one side is simply wrong when
the correct label is condition_branch or risk_warning, count it as a serious
error.

## Experiment 3: Coherent Method-Set Retrieval

### Research Question

Does the retriever return an organized method set rather than redundant similar
tips?

### Benchmark File

Create:

`paper-skills/eval_skill_memory/benchmarks/plan_bench.jsonl`

Each row:

```json
{
  "id": "plan_001",
  "query": "小数据文本分类如何降低 log loss？",
  "required_categories": [
    "validation",
    "model_strategy",
    "training_stability",
    "loss_calibration",
    "feature_pipeline",
    "ensemble",
    "risk_warning"
  ],
  "gold_examples": {
    "validation": ["stratified k-fold", "OOF"],
    "model_strategy": ["partial unfreezing", "DeBERTa-v3-large"],
    "training_stability": ["warmup", "early stopping", "gradient clipping"],
    "loss_calibration": ["label smoothing"],
    "feature_pipeline": ["stylometric features must be connected"],
    "ensemble": ["DeBERTa + XGBoost + Logistic Regression"],
    "risk_warning": ["avoid full finetuning ceiling", "avoid complex heads"]
  }
}
```

### Metrics

- Category Coverage@k
- Redundancy Rate@k
- Risk Reminder Recall
- Evidence Density
- Plan Helpfulness, judged blind by LLM and optionally by human review

Blind judge prompt requirement:

- Hide system name.
- Ask whether the returned set is actionable, diverse, condition-aware, and
  evidence-grounded.
- Require JSON output for reproducibility.

## Experiment 4: Agent Behavior

### Research Question

Do retrieval improvements change the agent's search behavior and final metric?

This is the expensive stage. Do not run it before Experiments 1-3 scripts are
working and reviewed.

### Conditions

Minimum final online conditions:

- No-Mem
- Current Flat Memory / experience_kb
- Trace2Skill Linear skill injection
- SkillGraph-C
- Hyper-Skill
- Hyper-Skill without conflict check
- Hyper-Skill without metric/support radius

### Metrics

- best validation log loss
- steps to first threshold, e.g. below 0.20 and below 0.15
- repeated-error rate
- code crash / bug rate
- conflict failure rate
- rare-skill adoption rate
- plan diversity
- token and wall-clock cost

### Seeds and Budget

For pilot:

- 2 seeds per condition
- same step/time/GPU budget

For paper:

- 3 to 5 seeds per condition
- frozen config
- report mean and 95% confidence intervals

## Implementation Tasks for Claude Code

### Phase A: Offline Benchmark Harness

Create directory:

`paper-skills/eval_skill_memory/`

Suggested files:

- `README.md`
- `build_benchmarks.py`
- `retrievers.py`
- `evaluate_rare_recall.py`
- `evaluate_conflicts.py`
- `evaluate_plan_quality.py`
- `schemas.py`
- `outputs/.gitignore`

Do not commit bulky outputs by default.

Expected outputs:

- `benchmarks/rare_skill_bench.jsonl`
- `benchmarks/conflict_bench.jsonl`
- `benchmarks/plan_bench.jsonl`
- `outputs/rare_recall_results.json`
- `outputs/conflict_results.json`
- `outputs/plan_quality_results.json`
- concise markdown summary table

### Phase B: Baselines

Implement:

- Flat BM25 retriever over graph nodes and references.
- Static SkillGraph retriever over `kind` edges, not `type`.
- SkillGraph-C with trace-order dependency edges.
- Hyper-Skill pilot retriever with explainable scoring.
- Required ablations.

Important schema warning: `graph.json` edges use `kind`, not `type`.

### Phase C: Validation

Run offline experiments first:

```bash
python paper-skills/eval_skill_memory/build_benchmarks.py
python paper-skills/eval_skill_memory/evaluate_rare_recall.py
python paper-skills/eval_skill_memory/evaluate_conflicts.py
python paper-skills/eval_skill_memory/evaluate_plan_quality.py
```

If dependencies are missing, document the exact missing package and provide a
fallback pure-Python mode for pilot experiments.

### Phase D: Report

Update `coordination/claude_report.md` with:

- files changed
- commands run
- result tables
- examples where Hyper-Skill wins
- examples where it loses
- blockers
- whether results are pilot-only or clean-main

Do not claim paper-grade numbers until clean split and leakage audit are done.

## Acceptance Criteria for First Claude Pass

The first implementation pass is successful if:

- The three benchmark JSONL files exist and have at least:
  - 12 rare-skill examples
  - 10 conflict examples
  - 5 plan examples
- At least three retrievers run end to end:
  - Flat
  - Static SkillGraph
  - Hyper-Skill pilot
- Evaluation scripts produce JSON and a readable summary table.
- The report includes at least two concrete case studies:
  - one low-frequency recall case
  - one conflict/condition-branch case
- No leaked credentials or bulky logs are added to git.

## Expected Paper Tables

Table 1: Rare Skill Retrieval

Columns:

- System
- Recall@3
- Recall@5
- MRR
- Rare Recall@5
- Condition Precision
- Evidence Coverage

Table 2: Conflict Identification

Columns:

- System
- Accuracy
- Conditional-F1
- Correct Action Rate
- False Deletion Rate
- Grounded Explanation Rate

Table 3: Plan Retrieval Quality

Columns:

- System
- Category Coverage
- Redundancy Rate
- Risk Reminder Recall
- Evidence Density
- Helpfulness

Table 4: Online Agent Behavior

Columns:

- System
- Best log loss
- Steps to threshold
- Bug rate
- Repeated-error rate
- Rare-skill adoption
- Token/time cost

## Risks and Required Notes

- Current static SkillGraph has no prereq edges and is expected to degenerate on
  single-task spooky. Treat this as a documented limitation and use SkillGraph-C
  as a stronger adapted baseline.
- The 15-SOP ground truth may include contaminated evidence. Audit before final
  claims.
- Hyper-Skill pilot scoring is not yet true hyperbolic training. Be explicit:
  "hyperbolic-inspired pilot retriever" until real coordinates are implemented.
- If Hyper-Skill loses to SkillGraph-C on offline metrics, do not hide it. Analyze
  whether condition/failure annotations are too weak or radius scoring is wrong.

## One-Sentence Narrative for the Paper

SkillGraph shows that skill memory needs structure; our experiments test whether
MLE procedural memory needs a stronger structure: one that preserves rare
condition-specific skills, distinguishes apparent conflicts from conditional
branches, and retrieves complete method sets grounded in execution evidence.
