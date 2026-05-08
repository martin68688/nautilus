# Failures

## ADMET-AI Torch Version Incompatibility
ADMET-AI predictions as domain-coupled features are incompatible with the current environment due to torch version mismatch.

**Root cause**: Dependency version conflict: ADMET-AI requires torch >= 2.8, but current environment uses torch 2.6.

**Evidence**:
- `judge_1` node_13: "Node_8 coder documented 'ADMET-AI: incompatible with torch 2.6 (requires >= 2.8)'."
- `judge_1` node_13: "node_13's primary differentiator — ADMET-AI predictions as domain-coupled features — is explicitly incompatible with the current environment."

**Seen**: 2
**Dreamer runs**: 1

## LightGBM Feature Saturation
Multiple attempts to enrich LightGBM features (OOF stacking, backbone embeddings) failed due to diversity collapse, with LGB OOF remaining stagnant at 0.5991-0.5999 across four consecutive nodes.

**Root cause**: Diminishing returns from feature enrichment and lack of model diversity in LightGBM-only approaches.

**Evidence**:
- `judge_0` node_18: "LGB has been saturated at 0.5991-0.5999 for four consecutive nodes (7, 8, 10, and sibling 11); all enrichment attempts (OOF stacking, backbone embeddings) failed via diversity collapse."
- `judge_1` node_12: "Pure LGB ceiling in this tree is ~0.55-0.60, regardless of features added."

**Seen**: 2
**Dreamer runs**: 1

## MGMB Overfitting from High Feature Ratio
MGMB endpoint (222 samples) suffered overfitting when MACCS+ErG features were added, causing RAE degradation from 0.6761 to 0.7079 due to 27.6:1 feature:sample ratio.

**Root cause**: Extreme dimensionality pressure from high feature-to-sample ratio on sparse endpoint.

**Evidence**:
- `judge_1` node_12: "MGMB overfitting confirmed (RAE 0.6761→0.7079 when MACCS+ErG added, 27.6:1 feature:sample ratio)."
- `judge_1` node_12: "Node_12 addresses the documented MGMB overfitting root cause (27.6:1 feature:sample ratio → confirmed 0.6761→0.7079 RAE degradation when MACCS+ErG added)."

**Seen**: 2
**Dreamer runs**: 1

## Uni-Mol2 API Uncertainty
Uni-Mol2-84M component in complex ensemble has uncertain unimol_tools split='select' API that may require custom implementation.

**Root cause**: Unclear or undocumented API for the Uni-Mol2 library's split parameter.

**Evidence**:
- `judge_0` node_3: "unimol_tools split='select' API is uncertain and may require custom implementation."
- `judge_0` node_3: "Feasibility concerns: (1) unimol_tools split='select' API is uncertain and may require custom implementation."

**Seen**: 2
**Dreamer runs**: 1

## Chemprop API Compatibility Risk
Potential compatibility issues between chemprop 2.x API and CheMeleon weight loading, though mitigated by version pinning.

**Root cause**: Version mismatch risk between chemprop library and pretrained model weights.

**Evidence**:
- `judge_0` node_2: "Main risk is chemprop 2.x API compatibility with CheMeleon weight loading, which is mitigated by pinning chemprop>=2.2.0."

**Seen**: 1
**Dreamer runs**: 1

## MiniMol Installation Uncertainty
MiniMol installation is untested in the current environment, creating feasibility risk for node 12's learned embedding component.

**Root cause**: Lack of prior testing or compatibility verification for MiniMol library.

**Evidence**:
- `judge_1` node_12: "MiniMol install is untested in this environment."
- `judge_1` node_12: "MiniMol: Untested in this environment, no evidence of compatibility issues but also no confirmation."

**Seen**: 2
**Dreamer runs**: 1

## Node 10 Checkpoint Accessibility
Uncertainty about checkpoint accessibility for seeds 42/123 in node 10 due to missing feedback.txt and 'salvaged' status.

**Root cause**: Missing documentation and uncertain artifact persistence from previous training run.

**Evidence**:
- `judge_0` node_18: "Risk: node_10 is 'salvaged' with no feedback.txt — checkpoint accessibility for seeds 42/123 is uncertain."

**Seen**: 1
**Dreamer runs**: 1

## Node_13 design centered on broken component
Node_13's design is architecturally centered on ADMET-AI, which is confirmed incompatible. The Spearman audit, leakage checks, and expected improvement estimate all depend on ADMET-AI succeeding, leaving no viable fallback path.

**Root cause**: Poor contingency planning: the design did not include a fallback strategy for the primary component's failure, unlike node_12 which listed ADMET-AI as a fallback for MiniMol.

**Evidence**:
- `judge_1` node_node_13: "No viable fallback is described in the design (unlike node_12 which listed ADMET-AI as fallback for MiniMol)."
- `judge_1` node_node_13: "Feasibility is low because the design is architecturally centered on ADMET-AI: the Spearman audit, the leakage checks, and the expected improvement estimate all depend on ADMET-AI succeeding, leaving the coder with no path to differentiate from a plain feature-selection run."

**Seen**: 2
**Dreamer runs**: 1

## Per-endpoint models miss multi-task synergies
Node_1 uses separate LightGBM models per endpoint, which forgoes multi-task sharing. This is a disadvantage for sparse endpoints like MGMB (222 samples) that could benefit from gradients on correlated endpoints.

**Root cause**: Architectural limitation: per-endpoint models cannot leverage shared representations across tasks, limiting performance on low-data endpoints.

**Evidence**:
- `judge_0` node_node_1: "per-endpoint models forgo multi-task sharing, which is a disadvantage for sparse endpoints like MGMB (222 samples)."
- `judge_0` node_node_1: "Node 1 (SIMPLE LightGBM): Moderate expected improvement (score 3) — rich features help tree models but per-endpoint design misses multi-task synergies for sparse endpoints."

**Seen**: 2
**Dreamer runs**: 1

## Ensemble time budget tightness
Node_3's estimated training time (215-285 min) is optimistic and could be exceeded if any component is slow or encounters errors, risking the 480 min budget. Sequential training dependencies increase wall time risk.

**Root cause**: Orchestration complexity: the heterogeneous ensemble requires sequential training of multiple components, each with potential delays, making the total time uncertain.

**Evidence**:
- `judge_0` node_node_3: "optimistic 215-285 min estimate could be exceeded if any component is slow or encounters errors on 480 min budget"
- `judge_0` node_node_3: "Feasibility is moderate rather than high due to orchestration complexity and time-budget tightness."

**Seen**: 2
**Dreamer runs**: 1

## Uni-Mol2 84M GPU VRAM requirement
Uni-Mol2 84M model requires >=8GB VRAM, which may not be available on all GPUs, posing a feasibility risk for node_3.

**Root cause**: Hardware constraint: the model size (84M parameters) demands significant GPU memory, which may exceed available resources.

**Evidence**:
- `judge_0` node_node_3: "Uni-Mol2 84M requires >=8GB VRAM"
- `judge_0` node_node_3: "ensure GPU has >=8GB VRAM and verify unimol_tools split='select' API before committing."

**Seen**: 2
**Dreamer runs**: 1

## Meta-Insights

### Dependency version conflicts are a recurring failure mode, particularly for deep learning libraries (torch, chemprop, unimol_tools) where API changes between minor versions break compatibility with pretrained models or downstream tools.
**Supporting**: ADMET-AI Torch Version Incompatibility, Chemprop API Compatibility Risk, Uni-Mol2 API Uncertainty

### Sparse endpoints (e.g., MGMB with 222 samples) are highly vulnerable to overfitting from feature proliferation and benefit from multi-task learning, but single-model-per-endpoint architectures fail to exploit this synergy.
**Supporting**: MGMB Overfitting from High Feature Ratio, Per-endpoint models miss multi-task synergies

### Architectural designs that lack fallback strategies for their primary differentiating component are high-risk; successful designs explicitly plan for component failure with alternative paths.
**Supporting**: Node_13 design centered on broken component, MiniMol Installation Uncertainty

### LightGBM-only approaches hit a hard performance ceiling (~0.60) regardless of feature engineering, indicating that model diversity (not just feature diversity) is required to break through saturation.
**Supporting**: LightGBM Feature Saturation

### Complex heterogeneous ensembles introduce orchestration risks (time budget, GPU memory, untested libraries) that compound, making feasibility assessments highly sensitive to the weakest link in the pipeline.
**Supporting**: Ensemble time budget tightness, Uni-Mol2 84M GPU VRAM requirement, Node 10 Checkpoint Accessibility

**Seen**: 1
**Dreamer runs**: 1

