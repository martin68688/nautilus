# Wins

## Butina scaffold splits for honest OOF estimation
Using Butina clustering to create scaffold-based cross-validation splits, which provides a more realistic out-of-fold estimate than random splits

**Condition**: When evaluating molecular property prediction models where scaffold generalization is important

**Evidence**:
- `judge_0` node_1: "Butina CV gives a more honest OOF estimate"

**Seen**: 1
**Dreamer runs**: 1

## NaN-masked multi-task MAE loss for MPNN
Using multi-task learning with NaN masking to handle missing values, allowing sparse endpoints to benefit from gradients on correlated endpoints via shared backbone

**Condition**: When training on datasets with multiple correlated endpoints and missing values, particularly with sparse endpoints like MGMB (222 samples)

**Evidence**:
- `judge_0` node_2: "Multi-task sharing is a key advantage: MGMB (222 samples) benefits from gradients on correlated endpoints (LogD, KSOL) via the shared backbone"

**Seen**: 1
**Dreamer runs**: 1

## CheMeleon pretrained MPNN backbone
Using CheMeleon pretrained MPNN weights as initialization for fine-tuning, leveraging the most-used backbone in the competition (11+ teams)

**Condition**: When graph topology encoding is needed for generalization to longer test SMILES (20% longer than train)

**Evidence**:
- `judge_0` node_2: "Node 2 uses CheMeleon pretrained MPNN (most-used backbone in the competition: 11+ teams)"

**Seen**: 1
**Dreamer runs**: 1

## Heterogeneous NN+tree ensemble architecture
Combining Uni-Mol2-84M (3D geometry), CheMeleon MPNN (graph topology), and LightGBM+embeddings (bounded predictions) in an ensemble

**Condition**: When pursuing maximum performance ceiling, matching the pattern of top-10 competition solutions

**Evidence**:
- `judge_0` node_3: "Node 3 combines Uni-Mol2-84M (3D geometry), CheMeleon MPNN (graph topology), and LightGBM+embeddings (bounded predictions) — the heterogeneous NN+tree ensemble architecture used by all top-10 competition solutions"

**Seen**: 1
**Dreamer runs**: 1

## Per-endpoint feature selection for overfitting mitigation
Applying feature selection per endpoint with K=min(500, sqrt(n)*10) to reduce dimensionality pressure and address overfitting

**Condition**: When dealing with high feature-to-sample ratios (27.6:1 for MGMB) causing overfitting

**Evidence**:
- `judge_1` node_12: "Per-endpoint feature selection (K=min(500, sqrt(n)*10)) directly reduces dimensionality pressure per endpoint, a novel technique not tried elsewhere in the tree"

**Seen**: 1
**Dreamer runs**: 1

## Multiple MPNN seeds for variance reduction
Adding additional MPNN training seeds to reduce fold variance, with diminishing returns following √n scaling

**Condition**: When MPNN models show high variance between seeds, as evidenced by previous improvements from 1→2 seeds

**Evidence**:
- `judge_2` node_18: "The 1→2 seed transition in node_10 improved MPNN OOF from 0.5007 to 0.4880 (0.013 gain). Diminishing returns law (√n scaling) predicts the 2→3 seed transition yields ~0.006-0.008 in MPNN OOF"

**Seen**: 1
**Dreamer runs**: 1

## Orthogonal fingerprint combination
Adding Avalon fingerprints (path-based) to complement ECFP4 (radius-based) fingerprints for structural orthogonality

**Condition**: When LightGBM feature space needs diversification beyond ECFP4 fingerprints

**Evidence**:
- `judge_2` node_18: "Avalon (rdkit.Avalon.pyAvalonTools, path-based, 1024-bit) is structurally orthogonal to ECFP4 (radius-based) and was explicitly recommended in node_8 feedback as an untested cheap addition"

**Seen**: 1
**Dreamer runs**: 1

## Three-way model blending with RandomForest
Adding RandomForest as a third diverse model branch (ECFP4+Avalon features) to existing MPNN+LGB paradigm for improved ensemble diversity

**Condition**: When seeking to improve blending performance beyond two-model ensembles, as explicitly recommended in previous feedback

**Evidence**:
- `judge_2` node_19: "Node_8 feedback explicitly recommended 'adding a third diverse model (XGBoost/RF) for 3-way blending' as the primary next step"

**Seen**: 1
**Dreamer runs**: 1

## Meta-Insights

### Ensemble diversity is the primary driver of performance ceiling — combining structurally orthogonal models (3D geometry, graph topology, tree-based) consistently outperforms any single model type, as evidenced by the heterogeneous NN+tree architecture and the three-way blending recommendation.
**Supporting**: Heterogeneous NN+tree ensemble architecture, Three-way model blending with RandomForest, Orthogonal fingerprint combination

### Overfitting mitigation strategies must be tailored to data sparsity — per-endpoint feature selection and multi-task learning with NaN masking both address the high feature-to-sample ratio problem, but from complementary angles (dimensionality reduction vs. gradient sharing).
**Supporting**: Per-endpoint feature selection for overfitting mitigation, NaN-masked multi-task MAE loss for MPNN

### Validation strategy must match the generalization challenge — scaffold-based splits (Butina) provide honest OOF estimates when test molecules are structurally novel, and pretrained backbones (CheMeleon) help bridge the distribution shift to longer SMILES.
**Supporting**: Butina scaffold splits for honest OOF estimation, CheMeleon pretrained MPNN backbone

### Diminishing returns follow predictable scaling laws — adding more seeds or more model branches yields decreasing marginal gains (√n scaling for seeds, saturation for ensemble size), suggesting optimal resource allocation requires cost-benefit analysis.
**Supporting**: Multiple MPNN seeds for variance reduction, Three-way model blending with RandomForest

**Seen**: 1
**Dreamer runs**: 1

