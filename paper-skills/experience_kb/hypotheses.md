# Hypotheses

## [CONFIRMED] Pure LightGBM with rich molecular features (~6400 features) provides a strong metric floor and moderate improvement over tutorial baseline, but has a performance ceiling of ~0.55-0.60 MA-RAE.


**Seen**: 3
**Dreamer runs**: 1

## [CONFIRMED] A pretrained CheMeleon MPNN with multi-task NaN-masked loss provides high expected improvement due to multi-task sharing, especially for sparse endpoints like MGMB.


**Seen**: 1
**Dreamer runs**: 1

## [CONFIRMED] A heterogeneous NN+tree ensemble (Uni-Mol2-84M + CheMeleon MPNN + LightGBM) achieves the highest performance ceiling, matching top competition solutions.


**Seen**: 1
**Dreamer runs**: 1

## [CONFIRMED] MGMB overfitting in node_1 was caused by a high feature-to-sample ratio (27.6:1).
**Proposed in**: `judge_1`
**Verdict in**: `judge_1`

**Evidence**:
- `judge_1`: "Parent node_1: MGMB overfitting confirmed (RAE 0.6761→0.7079 when MACCS+ErG added, 27.6:1 feature:sample ratio)."

**Seen**: 1
**Dreamer runs**: 1

## [CONFIRMED] ADMET-AI is incompatible with the current environment (torch 2.6).
**Proposed in**: `judge_1`
**Verdict in**: `judge_1`

**Evidence**:
- `judge_1`: "ADMET-AI incompatibility: node_8 coder explicitly documented "ADMET-AI: incompatible with torch 2.6 (requires >= 2.8)". This is fatal for node_13's primary strategy."

**Seen**: 1
**Dreamer runs**: 1

## [UNTESTED] Per-endpoint feature selection (K=min(500, sqrt(n)*10)) directly addresses MGMB overfitting and provides a novel improvement over pure LGB baselines.


**Seen**: 1
**Dreamer runs**: 1

## [UNTESTED] Adding a third MPNN seed (seed_456) to node_10 reduces fold variance and improves OOF performance, following the law of diminishing returns (√n scaling).


**Seen**: 1
**Dreamer runs**: 1

## [UNTESTED] Adding Avalon fingerprints (1024-bit, path-based) to the LGB feature matrix shifts LGB OOF performance because they are structurally orthogonal to ECFP4.


**Seen**: 1
**Dreamer runs**: 1

## [UNTESTED] Adding a RandomForest third branch (ECFP4+Avalon, 3072 features, bagging) to the existing MPNN+LGB paradigm plus 3rd MPNN seed improves performance.
**Proposed in**: `judge_2`
**Verdict in**: `None`

**Evidence**:
- `judge_2`: "Node_19 adds a RandomForest third branch (ECFP4+Avalon, 3072 features, bagging) to the existing MPNN+LGB paradigm plus 3rd MPNN seed, while reusing node_10's LGB OOF directly (no LGB retraining)."

**Seen**: 2
**Dreamer runs**: 1

## [UNTESTED] MiniMol adds a learned embedding that could generalize better on sparse endpoints.
**Proposed in**: `judge_1`
**Verdict in**: `None`

**Evidence**:
- `judge_1`: "MiniMol adds a learned embedding that could generalize better on sparse endpoints."

**Seen**: 1
**Dreamer runs**: 1

## [CONFIRMED] Node_10 is 'salvaged' with no feedback.txt — checkpoint accessibility for seeds 42/123 is uncertain.
**Proposed in**: `judge_2`
**Verdict in**: `judge_2`

**Evidence**:
- `judge_2`: "Risk: node_10 is 'salvaged' with no feedback.txt — checkpoint accessibility for seeds 42/123 is uncertain."

**Seen**: 1
**Dreamer runs**: 1

## Meta-Insights

### Pure tree-based models (LightGBM) hit a hard performance ceiling (~0.55-0.60 MA-RAE) regardless of feature engineering, and all attempts to enrich them (OOF stacking, backbone embeddings) fail due to diversity collapse. This confirms the necessity of neural network integration for breakthrough performance.
**Supporting**: [CONFIRMED] Pure LightGBM with rich molecular features (~6400 features) provides a strong metric floor and moderate improvement over tutorial baseline, but has a performance ceiling of ~0.55-0.60 MA-RAE., [CONFIRMED] MGMB overfitting in node_1 was caused by a high feature-to-sample ratio (27.6:1).

### The most promising untested strategies involve ensemble diversification: adding a third diverse model (RandomForest/XGBoost) for 3-way blending, adding orthogonal fingerprint types (Avalon), and increasing MPNN seed count for variance reduction. These follow the established pattern of top competition solutions.
**Supporting**: [UNTESTED] Adding a RandomForest third branch (ECFP4+Avalon, 3072 features, bagging) to the existing MPNN+LGB paradigm plus 3rd MPNN seed improves performance., [UNTESTED] Adding Avalon fingerprints (1024-bit, path-based) to the LGB feature matrix shifts LGB OOF performance because they are structurally orthogonal to ECFP4., [UNTESTED] Adding a third MPNN seed (seed_456) to node_10 reduces fold variance and improves OOF performance, following the law of diminishing returns (√n scaling)., [CONFIRMED] A heterogeneous NN+tree ensemble (Uni-Mol2-84M + CheMeleon MPNN + LightGBM) achieves the highest performance ceiling, matching top competition solutions.

### Sparse endpoints (like MGMB with only 222 samples) are the primary bottleneck: they suffer from overfitting in high-dimensional feature spaces and benefit most from multi-task learning with correlated endpoints. Targeted solutions include per-endpoint feature selection and learned embeddings from pretrained models.
**Supporting**: [CONFIRMED] MGMB overfitting in node_1 was caused by a high feature-to-sample ratio (27.6:1)., [UNTESTED] Per-endpoint feature selection (K=min(500, sqrt(n)*10)) directly addresses MGMB overfitting and provides a novel improvement over pure LGB baselines., [UNTESTED] MiniMol adds a learned embedding that could generalize better on sparse endpoints., [CONFIRMED] A pretrained CheMeleon MPNN with multi-task NaN-masked loss provides high expected improvement due to multi-task sharing, especially for sparse endpoints like MGMB.

**Seen**: 1
**Dreamer runs**: 1

