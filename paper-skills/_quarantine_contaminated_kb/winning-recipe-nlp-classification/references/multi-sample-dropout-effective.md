# Multi-Sample Dropout (K=4) Effective for Classification

## Finding
MSD with K=4 independent dropout passes on [CLS] embedding, averaged logits, provides implicit ensemble regularization at negligible cost.

## Evidence
- Run 091845 top4: DebertaMSD(K=4) → val=0.2517
- Same run without MSD: val=0.2653
- Improvement: ~0.014 logloss from MSD alone

## Mechanism
- Apply dropout K times to same [CLS] embedding → K different "views"
- Each view passes through same linear classifier → K sets of logits
- Averaging K logits smooths predictions, reduces overconfidence
- Equivalent to ensemble of K sub-networks sharing all weights except dropout masks
- No additional parameters; only K forward passes through single Linear layer

## Condition
Classification tasks with dropout-based models. K=4 for 3-class; scan K=2~8 for other tasks.
