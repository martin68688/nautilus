# APTOS 2019 Blindness Detection — cross-task transfer target

Predict diabetic-retinopathy severity from retinal fundus photographs. The
training table is `train.csv`; `id_code` identifies an image in
`train_images/`, and `diagnosis` is an ordered integer label from 0 through 4.
The test table and `test_images/` have no labels. The official task metric is
quadratic weighted Cohen's kappa (higher is better).

Use only labeled training rows for splitting, fitting, augmentation choices,
threshold selection, calibration, ensembling, and model selection. Preserve
patient/image identity across every split. Derive the output schema and test
row order from `sample_submission.csv`, and write `submission/submission.csv`
with exactly the columns `id_code,diagnosis`. Final predictions must be valid
severity grades accepted by the competition format.

All task assets are local at runtime. Do not download data, model weights, or
code during candidate execution. This target is distinct from the source Leaf
task: source scores, images, class mappings, predictions, checkpoints, feature
dimensions, and code are not target evidence. Transferred memories are
hypotheses and must be validated exclusively on APTOS training data.

