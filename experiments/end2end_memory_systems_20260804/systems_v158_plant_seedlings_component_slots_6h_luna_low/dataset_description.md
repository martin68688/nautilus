# Plant Seedlings Classification — cross-task transfer target

Classify each test photograph into one of the plant species represented by the
official training folders. Training labels are encoded only by the immediate
subdirectory names under `train/`; files under `test/` are unlabeled. The
official task metric is mean F1 score (higher is better).

Use only labeled training images for splitting, fitting, augmentation choices,
calibration, ensembling, and model selection. Keep near-duplicate views of the
same physical plant in one split whenever such groups can be inferred from the
target data. Derive the exact output columns, class spellings, and test row
order from `sample_submission.csv`, then write `submission/submission.csv`
with the same schema and one valid species label per test image.

All task assets are local at runtime. Do not download data, pretrained weights,
or code during candidate execution. This target is distinct from the source
Leaf task: source images, labels, scores, predictions, checkpoints, feature
dimensions, weights, and code are not target evidence. Transferred memories are
hypotheses and must be validated exclusively on Plant Seedlings training data.
