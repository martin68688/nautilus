# Plant Seedlings Classification — component-slot transfer target

Classify each test seedling image into one of the species represented by the
class-named directories under `train/`. Test images are stored under `test/`.
The authoritative output schema and test order come from
`sample_submission.csv`; write `submission/submission.csv` with exactly its
columns and one prediction for every test image. The official task metric is
the competition's mean F-score, so internal validation must report the same
classification objective or an explicitly labeled search proxy.

Use only labeled Plant Seedlings training images for splitting, fitting,
augmentation selection, class balancing, calibration, ensembling, and model
selection. Keep duplicate or near-duplicate image identities in a single fold
whenever such groups are detected. Do not infer labels from directory or file
ordering, and never use test images during fitting or model selection.

All task assets and model weights are local at candidate runtime. Do not
download data, weights, or code during candidate execution. This target is
distinct from the source Leaf task: source scores, images, class mappings,
predictions, checkpoints, feature dimensions, and executable code are not
target evidence. Transferred component cards are hypotheses and must be
validated exclusively on Plant Seedlings training data.
