# WP8 Tier-2 formal r8 failure diagnostic

The first formal block, `wp8-tier2-formal-aerial-seed-104729-r1`, is preserved as failed formal evidence and is not eligible for retry or reuse.

The GPU training devpod completed all five online conditions and was deleted with a verified Kubernetes `NotFound` attestation. The isolated CPU evaluator then produced terminal score artifacts for the frozen `no_memory` selection, but Result Fact writeback failed because the trusted terminal Receipts did not close the active ProtocolSpec payload obligations for evaluator, fit scope, and split lineage. Consequently:

- a terminal metric was observed;
- this is not a pre-metric abort;
- no normal Result Fact was published;
- the Session Overlay remained empty;
- the evaluator devpod was subsequently verified `NotFound`, but the old launcher did not emit an evaluator-deletion attestation on its failure branch;
- no stored score value was inspected during recovery analysis.

The preserved r8 block root has 331 files and tree SHA256 `84cb48c4c2c9d359e49125ce40e5d9808bd42e6552478b44e6ec51d15eb6aac4`. The exact source, staging, lifecycle, partial-result, log, ledger, and status hashes are recorded in the adjacent JSON diagnostic.

The correction must keep two meanings separate. `train_view_only` proves that the terminal holdout was isolated from training; it is not evidence that preprocessing was fit only on an internal fold. A corrected Result Fact therefore requires both a verified persisted runtime observation for the selected node and independently verified immutable terminal split/fit/metric Receipts. It must also fail closed if either chain is missing.

Any corrected formal experiment requires a transparent amendment bound to this failure, entirely new immutable roots, and a new staging Stop Gate. r8 will not be overwritten, completed post hoc, or included as a successful block.
