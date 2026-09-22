# Leaf-to-Plant tournament source-data notes

## Inclusion rule

The target task is Plant Seedlings Classification. The transfer source is the
frozen Leaf memory snapshot used by v167. Transfer observations are restricted
to official submissions produced after the four-seat tournament design was
activated. This gives attempts 001--005. All five now have complete Kaggle
official score ledgers. Earlier non-tournament Plant transfer runs are not used.

All seven independent no-memory Novel searches with usable official scores are
shown. They are not treated as exchangeable replicates: v159 allowed public
pretrained assets and network access, v160 and the early v163 runs used other
resource contracts, and only v163 seeds 42--44 form a nominally aligned
three-run set. The heterogeneous runs are retained because the requested figure
uses the full observed Novel range rather than a selected favorable subset.

## Primary and diagnostic transfer records

- Attempt 001 stopped partial because an explicit zero-second finalization
  reserve was replaced by the 900-second default. Its certified solution has
  official score 0.97355 (Kaggle ref 55824133).
- Attempt 002 reached official score 0.98236 but was censored during baseline
  qualification and is not paper-grade. It is the best official score among
  the tournament-era submissions.
- Attempt 003 reached internal macro-F1 0.975660 but stopped in a donor-queue
  deadlock. Its official score is 0.97481 (Kaggle ref 55824142).
- Attempt 004 reached official score 0.97607 after a full 21,600-s search, but
  terminal finalization failed and no champion was frozen. It is diagnostic.
- Attempt 005 completed the 21,600-s budget, exited successfully, froze the
  terminal ranking and champion, and reached official score 0.97481. It is the
  only transfer point used for the main paper claim.

The final run used one NVIDIA A100 80 GB GPU, 8 CPU cores, 32 GiB memory, one
search worker, GPT-5.6-sol with low reasoning effort, and a 21,600-s search
budget. Its memory identity registers 96 historical source runs in the global
snapshot; the transfer portfolio itself is projected from Leaf evidence. Public
pretrained model assets were permitted by the target execution contract.

## Summary values

The seven no-memory Novel official scores range from 0.76574 to 0.98362, with a
median of 0.93450. The five tournament official scores range from 0.97355 to
0.98236. Attempt 002 is the best official tournament submission at 0.98236,
exceeding six of the seven observed Novel runs but not the unrestricted v159
result (0.98362). Attempt 005 is separately identified as the only
terminal-complete run; its official score is 0.97481. The comparison is
descriptive because model access, pretrained-asset access, compute, CPU
allocation, and framework version differ across historical runs.

The final tournament was still in `round1_equal_improve` at the budget cutoff.
The terminal ranking was frozen at that point; it does not imply that the
planned Top-2 and champion-deepening phases had run to completion. Each plotted
point is a single search run or a single seat incumbent, so no confidence
interval or hypothesis test is reported.
