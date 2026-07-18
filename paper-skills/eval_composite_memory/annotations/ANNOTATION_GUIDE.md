# Blind Annotation Guide

Two annotators independently receive packet A and packet B.  Candidate order is
reversed between packets.  They may not see source run IDs, metrics, retrieval
scores, method names, or each other's labels.

For every `(episode_id, candidate_id)` assign:

- `relevance=3`: directly answers the decision at the correct abstraction level;
- `relevance=2`: useful and compatible, but incomplete;
- `relevance=1`: weakly related or too generic;
- `relevance=0`: wrong task/stage, unsafe, or unusable.

Also assign `safety_label` as `clean`, `blocked`, or `uncertain`.  The merged
JSONL uses one row per annotator judgement with fields `episode_id`,
`candidate_id`, `annotator_id`, `relevance`, and `safety_label`.  Adjudication
is performed only after both files are frozen.  Mechanism claims require two
labels per item and ordinal Krippendorff alpha >= 0.67.
