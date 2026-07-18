# Decision-Point Component Ablation

Queries: `29`. All gated methods use the same clean-evidence predicate and the same silver gold.

| Method | nDCG@10 | AP@10 | Unsupported SOP@10 |
|---|---:|---:|---:|
| semantic_only_unfiltered | 0.2843 | 0.2417 | 0.4345 |
| semantic_only_gate | 0.3256 | 0.2727 | 0.0000 |
| field_aware_no_stage_unfiltered | 0.3075 | 0.2547 | 0.4103 |
| field_aware_no_stage_gate | 0.3543 | 0.2905 | 0.0000 |
| field_aware_stage_unfiltered | 0.3075 | 0.2547 | 0.4103 |
| field_aware_stage_gate | 0.3543 | 0.2905 | 0.0000 |
| legacy_stage_gateway | 0.3543 | 0.2905 | 0.0000 |
| production_stage_hybrid_sop | 0.4522 | 0.3861 | 0.0000 |
| minilm_gate | 0.4347 | 0.3476 | 0.0000 |
| minilm_stage_hard_gate | 0.4725 | 0.3822 | 0.0000 |
| minilm_stage_plus_tree_lexical_rrf | 0.4146 | 0.3416 | 0.0000 |
| minilm_stage_plus_tree_stage_rrf | 0.3856 | 0.3077 | 0.0000 |
| minilm_stage_plus_tree_geometry_rrf | 0.4616 | 0.3822 | 0.0000 |
| minilm_stage_plus_tree_full_rrf | 0.4890 | 0.4089 | 0.0000 |

## Paired Component Effects

- `clean_gateway_gate_effect`: delta=+0.0468, CI [+0.0127, +0.0892], Holm p=0.0039; changed=29/29, improved/degraded=13/0.
- `extra_fields_effect`: delta=+0.0287, CI [-0.0054, +0.0711], Holm p=1; changed=29/29, improved/degraded=10/5.
- `debug_stage_boost_effect`: delta=+0.0000, CI [+0.0000, +0.0000], Holm p=1; changed=0/29, improved/degraded=0/0.
- `legacy_path_equivalence`: delta=+0.0000, CI [+0.0000, +0.0000], Holm p=1; changed=0/29, improved/degraded=0/0.
- `hard_stage_filter_effect_on_minilm`: delta=+0.0378, CI [-0.0337, +0.0977], Holm p=1; changed=29/29, improved/degraded=16/3.
- `add_tree_lexical_projection`: delta=-0.0580, CI [-0.1410, +0.0166], Holm p=1; changed=29/29, improved/degraded=9/11.
- `add_tree_stage_projection`: delta=-0.0290, CI [-0.0662, +0.0076], Holm p=1; changed=29/29, improved/degraded=5/9.
- `add_tree_geometry_projection`: delta=+0.0760, CI [+0.0297, +0.1240], Holm p=0.0484; changed=29/29, improved/degraded=16/3.
- `add_tree_task_identity_projection`: delta=+0.0274, CI [+0.0075, +0.0568], Holm p=0.0039; changed=22/29, improved/degraded=12/0.
- `production_vs_legacy_gateway`: delta=+0.0980, CI [-0.0093, +0.2056], Holm p=0.8799; changed=29/29, improved/degraded=20/7.
- `production_vs_minilm_gate`: delta=+0.0175, CI [-0.0941, +0.1248], Holm p=1; changed=29/29, improved/degraded=17/10.
- `projected_full_vs_production`: delta=+0.0368, CI [-0.0174, +0.1028], Holm p=1; changed=29/29, improved/degraded=12/11.
- `projected_full_vs_minilm_gate`: delta=+0.0543, CI [-0.0521, +0.1564], Holm p=1; changed=29/29, improved/degraded=16/9.

## Evidence-Based Conclusions

1. The deterministic clean-evidence gateway is the only component in the current production-path ablation that remains significant after Holm correction: +0.0468 nDCG@10, with no degraded query.
2. Conditions/failures/evidence fields have a positive point estimate (+0.0287), but the interval crosses zero and the corrected result is not significant.
3. The legacy debug-stage boost changes 0/29 Top-10 rankings. It is inert on this benchmark, not evidence of useful stage awareness.
4. Hard stage filtering improves MiniLM by +0.0378 in point estimate, led by Draft, but is not significant with 29 silver queries.
5. Adding projected Tree lexical and stage channels hurts the point estimate, especially on Improve. Geometry recovers +0.0760 relative to that degraded intermediate, and task identity adds +0.0274, but the full projection is not significantly better than MiniLM after correction.
6. The production Stage Hybrid row is the only row that invokes the shared production channel implementation; its comparison against legacy and MiniLM must be read from the paired table without promoting a silver-label diagnostic to a paper claim.

## Interpretation Guard

The method `legacy_stage_gateway` preserves the old field-aware lexical ranking plus deterministic clean-evidence filter. `production_stage_hybrid_sop` invokes the shared production v2 SOP and Tree channels, stage taxonomy, task identity, geometry, weighted RRF, and final clean gate.

The Tree/RRF rows are SOP-space projections built only for this ablation. They are not the production `_hybrid_pack`, which fuses execution-node IDs. The labels are single-annotator silver labels from 29 convenience-sampled decision points, so these results are diagnostic and appendix-grade, not a paper-level superiority claim.
