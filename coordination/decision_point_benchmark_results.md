# Independent Decision-Point Retrieval Benchmark

Queries: `29`. These are independent decision descriptions, not historical child-recovery queries.

| Method | Available | graded nDCG@10 | Adoption AP@10 | blocked RunNode@10 | unsupported SOP@10 | non-admissible@10 |
|---|---|---:|---:|---:|---:|---:|
| random_unfiltered | yes | 0.0349 | 0.0201 | 0.2241 | 0.3517 | 0.5759 |
| bm25_unfiltered | yes | 0.2333 | 0.1851 | 0.2828 | 0.2000 | 0.4828 |
| bm25_safety_filtered | yes | 0.3200 | 0.2723 | 0.0000 | 0.0000 | 0.0000 |
| tfidf_unfiltered | yes | 0.3050 | 0.2487 | 0.1897 | 0.2000 | 0.3897 |
| tfidf_safety_filtered | yes | 0.3454 | 0.2886 | 0.0000 | 0.0000 | 0.0000 |
| lsa_dense_unfiltered | yes | 0.3145 | 0.2309 | 0.2586 | 0.2000 | 0.4586 |
| lsa_dense_safety_filtered | yes | 0.3995 | 0.3273 | 0.0000 | 0.0000 | 0.0000 |
| minilm_dense_unfiltered | yes | 0.2912 | 0.2126 | 0.1690 | 0.2241 | 0.3931 |
| minilm_dense_safety_filtered | yes | 0.4347 | 0.3476 | 0.0000 | 0.0000 | 0.0000 |
| tree_only_mapped_no_task | yes | 0.2253 | 0.1330 | 0.0000 | 0.0793 | 0.0793 |
| tree_only_mapped_task_aware | yes | 0.2925 | 0.1862 | 0.0000 | 0.0448 | 0.0448 |
| legacy_stage_gateway | yes | 0.3543 | 0.2905 | 0.0000 | 0.0000 | 0.0000 |
| stage_hybrid_sop | yes | 0.4522 | 0.3861 | 0.0000 | 0.0000 | 0.0000 |
| oracle_upper | yes | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Claim Gates

- `independent_decision_points`: `True`
- `historical_child_as_gold`: `False`
- `six_task_families`: `True`
- `minimum_25_strict_queries`: `True`
- `unique_gold_sets_globally`: `True`
- `task_compatible_gold_only`: `True`
- `task_matched_blocked_distractors`: `True`
- `like_for_like_text_scorer_pairs`: `True`
- `convenience_seed_selection_disclosed`: `True`
- `two_blind_annotators_and_adjudication`: `False`
- `offline_retrieval_claim_allowed`: `False`
- `online_downstream_claim_allowed`: `False`
- `reason`: `Current labels are silver expert seeds; blind human adjudication and concurrent online training remain outstanding.`

## Paired Comparisons

- `bm25_gate_effect`: delta=0.0867, bootstrap 95% CI [0.0398, 0.1411], sign-flip Holm p=0.0016.
- `tfidf_gate_effect`: delta=0.0404, bootstrap 95% CI [0.0240, 0.0587], sign-flip Holm p=0.0016.
- `lsa_gate_effect`: delta=0.0850, bootstrap 95% CI [0.0513, 0.1246], sign-flip Holm p=0.0011.
- `minilm_gate_effect`: delta=0.1435, bootstrap 95% CI [0.0891, 0.2019], sign-flip Holm p=0.0011.
- `task_identity_effect_on_tree`: delta=0.0672, bootstrap 95% CI [0.0351, 0.1053], sign-flip Holm p=0.0016.
- `true_stage_hybrid_vs_legacy_gateway`: delta=0.0980, bootstrap 95% CI [-0.0093, 0.2056], sign-flip Holm p=0.264.
- `stage_hybrid_vs_tree_no_task`: delta=0.2269, bootstrap 95% CI [0.1179, 0.3337], sign-flip Holm p=0.0011.
- `stage_hybrid_vs_bm25_safety_filtered`: delta=0.1323, bootstrap 95% CI [0.0481, 0.2258], sign-flip Holm p=0.0295.
- `stage_hybrid_vs_tfidf_safety_filtered`: delta=0.1069, bootstrap 95% CI [0.0158, 0.2083], sign-flip Holm p=0.1592.
- `stage_hybrid_vs_lsa_safety_filtered`: delta=0.0527, bootstrap 95% CI [-0.0604, 0.1613], sign-flip Holm p=0.7605.
- `stage_hybrid_vs_minilm_safety_filtered`: delta=0.0175, bootstrap 95% CI [-0.0941, 0.1248], sign-flip Holm p=0.7605.

## Interpretation

This run is a silver-label diagnostic over 29 of 60 deterministic convenience seeds. The retained task-family distribution is uneven, so per-family and per-stage values are descriptive only. It cannot support a paper claim until two blind annotators adjudicate every decision point. Online downstream improvement is not measured here.
