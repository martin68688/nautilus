"""Frozen source/variant catalog for the v123 Leaf Replay Research Top-10.

The score-frontier Top-5 and diverse-frontier Top-5 contain three duplicate
sources.  ``TOP10_SLOTS`` preserves the ten ranking slots, while ``CANDIDATES``
contains the seven byte-unique programs that must actually be reproduced.
"""

from __future__ import annotations

from typing import Any


V21_ROOT = "/workspace/experiment-end2end-memory-agent-v21/runs"
V22_ROOT = "/workspace/experiment-end2end-memory-agent-v22/runs"

RUNFOREST_JOURNAL = (
    f"{V21_ROOT}/e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "runforest_only__seed-1/attempt-001/agent/logs/20260807_055845_"
    "e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "runforest_only__seed-1/journal.json"
)
MACLA_JOURNAL = (
    f"{V21_ROOT}/e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "macla_style_port__seed-1/attempt-001/agent/logs/20260807_055819_"
    "e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "macla_style_port__seed-1/journal.json"
)
DYNAMIC_JOURNAL = (
    f"{V22_ROOT}/e2e-pilot-agentic-three-role-v22__leaf-classification__"
    "dynamic_hybrid__seed-1/attempt-000/agent/logs/20260806_185700_"
    "e2e-pilot-agentic-three-role-v22__leaf-classification__"
    "dynamic_hybrid__seed-1/journal.json"
)
GOME_JOURNAL = (
    f"{V21_ROOT}/e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "gome_style_port__seed-1/attempt-000/agent/logs/20260806_220452_"
    "e2e-pilot-agentic-three-role-v21__leaf-classification__"
    "gome_style_port__seed-1/journal.json"
)


TOP10_SLOTS: tuple[str, ...] = (
    "leaf-official-runforest-efficientnet-b3-00168",
    "leaf-official-macla-multideep-00599",
    "leaf-official-dynamic-multibackbone-01008",
    "leaf-official-gome-crossattention-01176",
    "leaf-internal-dynamic-multibackbone-013847",
    "leaf-official-runforest-efficientnet-b3-00168",
    "leaf-official-macla-multideep-00599",
    "leaf-official-dynamic-multibackbone-01008",
    "leaf-internal-macla-lightgbm-034733",
    "leaf-internal-runforest-vit-attention-047483",
)


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "leaf-official-runforest-efficientnet-b3-00168",
        "system_id": "runforest_only",
        "historical_metric": 0.00168,
        "historical_metric_note": "previous official score; rerun from locked source",
        "journal": RUNFOREST_JOURNAL,
        "node_id": "4d2afeda635a44b4adb1c7c97d7a45d6",
        "expected_code_sha256": (
            "572cd25bf41b9208adaa59607cb5203a6d7ac5b2650838d2afbb74dc38a03ec3"
        ),
        "array_variants": {"historical_double_temperature": "test_probs"},
        "official_submission_variant": "historical_double_temperature",
        "memory_disposition_before_official_score": "verified_clean",
    },
    {
        "candidate_id": "leaf-official-macla-multideep-00599",
        "system_id": "macla_style_port",
        "historical_metric": 0.00599,
        "historical_metric_note": "previous official score; protocol-warning reference",
        "journal": MACLA_JOURNAL,
        "node_id": "2e81a77222c7401c9c2585c5e6323905",
        "expected_code_sha256": (
            "b24b47c706ef6f8b0a5255500b561e5715c927ee9910500ca76c555e6df79555"
        ),
        "array_variants": {"macla_exact": "test_pred_final"},
        "official_submission_variant": "macla_exact",
        "memory_disposition_before_official_score": "protocol_warning_reference",
    },
    {
        "candidate_id": "leaf-official-dynamic-multibackbone-01008",
        "system_id": "dynamic_hybrid",
        "historical_metric": 0.01008,
        "historical_metric_note": "previous official NN-only score",
        "journal": DYNAMIC_JOURNAL,
        "node_id": "36537a9cab674391b848515dfdeca00b",
        "expected_code_sha256": (
            "6dc242b810cb63eba3df4c2e8f50defc50b1d7c1770be71866360416d1ec8431"
        ),
        "array_variants": {"nn_only": "test_preds"},
        "official_submission_variant": "nn_only",
        "memory_disposition_before_official_score": "verified_clean",
    },
    {
        "candidate_id": "leaf-official-gome-crossattention-01176",
        "system_id": "gome_style_port",
        "historical_metric": 0.01176,
        "historical_metric_note": "previous official score; protocol-warning reference",
        "journal": GOME_JOURNAL,
        "node_id": "05bb9616e28748d8b7aad1e7389e7ac9",
        "expected_code_sha256": (
            "d37fe1fb3a05699893c7c92ae2906519eab7d72cdc26c1f8fb00f1d71d41b9ce"
        ),
        "array_variants": {"gome_exact": "test_probs_norm"},
        "official_submission_variant": "gome_exact",
        "memory_disposition_before_official_score": "protocol_warning_reference",
    },
    {
        "candidate_id": "leaf-internal-dynamic-multibackbone-013847",
        "system_id": "dynamic_hybrid",
        "historical_metric": 0.013846588555656958,
        "historical_metric_note": "full OOF only; first official reproduction",
        "journal": DYNAMIC_JOURNAL,
        "node_id": "7ad758b35cdd4142a639cad67235b041",
        "expected_code_sha256": (
            "84d2a335d2a80f7fd1f791db0e772f9f1db7fdc9de345a796f613526b60168f2"
        ),
        "array_variants": {"dynamic_gradual_unfreeze": "test_preds"},
        "official_submission_variant": "dynamic_gradual_unfreeze",
        "memory_disposition_before_official_score": "verified_clean",
    },
    {
        "candidate_id": "leaf-internal-macla-lightgbm-034733",
        "system_id": "macla_style_port",
        "historical_metric": 0.034733155834457116,
        "historical_metric_note": "full OOF only; first official reproduction",
        "journal": MACLA_JOURNAL,
        "node_id": "e00c12199acf41538ed8366163bd6606",
        "expected_code_sha256": (
            "e65f7155d08dcf4c236fb5d0e9a145a05c394e23802cd7e6fa9e28c17b8a5900"
        ),
        "array_variants": {"macla_lgb_nn_blend": "test_probs_clipped"},
        "official_submission_variant": "macla_lgb_nn_blend",
        "memory_disposition_before_official_score": "verified_clean",
    },
    {
        "candidate_id": "leaf-internal-runforest-vit-attention-047483",
        "system_id": "runforest_only",
        "historical_metric": 0.047482858644683164,
        "historical_metric_note": "full OOF only; first official reproduction",
        "journal": RUNFOREST_JOURNAL,
        "node_id": "df9bccbe91dd435ab8e72658518246ee",
        "expected_code_sha256": (
            "90f5105822d2e9a3ed24a5b87503f51ce02a6cced15f0991d4942b78b73245cc"
        ),
        "array_variants": {"runforest_vit_attention": "test_probs"},
        "official_submission_variant": "runforest_vit_attention",
        "memory_disposition_before_official_score": "verified_clean",
    },
)


BY_ID = {candidate["candidate_id"]: candidate for candidate in CANDIDATES}
EXPECTED_CODE_SHA256 = {
    candidate["node_id"]: candidate["expected_code_sha256"]
    for candidate in CANDIDATES
}
