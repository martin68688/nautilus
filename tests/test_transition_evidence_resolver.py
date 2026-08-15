import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _strict_node(
    node_id: str,
    code: str,
    metric: float,
    *,
    metric_provenance: str = "strict_internal_holdout",
) -> dict:
    return {
        "id": node_id,
        "type": "RunNode",
        "task": "leaf-classification",
        "code_sha256": _sha_text(code),
        "metric": metric,
        "is_buggy": False,
        "is_valid": True,
        "audit_status": "clean",
        "metric_disposition": "rank_eligible",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "quarantined": False,
        "protocol_biased": False,
        "metric_provenance": metric_provenance,
        "leakage_audit": {
            "status": "clean",
            "rank_eligible": True,
            "memory_disposition": "positive_eligible",
            "paper_grade_eligible": True,
        },
    }


def _materialized_fixture(tmp_path: Path):
    from experiments.end2end_memory_systems_20260804.build_transition_evidence_capsules import (
        build,
    )

    specs = [
        (
            "debug-run",
            "debug_fixed",
            "def train():\n    return 1 / 0\n",
            "def train():\n    return 1\n",
            0.9,
            0.5,
            "strict_internal_holdout",
        ),
        (
            "internal-run",
            "metric_improved",
            "LR = 1e-3\n",
            "LR = 3e-4\n",
            0.4,
            0.3,
            "strict_internal_holdout",
        ),
        (
            "official-run",
            "metric_improved",
            "MODEL = 'resnet18'\n",
            "MODEL = 'convnext_tiny'\n",
            0.4,
            0.2,
            "official_kaggle_scored_test",
        ),
        # Same executable pair under a different historical transition.  The
        # builder must retain this alias while deduplicating the pair/code.
        (
            "internal-alias-run",
            "metric_improved",
            "LR = 1e-3\n",
            "LR = 3e-4\n",
            0.4,
            0.3,
            "strict_internal_holdout",
        ),
    ]
    graph_nodes = []
    inventory_rows = []
    transition_ids = {}
    for index, (
        logical_run,
        outcome,
        before,
        after,
        parent_metric,
        child_metric,
        provenance,
    ) in enumerate(specs):
        attempt = "attempt-000"
        staged_run = f"{logical_run}::{attempt}"
        parent_raw_id = f"parent{index:06d}abcdefghijkl"
        child_raw_id = f"child{index:07d}abcdefghijkl"
        transition_id = (
            f"run::{staged_run}::transition::"
            f"{parent_raw_id[:12]}::{child_raw_id[:12]}"
        )
        parent_id = f"run::{staged_run}::node::parent"
        child_id = f"run::{staged_run}::node::child"
        transition_ids[logical_run] = transition_id
        graph_nodes.extend(
            [
                {
                    "id": parent_id,
                    "type": "RunNode",
                    "task": "leaf-classification",
                    "code_sha256": _sha_text(before),
                    "metric": parent_metric,
                    "is_buggy": outcome == "debug_fixed",
                },
                _strict_node(
                    child_id,
                    after,
                    child_metric,
                    metric_provenance=provenance,
                ),
                {
                    "id": transition_id,
                    "type": "Transition",
                    "task": "leaf-classification",
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "outcome": outcome,
                    "stage_pair": (
                        "debug->debug" if outcome == "debug_fixed" else "improve->improve"
                    ),
                    "parent_metric": parent_metric,
                    "child_metric": child_metric,
                    "metric_improvement": parent_metric - child_metric,
                },
            ]
        )
        relative = (
            f"mlevolve/runs/{logical_run}/{attempt}/journal.json"
        )
        journal_path = tmp_path / relative
        _write(
            journal_path,
            {
                "nodes": [
                    {"id": parent_raw_id, "code": before},
                    {"id": child_raw_id, "code": after},
                ],
                "node2parent": {child_raw_id: parent_raw_id},
            },
        )
        inventory_rows.append(
            {
                "path": relative,
                "size_bytes": journal_path.stat().st_size,
                "sha256": _sha_file(journal_path),
            }
        )

    sop_id = "recipe::leaf::improve"
    graph_nodes.append(
        {
            "id": sop_id,
            "type": "SOP",
            "task": "leaf-classification",
            "supporting_transition_ids": [
                transition_ids["internal-run"],
                transition_ids["official-run"],
            ],
        }
    )
    graph_path = tmp_path / "graph.json"
    inventory_path = tmp_path / "SOURCE_INVENTORY.json"
    _write(graph_path, {"meta": {"schema": "test"}, "nodes": graph_nodes})
    _write(
        inventory_path,
        {"inventory_sha256": "fixture-inventory", "files": inventory_rows},
    )
    payload = build(
        graph_path=graph_path,
        source_inventory_path=inventory_path,
        source_root=tmp_path,
        expected_debug_unique_pairs=1,
        expected_improve_unique_pairs=2,
    )
    capsule_path = tmp_path / "transition-evidence.json"
    _write(capsule_path, payload)
    nodes = {str(row["id"]): row for row in graph_nodes}
    return payload, capsule_path, graph_path, nodes, transition_ids, sop_id


def _resolver(tmp_path: Path):
    from agents.memory.evidence_resolver import TransitionEvidenceResolver

    payload, capsule_path, graph_path, nodes, transition_ids, sop_id = (
        _materialized_fixture(tmp_path)
    )
    resolver = TransitionEvidenceResolver(
        capsule_path=capsule_path,
        expected_file_sha256=_sha_file(capsule_path),
        graph_path=graph_path,
        graph_nodes=nodes,
        max_pairs=3,
    )
    return resolver, payload, capsule_path, graph_path, nodes, transition_ids, sop_id


def test_builder_materializes_strict_debug_and_improve_pairs_with_alias_dedup(
    tmp_path,
):
    payload, _capsule, _graph, _nodes, transition_ids, _sop_id = (
        _materialized_fixture(tmp_path)
    )
    assert payload["transition_count"] == 4
    assert payload["unique_pair_count"] == 3
    assert payload["debug_transition_count"] == 1
    assert payload["debug_unique_pair_count"] == 1
    assert payload["improve_transition_count"] == 3
    assert payload["improve_unique_pair_count"] == 2
    alias_pair = next(
        row for row in payload["pairs"] if len(row["alias_transition_ids"]) == 2
    )
    assert set(alias_pair["alias_transition_ids"]) == {
        transition_ids["internal-run"],
        transition_ids["internal-alias-run"],
    }


def test_resolver_opens_only_selected_sop_and_prefers_official_improve(tmp_path):
    resolver, _payload, _capsule, _graph, _nodes, transition_ids, sop_id = (
        _resolver(tmp_path)
    )
    selected = [
        {
            "id": sop_id,
            "source": "sop",
            "clean_supporting_transition_ids": [
                transition_ids["internal-run"],
                transition_ids["official-run"],
            ],
        }
    ]
    receipt = resolver.resolve(
        selected_items=selected,
        stage="improve",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert [row["id"] for row in selected] == [sop_id]
    assert receipt["selected_candidate_ids"] == [sop_id]
    assert receipt["opened_transition_ids"] == [transition_ids["official-run"]]
    opened = receipt["opened_evidence"][0]
    assert opened["evidence_class"] == "official_observed"
    assert opened["resolution_path"] == "selected_sop_to_supporting_transition"
    assert "-MODEL = 'resnet18'" in opened["canonical_diff"]
    assert "+MODEL = 'convnext_tiny'" in opened["canonical_diff"]
    assert "LR = 3e-4" not in opened["after_code"]
    assert receipt["fallback_used"] is False


def test_resolver_opens_exact_debug_transition_and_strategy_receives_code(tmp_path):
    from agents.memory_strategy_agent import build_memory_cards

    resolver, _payload, _capsule, _graph, _nodes, transition_ids, _sop_id = (
        _resolver(tmp_path)
    )
    transition_id = transition_ids["debug-run"]
    receipt = resolver.resolve(
        selected_items=[{"id": transition_id, "source": "runforest"}],
        stage="debug",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert receipt["opened_transition_ids"] == [transition_id]
    cards = build_memory_cards(
        {
            "resolved_evidence": receipt["opened_evidence"],
            "final_prompt_candidates": [
                {"candidate_id": transition_id, "prompt_text": "compact only"}
            ],
        },
        max_cards=1,
        card_max_chars=10000,
    )
    assert cards[0]["router_visibility"] == "resolved_evidence"
    assert cards[0]["resolved_transition_id"] == transition_id
    assert "return 1 / 0" in cards[0]["before_code"]
    assert "return 1" in cards[0]["after_code"]
    assert "historical_transition_diff" not in cards[0]


def test_resolver_rejects_payload_or_graph_identity_mismatch(tmp_path):
    from agents.memory.evidence_resolver import TransitionEvidenceResolver

    _resolver_instance, payload, capsule_path, graph_path, nodes, *_ = _resolver(
        tmp_path
    )
    tampered = copy.deepcopy(payload)
    tampered["code_blobs"][0]["code"] += "# tampered\n"
    tampered_path = tmp_path / "tampered.json"
    _write(tampered_path, tampered)
    with pytest.raises(ValueError, match="payload hash mismatch"):
        TransitionEvidenceResolver(
            capsule_path=tampered_path,
            expected_file_sha256=_sha_file(tampered_path),
            graph_path=graph_path,
            graph_nodes=nodes,
        )

    changed_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    changed_graph["meta"]["changed"] = True
    changed_graph_path = tmp_path / "changed-graph.json"
    _write(changed_graph_path, changed_graph)
    with pytest.raises(ValueError, match="different RunForest graph"):
        TransitionEvidenceResolver(
            capsule_path=capsule_path,
            expected_file_sha256=_sha_file(capsule_path),
            graph_path=changed_graph_path,
            graph_nodes=nodes,
        )


def test_first_stage_compaction_never_exposes_resolver_code_or_diff():
    from agents.memory.multigranular_grep import _compact

    compact = _compact(
        {
            "id": "transition::selected",
            "source": "runforest",
            "granularity": "transition",
            "row": {
                "task": "leaf-classification",
                "stage": "improve",
                "method_family": "vision",
                "after_code": "SECRET FULL SOURCE",
                "canonical_diff": "SECRET DIFF",
            },
            "fields": {"method": "safe compact summary"},
        }
    )
    serialized = json.dumps(compact, sort_keys=True)
    assert "SECRET FULL SOURCE" not in serialized
    assert "SECRET DIFF" not in serialized
    assert "safe compact summary" in serialized
