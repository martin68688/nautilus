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
    endpoint_ids = {}
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
        endpoint_ids[logical_run] = (parent_id, child_id)
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
        graph_nodes[-3]["original_node_id"] = parent_raw_id
        graph_nodes[-2]["original_node_id"] = child_raw_id
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
    debug_parent_id, debug_child_id = endpoint_ids["debug-run"]
    # Legacy formal atomic Debug evidence used "accept" before the newer
    # rank_eligible spelling.  Its independently audited atomic claim is the
    # authority that permits Debug-only materialization.
    next(
        row for row in graph_nodes if row.get("id") == debug_child_id
    )["metric_disposition"] = "accept"
    atomic_id = "atomic-transition::leaf-classification::fixture-debug"
    graph_nodes.append(
        {
            "id": atomic_id,
            "type": "Transition",
            "task": "leaf-classification",
            "parent_node_id": debug_parent_id,
            "child_node_id": debug_child_id,
            "outcome": "debug_fixed",
            "stage_pair": "debug->debug",
            "quarantined": False,
            "protocol_biased": False,
            "atomic_repair_claim": {
                "schema": "mlevolve_atomic_memory_claim_v1",
                "id": "claim::leaf-classification::repair_claim::fixture-debug",
                "claim_status": "authorized_debug_only",
                "task_id": "leaf-classification",
                "outcome": "debug_fixed",
                "source_transition_id": transition_ids["debug-run"],
                "source_parent_node_id": debug_parent_id,
                "source_child_node_id": debug_child_id,
                "operation_visibility": {
                    "allowed_operations": ["debug_hypothesis", "debug_repair"],
                    "forbidden_operations": [
                        "improve_method_selection",
                        "metric_ranking",
                    ],
                },
                "taint": {
                    "claim": "clean",
                    "code": "clean",
                    "source_program": {
                        "status": "clean",
                        "rank_eligible": True,
                        "memory_disposition": "positive_eligible",
                    },
                },
                "verification": {
                    "before_code_sha256": _sha_text(specs[0][2]),
                    "after_code_sha256": _sha_text(specs[0][3]),
                    "claim_scope_independently_audited": True,
                    "full_program_clean": True,
                    "observed_child_execution_success": True,
                    "observed_parent_failure": True,
                    "repair_action_bound_to_transition": True,
                },
                "repair_action": "Replace the crashing denominator with a safe value.",
                "before_after": [
                    {"symbol": "train", "before": "return 1 / 0", "after": "return 1"}
                ],
                "failure_signature": {"exception_names": ["ZeroDivisionError"]},
            },
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
    assert payload["candidate_alias_count"] == 2
    assert payload["materialized_candidate_alias_count"] == 2
    alias_pair = next(
        row for row in payload["pairs"] if len(row["alias_transition_ids"]) == 2
    )
    assert set(alias_pair["alias_transition_ids"]) == {
        transition_ids["internal-run"],
        transition_ids["internal-alias-run"],
    }


def test_augment_recovers_frozen_atomic_source_program_and_aliases(tmp_path):
    from agents.memory.evidence_resolver import TransitionEvidenceResolver
    from experiments.end2end_memory_systems_20260804.build_transition_evidence_capsules import (
        _pair_rows,
        _payload_hash,
        augment_atomic_aliases,
    )

    payload, _capsule, graph_path, nodes, transition_ids, _sop_id = (
        _materialized_fixture(tmp_path)
    )
    debug_transition_id = transition_ids["debug-run"]
    debug_row = next(
        row
        for row in payload["transitions"]
        if row["transition_id"] == debug_transition_id
    )
    endpoint_ids = {debug_row["parent_node_id"], debug_row["child_node_id"]}
    endpoint_shas = {
        debug_row["before_code_sha256"],
        debug_row["after_code_sha256"],
    }
    base = copy.deepcopy(payload)
    base["transitions"] = [
        row
        for row in base["transitions"]
        if row["transition_id"] != debug_transition_id
    ]
    base["nodes"] = [
        row for row in base["nodes"] if row["node_id"] not in endpoint_ids
    ]
    base["code_blobs"] = [
        row
        for row in base["code_blobs"]
        if row["code_sha256"] not in endpoint_shas
    ]
    base["pairs"] = _pair_rows(base["transitions"])
    base["journal_sha256s"] = [
        value
        for value in base["journal_sha256s"]
        if value != debug_row["source_journal_sha256"]
    ]
    base["capsule_sha256"] = _payload_hash(base, "capsule_sha256")
    base_path = tmp_path / "base-with-missing-atomic-program.json"
    _write(base_path, base)

    journal_path = tmp_path / debug_row["source_journal"]
    augmented = augment_atomic_aliases(
        base_capsule_path=base_path,
        graph_path=graph_path,
        atomic_journal_paths=[journal_path],
    )
    assert debug_transition_id in augmented["atomic_alias_extension"][
        "recovered_transition_ids"
    ]
    assert augmented["atomic_alias_extension"]["extension_journal_sha256s"] == [
        debug_row["source_journal_sha256"]
    ]
    aliases = {
        row["candidate_id"]: row for row in augmented["candidate_aliases"]
    }
    atomic_id = "atomic-transition::leaf-classification::fixture-debug"
    repair_id = "repair-claim::leaf-classification::fixture-debug"
    assert aliases[atomic_id]["materialized"] is True
    assert aliases[repair_id]["materialized"] is True

    augmented_path = tmp_path / "augmented.json"
    _write(augmented_path, augmented)
    resolver = TransitionEvidenceResolver(
        capsule_path=augmented_path,
        expected_file_sha256=_sha_file(augmented_path),
        graph_path=graph_path,
        graph_nodes=nodes,
    )
    receipt = resolver.resolve(
        selected_items=[{"id": atomic_id, "source": "runforest"}],
        stage="debug",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert receipt["opened_transition_ids"] == [debug_transition_id]


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


def test_resolver_bridges_atomic_and_repair_claim_aliases_without_stage_leakage(
    tmp_path,
):
    resolver, _payload, _capsule, _graph, _nodes, transition_ids, _sop_id = (
        _resolver(tmp_path)
    )
    source_transition_id = transition_ids["debug-run"]
    atomic_id = "atomic-transition::leaf-classification::fixture-debug"
    repair_id = "repair-claim::leaf-classification::fixture-debug"

    atomic_receipt = resolver.resolve(
        selected_items=[{"id": atomic_id, "source": "runforest"}],
        stage="debug",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert atomic_receipt["selected_candidate_ids"] == [atomic_id]
    assert atomic_receipt["selected_ids_unchanged"] is True
    assert atomic_receipt["opened_transition_ids"] == [source_transition_id]
    assert atomic_receipt["opened_evidence"][0]["resolution_path"] == (
        "selected_atomic_alias_to_source_transition"
    )

    repair_receipt = resolver.resolve(
        selected_items=[{"id": repair_id, "source": "sop"}],
        stage="debug",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert repair_receipt["opened_transition_ids"] == [source_transition_id]
    assert repair_receipt["opened_evidence"][0]["resolution_path"] == (
        "selected_repair_claim_alias_to_source_transition"
    )
    assert repair_receipt["opened_evidence"][0]["evidence_class"] == (
        "debug_repair_reference"
    )

    improve_receipt = resolver.resolve(
        selected_items=[{"id": repair_id, "source": "sop"}],
        stage="improve",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert improve_receipt["status"] == "resolved"
    assert improve_receipt["opened_transition_ids"] == [source_transition_id]
    opened = improve_receipt["opened_evidence"][0]
    assert opened["evidence_class"] == "debug_repair_reference"
    assert opened["source_evidence_class"] == "strict_debug_observed"
    assert opened["metric_authority"] == "reference_only"
    assert opened["metric_authorized"] is False
    assert opened["code_visibility"] == "full_transition_code"
    assert "return 1" in opened["after_code"]


def test_resolver_preserves_unavailable_atomic_alias_identity_without_sop_fallback(
    tmp_path,
):
    from agents.memory.evidence_resolver import TransitionEvidenceResolver
    from experiments.end2end_memory_systems_20260804.build_transition_evidence_capsules import (
        _payload_hash,
    )

    _resolver_instance, payload, _capsule, graph_path, nodes, *_ = _resolver(
        tmp_path
    )
    atomic_id = "atomic-transition::leaf-classification::fixture-debug"
    repair_id = "repair-claim::leaf-classification::fixture-debug"
    without_aliases = copy.deepcopy(payload)
    without_aliases["candidate_aliases"] = []
    without_aliases["candidate_alias_count"] = 0
    without_aliases["materialized_candidate_alias_count"] = 0
    without_aliases["capsule_sha256"] = _payload_hash(
        without_aliases, "capsule_sha256"
    )
    capsule_path = tmp_path / "transition-evidence-without-aliases.json"
    _write(capsule_path, without_aliases)
    resolver = TransitionEvidenceResolver(
        capsule_path=capsule_path,
        expected_file_sha256=_sha_file(capsule_path),
        graph_path=graph_path,
        graph_nodes=nodes,
    )

    atomic_receipt = resolver.resolve(
        selected_items=[{"id": atomic_id, "source": "runforest"}],
        stage="debug",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert atomic_receipt["status"] == "unresolved"
    assert atomic_receipt["unresolved"] == [
        {
            "candidate_id": atomic_id,
            "resolution_path": "selected_atomic_alias_to_source_transition",
            "reason": "no_stage_compatible_materialized_transition",
            "candidate_transition_ids": [atomic_id],
        }
    ]

    repair_receipt = resolver.resolve(
        selected_items=[{"id": repair_id, "source": "sop"}],
        stage="improve",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [atomic_id],
    )
    assert repair_receipt["status"] == "resolved"
    assert repair_receipt["opened_transition_ids"] == [
        next(
            row["atomic_repair_claim"]["source_transition_id"]
            for row in nodes.values()
            if row.get("id") == atomic_id
        )
    ]
    opened = repair_receipt["opened_evidence"][0]
    assert opened["evidence_class"] == "debug_repair_reference"
    assert opened["metric_authority"] == "reference_only"
    assert opened["code_visibility"] == "full_transition_code"


def test_resolver_exposes_verified_repair_diff_when_full_program_is_unavailable(
    tmp_path,
):
    from agents.memory_strategy_agent import build_memory_cards
    from agents.memory.evidence_resolver import TransitionEvidenceResolver
    from experiments.end2end_memory_systems_20260804.build_transition_evidence_capsules import (
        _pair_rows,
        _payload_hash,
    )

    _resolver_instance, payload, _capsule, graph_path, nodes, transition_ids, _ = (
        _resolver(tmp_path)
    )
    source_transition_id = transition_ids["debug-run"]
    source_row = next(
        row for row in payload["transitions"]
        if row["transition_id"] == source_transition_id
    )
    endpoint_ids = {source_row["parent_node_id"], source_row["child_node_id"]}
    endpoint_shas = {
        source_row["before_code_sha256"],
        source_row["after_code_sha256"],
    }
    restricted = copy.deepcopy(payload)
    restricted["candidate_aliases"] = []
    restricted["candidate_alias_count"] = 0
    restricted["materialized_candidate_alias_count"] = 0
    restricted["transitions"] = [
        row for row in restricted["transitions"]
        if row["transition_id"] != source_transition_id
    ]
    restricted["nodes"] = [
        row for row in restricted["nodes"] if row["node_id"] not in endpoint_ids
    ]
    restricted["code_blobs"] = [
        row for row in restricted["code_blobs"]
        if row["code_sha256"] not in endpoint_shas
    ]
    restricted["pairs"] = _pair_rows(restricted["transitions"])
    restricted["capsule_sha256"] = _payload_hash(
        restricted, "capsule_sha256"
    )
    capsule_path = tmp_path / "transition-evidence-repair-diff-only.json"
    _write(capsule_path, restricted)
    resolver = TransitionEvidenceResolver(
        capsule_path=capsule_path,
        expected_file_sha256=_sha_file(capsule_path),
        graph_path=graph_path,
        graph_nodes=nodes,
    )
    repair_id = "repair-claim::leaf-classification::fixture-debug"
    receipt = resolver.resolve(
        selected_items=[{"id": repair_id, "source": "sop"}],
        stage="improve",
        task_id="leaf-classification",
        active_transitions_for_sop=lambda _sop_id: [],
    )
    assert receipt["status"] == "resolved"
    assert receipt["opened_transition_ids"] == [source_transition_id]
    opened = receipt["opened_evidence"][0]
    assert opened["evidence_class"] == "debug_repair_reference"
    assert opened["metric_authority"] == "reference_only"
    assert opened["code_visibility"] == "verified_repair_diff"
    assert "- train: return 1 / 0" in opened["repair_diff"]
    assert "+ train: return 1" in opened["repair_diff"]
    assert "before_code" not in opened
    assert "after_code" not in opened
    cards = build_memory_cards(
        {"resolved_evidence": receipt["opened_evidence"]},
        max_cards=1,
        card_max_chars=10000,
    )
    assert cards[0]["router_visibility"] == "resolved_evidence"
    assert cards[0]["evidence_class"] == "debug_repair_reference"
    assert cards[0]["metric_authority"] == "reference_only"
    assert "+ train: return 1" in cards[0]["repair_diff"]


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
