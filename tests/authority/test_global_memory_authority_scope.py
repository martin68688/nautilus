from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from agents.memory.global_memory import GlobalMemoryLayer
from agents.memory.record import MemRecord


PROTOCOL_REF = "test@1#protocol-hash"


def _decision() -> dict:
    return {
        "decision_id": "decision-1",
        "outcome": "allow",
        "policy_version": "policy-v1",
        "claim_id": "claim-1",
        "artifact_id": "artifact-1",
        "operation": "generate_candidate",
        "decision_stage": "retrieval",
        "generation_stage": "improve",
        "governance_stage": "retrieval",
        "permitted_scope": {
            "claim_types": ["method_hypothesis"],
            "operations": ["generate_candidate"],
            "stages": ["retrieval"],
            "protocol_hashes": ["protocol-hash"],
            "task_ids": ["task-1"],
            "generation_stages": ["improve"],
            "governance_stages": ["retrieval"],
        },
        "satisfied_paths": ["path-1"],
        "missing_obligations": [],
        "blocking_receipts": [],
        "required_action": None,
    }


def _layer(decision: dict | None = None, *, label: int = 1) -> GlobalMemoryLayer:
    record = MemRecord("node_artifact-1", "improve - artifact", "plan", "method", label)
    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.records = [record]
    layer.retriever = SimpleNamespace(search=lambda *_args, **_kwargs: [(record, 0.9)])
    layer.authority_mode = "enforce"
    layer.active_protocol_ref = PROTOCOL_REF
    layer.authority_policy_version = "policy-v1"
    layer.active_task_id = "task-1"
    layer.authority_decisions = {}
    layer.node_metadata_map = {
        record.record_id: {
            "artifact_id": "artifact-1",
            "task_id": "task-1",
            "claim_refs": ["claim-1"],
            "authority_decision_refs": ["decision-1"],
            "authority_decisions": [] if decision is None else [decision],
            "authority_policy_version": "policy-v1",
            "protocol_ref": PROTOCOL_REF,
        }
    }
    return layer


def _retrieve(layer: GlobalMemoryLayer):
    return layer.retrieve_similar_records(
        "query",
        authority_operation="generate_candidate",
        authority_generation_stage="improve",
        authority_governance_stage="retrieval",
        authority_task_id="task-1",
    )


def test_matching_outcome_scope_policy_protocol_and_stages_are_required() -> None:
    assert len(_retrieve(_layer(_decision()))) == 1


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("outcome",), "deny"),
        (("policy_version",), "stale-policy"),
        (("operation",), "rank"),
        (("generation_stage",), "draft"),
        (("governance_stage",), "branch_selection"),
        (("permitted_scope", "operations"), ["rank"]),
        (("permitted_scope", "protocol_hashes"), ["wrong-hash"]),
        (("permitted_scope", "task_ids"), ["other-task"]),
        (("permitted_scope", "generation_stages"), ["draft"]),
        (("permitted_scope", "governance_stages"), ["branch_selection"]),
    ],
)
def test_wrong_decision_field_is_suppressed(path, value) -> None:
    decision = copy.deepcopy(_decision())
    target = decision
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert _retrieve(_layer(decision)) == []


def test_pseudo_decision_ref_without_decision_cannot_pass_enforce() -> None:
    assert _retrieve(_layer(None)) == []


def test_staged_global_memory_enforcement_preserves_out_of_scope_legacy() -> None:
    layer = _layer(None)
    layer.authority_enforce_operations = {"generate_candidate"}
    layer.authority_enforce_generation_stages = {"debug"}
    layer.authority_enforce_governance_stages = {"retrieval"}

    # Improve is outside this canary window, so the established result remains.
    assert len(_retrieve(layer)) == 1
    # Debug is inside the window and lacks an allow decision, so it is suppressed.
    assert layer.retrieve_similar_records(
        "query",
        authority_operation="generate_candidate",
        authority_generation_stage="debug",
        authority_governance_stage="retrieval",
        authority_task_id="task-1",
    ) == []
    # Unknown generation stage cannot be used to evade a configured window.
    assert layer.retrieve_similar_records(
        "query",
        authority_operation="generate_candidate",
        authority_generation_stage=None,
        authority_governance_stage="retrieval",
        authority_task_id="task-1",
    ) == []


def test_wrong_record_protocol_or_metadata_policy_is_suppressed() -> None:
    layer = _layer(_decision())
    layer.node_metadata_map["node_artifact-1"]["protocol_ref"] = "test@2#other"
    assert _retrieve(layer) == []
    layer = _layer(_decision())
    layer.node_metadata_map["node_artifact-1"]["authority_policy_version"] = "stale"
    assert _retrieve(layer) == []


def test_negative_memory_is_visible_only_to_debug_or_inspect_views() -> None:
    layer = _layer(None, label=-1)
    assert _retrieve(layer) == []
    debug = layer.retrieve_similar_records(
        "query",
        authority_operation="debug_hypothesis",
        authority_generation_stage="debug",
        authority_governance_stage="retrieval",
        authority_task_id="task-1",
    )
    assert len(debug) == 1


def test_authority_decision_snapshot_is_persisted_and_reloaded(tmp_path) -> None:
    class FakeRetriever:
        vector_index = None

        def build_index(self, _records, _texts):
            self.vector_index = object()

        def add_to_index(self, _records, _texts):
            return None

    layer = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    layer.memory_dir = tmp_path
    layer._lock = threading.RLock()
    layer._load_error = None
    layer.records = []
    layer.node_metadata_map = {}
    layer.retriever = FakeRetriever()
    layer.authority_mode = "enforce"
    layer.active_protocol_ref = PROTOCOL_REF
    layer.authority_policy_version = "policy-v1"
    layer.active_task_id = "task-1"
    layer.authority_decisions = {"decision-1": _decision()}

    node = SimpleNamespace(
        id="artifact-1",
        stage="improve",
        plan="plan",
        code="print('ok')",
        code_summary="method",
        metric=SimpleNamespace(value=0.5, maximize=True),
        exec_time=1.0,
        is_buggy=False,
        claim_refs=["claim-1"],
        authority_decision_refs=["decision-1"],
        protocol_ref=PROTOCOL_REF,
        protocol_repair={},
    )
    assert layer.save_node(node) is True

    persisted = json.loads((tmp_path / "records.json").read_text(encoding="utf-8"))
    assert persisted[0]["authority_policy_version"] == "policy-v1"
    assert persisted[0]["authority_decisions"][0]["decision_id"] == "decision-1"

    reloaded = GlobalMemoryLayer.__new__(GlobalMemoryLayer)
    reloaded.memory_dir = tmp_path
    reloaded._lock = threading.RLock()
    reloaded._load_error = None
    reloaded.records = []
    reloaded.node_metadata_map = {}
    reloaded.retriever = FakeRetriever()
    reloaded._load_memory()
    metadata = reloaded.node_metadata_map["node_artifact-1"]
    assert metadata["authority_decision_refs"] == ["decision-1"]
    assert metadata["authority_decisions"][0]["permitted_scope"]["operations"] == [
        "generate_candidate"
    ]
