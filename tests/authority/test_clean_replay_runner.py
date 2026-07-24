from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from authority.clean_replay import ReplayQueueEntry
from authority.clean_replay_runner import (
    execute_replay_entry,
    validate_replay_execution_attempt,
)
from authority.collectors import TrustedCollectorHost
from authority.memory_snapshot import (
    ImmutableBaseBundle,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.models import ReceiptType
from authority.protocol_registry import ProtocolRegistry
from authority.runtime_protocol import verify_runtime_protocol_observation
from tests.test_memory_snapshot_overlay import build_tiny_bundle


REPO = Path(__file__).resolve().parents[2]
TASK_ID = "synthetic-image-domain-task"
RUN_ID = "synthetic-run"
PARENT_ID = f"run::{RUN_ID}::node::parent"
CHILD_ID = f"run::{RUN_ID}::node::child"
CLAIM_ID = "claim::synthetic-method"
CLAUSE_ID = "clause::synthetic-method"
METHOD_HYPOTHESIS = "Use deterministic logistic regression on the frozen features."


def _scientific_runtime_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import numpy, pandas, sklearn"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


SCIENTIFIC_RUNTIME_AVAILABLE = _scientific_runtime_available()


REPLAY_CODE = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
sample = pd.read_csv("input/sample_submission.csv")
features = ["x1", "x2"]
X_train, X_valid, y_train, y_valid = train_test_split(
    train[features],
    train["target"],
    test_size=0.25,
    random_state=17,
    stratify=train["target"],
)
model = LogisticRegression(C=1.0, max_iter=200, random_state=17)
model.fit(X_train, y_train)
valid_predictions = model.predict_proba(X_valid)[:, 1]
validation_score = roc_auc_score(y_valid, valid_predictions)
final_model = LogisticRegression(C=1.0, max_iter=200, random_state=17)
final_model.fit(train[features], train["target"])
sample["target"] = final_model.predict_proba(test[features])[:, 1]
sample.to_csv("submission/submission.csv", index=False)
print(f"Final Validation Score: {validation_score:.12f}")
""".lstrip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_manifest(bundle: Path, manifest: dict, relatives: list[str]) -> None:
    for relative in relatives:
        manifest["artifact_hashes"][relative] = sha256_file(bundle / relative)
    manifest["graph_hashes"]["runforest"] = manifest["artifact_hashes"][
        "runforest/graph.json"
    ]
    manifest["protocol_registry_hash"] = manifest["artifact_hashes"][
        "protocol_registry/mlevolve-default-v2.json"
    ]
    manifest["manifest_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
    )
    write_json_atomic(bundle / "manifest.json", manifest)


def _build_replay_bundle(tmp_path: Path) -> tuple[ImmutableBaseBundle, ReplayQueueEntry]:
    bundle_path, manifest = build_tiny_bundle(tmp_path / "bundle-root")
    code_sha256 = hashlib.sha256(REPLAY_CODE.encode("utf-8")).hexdigest()
    write_json_atomic(
        bundle_path / "runforest" / "graph.json",
        {
            "schema": "synthetic_clean_replay_graph_v1",
            "nodes": [
                {"id": f"run::{RUN_ID}", "type": "Run", "run_id": RUN_ID},
                {
                    "id": PARENT_ID,
                    "type": "RunNode",
                    "run_id": RUN_ID,
                    "raw_node_id": "parent",
                    "task": TASK_ID,
                },
                {
                    "id": CHILD_ID,
                    "type": "RunNode",
                    "run_id": RUN_ID,
                    "raw_node_id": "child",
                    "task": TASK_ID,
                    "code_sha256": code_sha256,
                    "metric_maximize": True,
                },
            ],
            "edges": [],
        },
    )
    write_json_atomic(
        bundle_path / "raw_journals" / RUN_ID / "journal.json",
        {
            "nodes": [
                {"id": "parent", "code": ""},
                {"id": "child", "code": REPLAY_CODE},
            ],
            "node2parent": {"child": "parent"},
        },
    )
    _write_jsonl(
        bundle_path / "authority" / "claims.jsonl",
        [
            {
                "claim_id": CLAIM_ID,
                "claim_type": "method_hypothesis",
                "subject_artifact_id": CHILD_ID,
            }
        ],
    )
    _write_jsonl(
        bundle_path / "sop" / "clauses.jsonl",
        [
            {
                "clause_id": CLAUSE_ID,
                "text": METHOD_HYPOTHESIS,
                "claim_refs": [CLAIM_ID],
                "source_artifact_refs": [CHILD_ID],
                "source_domains": ["image"],
                "transfer_scope": "same_domain",
            }
        ],
    )
    protocol_payload = json.loads(
        (REPO / "mlevolve" / "config" / "protocols" / "mlevolve-default-v2.json")
        .read_text(encoding="utf-8")
    )
    write_json_atomic(
        bundle_path / "protocol_registry" / "mlevolve-default-v2.json",
        protocol_payload,
    )
    _refresh_manifest(
        bundle_path,
        manifest,
        [
            "runforest/graph.json",
            f"raw_journals/{RUN_ID}/journal.json",
            "authority/claims.jsonl",
            "sop/clauses.jsonl",
            "protocol_registry/mlevolve-default-v2.json",
        ],
    )
    entry = ReplayQueueEntry(
        candidate_id="candidate::synthetic-child",
        task_id=TASK_ID,
        queue_rank=1,
        source_artifact_id=f"run::{RUN_ID}",
        parent_artifact_id=PARENT_ID,
        child_artifact_id=CHILD_ID,
        original_claim_id=CLAIM_ID,
        source_clause_id=CLAUSE_ID,
        code_sha256=code_sha256,
        method_hypothesis=METHOD_HYPOTHESIS,
        method_family="logistic-regression",
        audit_status="candidate_replay",
        source_refs=(
            f"run::{RUN_ID}",
            PARENT_ID,
            CHILD_ID,
            CLAIM_ID,
            CLAUSE_ID,
        ),
        protocol_issue_codes=("LEGACY_PROTOCOL_EVIDENCE_MISSING",),
        historical_metric_delta=0.1,
        selection_basis=("immutable_queue_rank",),
        historical_metric_used_as_evidence=False,
    ).finalize()
    return ImmutableBaseBundle.load(bundle_path), entry


def _write_task_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    with (data_dir / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "x1", "x2", "target"])
        writer.writeheader()
        for value in range(48):
            writer.writerow(
                {
                    "id": f"train-{value}",
                    "x1": float(value),
                    "x2": float(value % 7),
                    "target": value % 2,
                }
            )
    with (data_dir / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "x1", "x2"])
        writer.writeheader()
        for value in range(12):
            writer.writerow(
                {
                    "id": f"test-{value}",
                    "x1": float(value + 100),
                    "x2": float(value % 7),
                }
            )
    with (data_dir / "sample_submission.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target"])
        writer.writeheader()
        for value in range(12):
            writer.writerow({"id": f"test-{value}", "target": 0.0})


@pytest.mark.skipif(
    not SCIENTIFIC_RUNTIME_AVAILABLE,
    reason="requires a binary-compatible numpy/pandas/sklearn runtime",
)
def test_clean_replay_executes_real_observer_and_emits_certification_material(
    tmp_path: Path,
) -> None:
    bundle, entry = _build_replay_bundle(tmp_path)
    data_dir = tmp_path / "task-data"
    _write_task_data(data_dir)
    registry = ProtocolRegistry(bundle.path / "protocol_registry")
    protocol_ref = registry.get("mlevolve-default", "2").ref()
    attempt_dir = tmp_path / "attempt"

    record = execute_replay_entry(
        bundle=bundle,
        entry=entry,
        protocol_ref=protocol_ref,
        registry=registry,
        data_dir=data_dir,
        attempt_dir=attempt_dir,
        run_id="clean-replay::synthetic",
        timeout=30,
        cpu_number=1,
        num_gpus=1,
        collector_host=TrustedCollectorHost("synthetic-clean-replay-host"),
    )

    observation = json.loads(
        (attempt_dir / "protocol_observation.json").read_text(encoding="utf-8")
    )
    receipts = json.loads(
        (attempt_dir / "trusted_receipts.json").read_text(encoding="utf-8")
    )["receipts"]
    required_receipt_types = {
        ReceiptType.METHOD_IDENTITY.value,
        ReceiptType.CODE_EXECUTION.value,
        ReceiptType.SPLIT_LINEAGE.value,
        ReceiptType.FIT_SCOPE.value,
        ReceiptType.PREDICTION_SCOPE.value,
        ReceiptType.EVALUATOR.value,
        ReceiptType.SELECTION_FREEZE.value,
    }

    assert record["status"] == "certification_material_ready"
    assert record["execution_attempted"] is True
    assert record["execution_status"] == "completed"
    assert record["runtime_observation_verified_in_execution_host"] is True
    assert record["persisted_observation_integrity_valid"] is True
    assert record["failure_reasons"] == []
    assert set(record["trusted_receipt_types"]) == required_receipt_types
    assert {row["receipt_type"] for row in receipts} == required_receipt_types
    assert all(row["trust_status"] == "trusted_host" for row in receipts)
    assert verify_runtime_protocol_observation(observation)
    assert set(observation["scope_hashes"]) == {
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
    }
    assert record["submission"]["row_count"] == 12
    assert record["submission"]["columns"] == ["id", "target"]
    assert record["metric_value"] is not None
    assert record["replay_code_sha256"] == entry.code_sha256
    validated = validate_replay_execution_attempt(
        bundle=bundle,
        entry=entry,
        protocol_ref=protocol_ref,
        registry=registry,
        attempt_dir=attempt_dir,
    )
    assert validated.record == record
    assert validated.validation["status"] == "validated"
    assert validated.validation["runtime_host_verification_explicit"] is True
    assert validated.validation["runtime_host_verification_inferred"] is False
    assert len(validated.receipts) == 7
    bundle.assert_unchanged()


def test_clean_replay_static_block_happens_before_data_copy_or_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, entry = _build_replay_bundle(tmp_path)
    registry = ProtocolRegistry(bundle.path / "protocol_registry")
    protocol_ref = registry.get("mlevolve-default", "2").ref()
    attempt_dir = tmp_path / "blocked-attempt"

    monkeypatch.setattr(
        "authority.clean_replay_runner.audit_code",
        lambda _code: {
            "schema": "mlevolve_leakage_audit_v2",
            "detector_version": "deterministic_static_test",
            "detector_status": "complete",
            "code_sha256": entry.code_sha256,
            "structural_sha256": "a" * 64,
            "status": "blocked",
            "issues": [{"issue_code": "KNOWN_FATAL_TEST_ISSUE"}],
            "hard_block": True,
            "metric_disposition": "reject",
        },
    )

    def forbidden_interpreter(**_kwargs):
        raise AssertionError("static-blocked replay reached the Interpreter")

    record = execute_replay_entry(
        bundle=bundle,
        entry=entry,
        protocol_ref=protocol_ref,
        registry=registry,
        data_dir=tmp_path / "must-not-be-read",
        attempt_dir=attempt_dir,
        run_id="clean-replay::blocked",
        timeout=30,
        cpu_number=1,
        num_gpus=1,
        collector_host=TrustedCollectorHost("blocked-clean-replay-host"),
        interpreter_factory=forbidden_interpreter,
    )

    assert record["status"] == "blocked_static_audit"
    assert record["execution_attempted"] is False
    assert record["execution_status"] == "not_started_static_audit_blocked"
    assert record["runtime_observation_verified_in_execution_host"] is False
    assert record["persisted_observation_integrity_valid"] is False
    assert record["execution_time_seconds"] == 0.0
    assert record["metric_value"] is None
    assert record["submission"] is None
    assert record["trusted_receipt_types"] == []
    assert record["failure_reasons"] == [
        "deterministic_static_audit_not_clean"
    ]
    assert not (attempt_dir / "input").exists()
    assert not (attempt_dir / "working").exists()
    assert not (attempt_dir / "submission").exists()
    bundle.assert_unchanged()
