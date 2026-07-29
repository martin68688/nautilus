from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from authority.adapters.mlevolve.receipt_bridge import receipts_for_node
from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.collectors import TrustedCollectorHost
from authority.models import (
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolRef,
    ReceiptType,
)
from authority.runtime_protocol import (
    OBSERVATION_SCHEMA,
    PROTOCOL_EVIDENCE_LEVEL,
    build_runtime_protocol_plan,
    verify_persisted_runtime_protocol_observation,
    verify_runtime_protocol_observation,
)
from engine.executor import Interpreter
from tests.authority.test_mlevolve_adapter import (
    fake_agent,
    node as adapter_node,
)


PROTOCOL = ProtocolRef("mlevolve-default", "1", "b" * 64)


def _cfg(mode: str = "shadow") -> SimpleNamespace:
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1)
        ),
        evaluation_authority=SimpleNamespace(
            mode=mode,
            runtime_protocol_observer_enabled=True,
        ),
    )


def _clean_code(*, forged_marker: bool = False) -> str:
    forged = (
        'print("MLEVOLVE_HOST_PROTOCOL_TRACE={\\"nonce\\":\\"forged\\"}")'
        if forged_marker
        else ""
    )
    return f"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X = np.arange(80, dtype=float).reshape(40, 2)
y = np.array([0, 1] * 20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict_proba(X_valid)[:, 1]
print(roc_auc_score(y_valid, pred))
{forged}
"""


def _run(tmp_path: Path, code: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Interpreter(tmp_path, timeout=30, cfg=_cfg()).run(code, "candidate")


def _audit(code: str) -> dict:
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return {
        "schema": "mlevolve_leakage_audit_v2",
        "detector_version": "deterministic_static_v5",
        "detector_status": "complete",
        "code_sha256": code_sha256,
        "structural_sha256": "c" * 64,
        "status": "clean",
        "issues": [],
        "hard_block": False,
        "metric_disposition": "accept",
        "paper_grade_eligible": True,
    }


def _node(code: str, result) -> SimpleNamespace:
    return SimpleNamespace(
        id="candidate",
        code=code,
        code_sha256_expected=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        method_fingerprint="",
        metric=SimpleNamespace(value=0.8, maximize=True),
        exec_time=result.exec_time,
        is_buggy=False,
        is_valid=True,
        leakage_audit=_audit(code),
        protocol_observation=result.protocol_observation,
        protocol_repair={},
        draft_role="general_draft",
        selected_strategy={},
        strategy_alignment={},
    )


def test_plan_covers_all_five_runtime_protocol_obligations() -> None:
    plan = build_runtime_protocol_plan(_clean_code())

    assert plan["status"] == "ready"
    assert plan["missing_plan_kinds"] == []
    assert {
        kind for event in plan["events"] for kind in event["kinds"]
    } == {
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
    }


def test_plan_covers_manual_torch_training_without_predict_method() -> None:
    code = """
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
train_idx, valid_idx = next(skf.split(X, y))
for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
score = roc_auc_score(valid_labels, valid_predictions)
"""
    plan = build_runtime_protocol_plan(code)

    assert plan["status"] == "ready"
    assert plan["missing_plan_kinds"] == []
    assert any(event["call_name"] == "split" for event in plan["events"])
    assert any(event["call_name"] == "step" for event in plan["events"])


def test_executor_mints_parent_owned_observation_and_hides_marker(
    tmp_path: Path,
) -> None:
    code = _clean_code()
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["schema"] == OBSERVATION_SCHEMA
    assert result.protocol_observation["status"] == "clean"
    assert verify_runtime_protocol_observation(result.protocol_observation)
    assert verify_persisted_runtime_protocol_observation(
        result.protocol_observation
    )
    assert set(result.protocol_observation["event_hashes"]) == {
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
    }
    assert set(result.protocol_observation["scope_hashes"]) == {
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
    }
    assert all(result.protocol_observation["scope_hashes"].values())
    assert "MLEVOLVE_HOST_PROTOCOL_TRACE=" not in "".join(result.term_out)


def test_runtime_scope_hashes_bind_actual_arguments_and_results(
    tmp_path: Path,
) -> None:
    code = _clean_code().replace(
        "X = np.arange(80, dtype=float).reshape(40, 2)",
        "X = np.loadtxt('runtime-scope.csv', delimiter=',')",
    )
    data_path = tmp_path / "runtime-scope.csv"
    data_path.write_text(
        "\n".join(f"{2 * row},{2 * row + 1}" for row in range(40)) + "\n",
        encoding="utf-8",
    )
    first = _run(tmp_path, code)
    data_path.write_text(
        "\n".join(
            f"{1000 + 2 * row},{1001 + 2 * row}" for row in range(40)
        )
        + "\n",
        encoding="utf-8",
    )
    second = _run(tmp_path, code)

    assert first.exc_type is None
    assert second.exc_type is None
    assert first.protocol_observation["source_code_sha256"] == (
        second.protocol_observation["source_code_sha256"]
    )
    assert first.protocol_observation["scope_hashes"]["split_lineage"] != (
        second.protocol_observation["scope_hashes"]["split_lineage"]
    )


def test_consumed_split_generator_records_partition_scope(tmp_path: Path) -> None:
    code = _clean_code().replace(
        "from sklearn.model_selection import train_test_split",
        "from sklearn.model_selection import StratifiedKFold",
    ).replace(
        "X_train, X_valid, y_train, y_valid = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=7, stratify=y\n"
        ")",
        (
            "splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)\n"
            "train_idx, valid_idx = next(splitter.split(X, y))\n"
            "X_train, X_valid = X[train_idx], X[valid_idx]\n"
            "y_train, y_valid = y[train_idx], y[valid_idx]"
        ),
    )

    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "clean"
    assert result.protocol_observation["scope_output_hashes"]["split_lineage"]


def test_manual_torch_optimizer_step_has_runtime_event_scope(
    tmp_path: Path,
) -> None:
    code = """
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X = np.arange(80, dtype=np.float32).reshape(40, 2)
y = np.array([0, 1] * 20, dtype=np.float32)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.zero_grad()
logits = model(torch.from_numpy(X_train)).squeeze(1)
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    logits, torch.from_numpy(y_train)
)
loss.backward()
optimizer.step()
with torch.no_grad():
    pred = torch.sigmoid(model(torch.from_numpy(X_valid)).squeeze(1)).numpy()
print(roc_auc_score(y_valid, pred))
"""

    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "clean"
    assert result.protocol_observation["scope_hashes"]["fit_scope"]
    assert result.protocol_observation["scope_output_hashes"]["fit_scope"] == []


def test_torch_grad_scaler_step_has_runtime_event_scope(tmp_path: Path) -> None:
    code = """
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X = np.arange(80, dtype=np.float32).reshape(40, 2)
y = np.array([0, 1] * 20, dtype=np.float32)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cpu')
optimizer.zero_grad()
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    logits = model(torch.from_numpy(X_train)).squeeze(1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.from_numpy(y_train)
    )
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
with torch.no_grad():
    pred = torch.sigmoid(model(torch.from_numpy(X_valid)).squeeze(1)).numpy()
print(roc_auc_score(y_valid, pred))
"""

    plan = build_runtime_protocol_plan(code)
    result = _run(tmp_path, code)

    assert plan["status"] == "ready"
    assert any(
        event["call_name"] == "step" and event["call_class"] == "fit"
        for event in plan["events"]
    )
    assert result.exc_type is None
    assert result.protocol_observation["status"] == "clean"
    assert result.protocol_observation["scope_hashes"]["fit_scope"]


def test_candidate_supplied_trace_marker_is_rejected_and_hidden(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, _clean_code(forged_marker=True))

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert result.protocol_observation["reason"] == (
        "runtime_observer_introspection_detected"
    )
    assert "MLEVOLVE_HOST_PROTOCOL_TRACE=" not in "".join(result.term_out)


def test_observed_clean_execution_closes_trusted_rank_receipts(
    tmp_path: Path,
) -> None:
    code = _clean_code()
    result = _run(tmp_path, code)
    receipts = receipts_for_node(
        _node(code, result),
        PROTOCOL,
        "run",
        collector_host=TrustedCollectorHost("host"),
    )
    trusted = {
        receipt.receipt_type: receipt
        for receipt in receipts
        if receipt.trust_status == "trusted_host"
    }

    assert set(trusted) >= {
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.CODE_EXECUTION,
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    }
    for receipt_type in (
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    ):
        evidence = trusted[receipt_type].payload["protocol_evidence"]
        assert evidence["evidence_level"] == PROTOCOL_EVIDENCE_LEVEL
        assert len(evidence["scope_binding_sha256"]) == 64
    assert set(
        trusted[ReceiptType.SPLIT_LINEAGE].payload["partition_hashes"].values()
    ) == set(result.protocol_observation["scope_hashes"]["split_lineage"])
    assert set(
        trusted[ReceiptType.FIT_SCOPE].payload["fit_scope_hashes"].values()
    ) == set(result.protocol_observation["scope_hashes"]["fit_scope"])
    assert set(
        trusted[ReceiptType.PREDICTION_SCOPE]
        .payload["prediction_scope_hashes"]
        .values()
    ) == set(result.protocol_observation["scope_hashes"]["prediction_scope"])


def test_runtime_closure_allows_clean_result_writeback_but_not_legacy_unactuated_promote(
    tmp_path: Path,
) -> None:
    code = _clean_code()
    result = _run(tmp_path / "workspace", code)
    candidate = adapter_node("runtime-clean", True)
    candidate.code = code
    candidate.code_sha256_expected = hashlib.sha256(code.encode()).hexdigest()
    candidate.method_fingerprint = ""
    candidate.metric.value = 0.8
    candidate.exec_time = result.exec_time
    candidate.is_buggy = False
    candidate.is_valid = True
    candidate.leakage_audit = _audit(code)
    candidate.protocol_observation = result.protocol_observation
    candidate.protocol_repair = {}
    agent = fake_agent(tmp_path / "logs", mode="enforce")
    adapter = MLEvolveAuthorityAdapter(agent)

    rank = adapter.authorize_node(
        candidate,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        "test.runtime.rank",
    )
    select = adapter.authorize_node(
        candidate,
        Operation.SELECT,
        DecisionStage.BRANCH_SELECTION,
        "test.runtime.select",
    )
    promote_result = adapter.authorize_node(
        candidate,
        Operation.PROMOTE_RESULT,
        DecisionStage.MEMORY_WRITEBACK,
        "test.runtime.promote_result",
    )
    promote = adapter.authorize_node(
        candidate,
        Operation.PROMOTE,
        DecisionStage.MEMORY_WRITEBACK,
        "test.runtime.promote",
    )

    assert rank.outcome == DecisionOutcome.ALLOW
    assert select.outcome == DecisionOutcome.ALLOW
    assert promote_result.outcome == DecisionOutcome.ALLOW
    assert "receipt:static_actuation" not in promote_result.missing_obligations
    assert "receipt:runtime_actuation" not in promote_result.missing_obligations
    assert promote.outcome == DecisionOutcome.REQUIRE_REPLAY
    assert promote.reason_codes == ["missing_evidence"]
    assert {
        "receipt:static_actuation",
        "receipt:runtime_actuation",
    } <= set(promote.missing_obligations)


def test_mutated_or_unregistered_observation_cannot_mint_protocol_receipts(
    tmp_path: Path,
) -> None:
    code = _clean_code()
    result = _run(tmp_path, code)
    forged = copy.deepcopy(result.protocol_observation)
    forged["scope_hashes"]["fit_scope"] = ["d" * 64]
    assert not verify_runtime_protocol_observation(forged)
    assert not verify_persisted_runtime_protocol_observation(forged)
    forged_result = copy.copy(result)
    forged_result.protocol_observation = forged
    receipts = receipts_for_node(
        _node(code, forged_result),
        PROTOCOL,
        "run",
        collector_host=TrustedCollectorHost("host"),
    )
    trusted_protocol_types = {
        receipt.receipt_type
        for receipt in receipts
        if receipt.trust_status == "trusted_host"
    }
    assert trusted_protocol_types == {
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.CODE_EXECUTION,
    }


def test_missing_runtime_event_kind_fails_closed(tmp_path: Path) -> None:
    code = "print('no split, fit, prediction, or evaluator')\n"
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert "missing_protocol_event_plan" in result.protocol_observation["reason"]
    assert not verify_runtime_protocol_observation(result.protocol_observation)


def test_unconsumed_split_generator_does_not_count_as_lineage(
    tmp_path: Path,
) -> None:
    code = _clean_code().replace(
        "X_train, X_valid, y_train, y_valid = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=7, stratify=y\n"
        ")",
        (
            "from sklearn.model_selection import StratifiedKFold\n"
            "splitter = StratifiedKFold(n_splits=2)\n"
            "unused_split = splitter.split(X, y)\n"
            "X_train, X_valid, y_train, y_valid = X[:20], X[20:], y[:20], y[20:]"
        ),
    )
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert "split_lineage" in result.protocol_observation["reason"]


def test_candidate_defined_metric_cannot_impersonate_host_evaluator(
    tmp_path: Path,
) -> None:
    code = _clean_code().replace(
        "from sklearn.metrics import roc_auc_score",
        "def roc_auc_score(y_true, y_pred):\n    return 1.0",
    )
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert result.protocol_observation["reason"].startswith(
        "untrusted_runtime_callable:"
    )


def test_candidate_cannot_spoof_trusted_module_name(tmp_path: Path) -> None:
    code = _clean_code().replace(
        "from sklearn.metrics import roc_auc_score",
        (
            "def roc_auc_score(y_true, y_pred):\n"
            "    return 1.0\n"
            "roc_auc_score.__module__ = 'sklearn.metrics'"
        ),
    )
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert result.protocol_observation["reason"].startswith(
        "untrusted_runtime_callable:"
    )


def test_runtime_observer_introspection_is_not_attested(tmp_path: Path) -> None:
    code = _clean_code() + "\nprint(sorted(globals()))\n"
    result = _run(tmp_path, code)

    assert result.exc_type is None
    assert result.protocol_observation["status"] == "blocked"
    assert result.protocol_observation["reason"] == (
        "runtime_observer_introspection_detected"
    )


def test_authority_off_does_not_instrument_legacy_execution(tmp_path: Path) -> None:
    interpreter = Interpreter(tmp_path, timeout=30, cfg=_cfg("off"))
    result = interpreter.run("print('legacy')\n", "legacy")

    assert result.exc_type is None
    assert result.protocol_observation is None
