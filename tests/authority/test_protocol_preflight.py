from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from authority.protocol_execution_contract import (
    compile_protocol_execution_contract,
    write_contract_artifacts,
)
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.adapters import boosting, sklearn as sklearn_adapter, torch as torch_adapter
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.data_views import materialize_data_views
from protocol_runtime.preflight import (
    PreflightAdmissionError,
    PreflightStatus,
    ProtocolPreflightRunner,
    build_bounded_repair_receipt,
    preflight_cache_key,
    static_compatibility_check,
    validate_preflight_admission,
)


REGISTRY = ProtocolRegistry("mlevolve/config/protocols")
BUDGET = {
    "max_epochs": 1,
    "max_folds": 1,
    "max_models": 2,
    "timeout_seconds": 30,
}
IMAGE = "sha256:test-image"
SDK = hashlib.sha256(b"protocol-sdk-v1").hexdigest()


class PreflightBoostingRegressor:
    __module__ = "xgboost.sklearn"

    def fit(self, features, labels):
        self.value = sum(labels) / len(labels)
        return self

    def predict(self, features):
        return [self.value for _ in features]


def sklearn_candidate(session):
    from sklearn.tree import DecisionTreeClassifier

    views = session.get_split()
    model = sklearn_adapter.fit(
        session,
        DecisionTreeClassifier(max_depth=1, random_state=1),
        views.train,
        feature_keys=("x",),
        label_key="label",
    )
    predictions = sklearn_adapter.predict(
        session, model, views.validation, feature_keys=("x",)
    )
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection(
        "sklearn", based_on=views.validation, artifact_hash="1" * 64
    )


def boosting_candidate(session):
    views = session.get_split()
    model = boosting.fit(
        session,
        PreflightBoostingRegressor(),
        views.train,
        feature_keys=("x",),
        label_key="fare",
    )
    predictions = boosting.predict(
        session, model, views.validation, feature_keys=("x",)
    )
    session.evaluate_internal(views.validation, predictions, label_key="fare")
    session.freeze_selection(
        "boosting", based_on=views.validation, artifact_hash="2" * 64
    )


def torch_candidate(session):
    import torch

    views = session.get_split()
    with torch_adapter.fit_scope(
        session, component="torch_model", data_view=views.train
    ) as train_rows:
        weight = torch.tensor([float(len(train_rows))], requires_grad=True)
        (weight * 0.0).sum().backward()
    with torch_adapter.prediction_scope(
        session, component="torch_model", data_view=views.validation
    ) as validation_rows:
        predictions = [row["label"] for row in validation_rows]
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection(
        "torch", based_on=views.validation, artifact_hash="3" * 64
    )


def runtime_missing_evidence_candidate(session):
    session.get_split()
    if False:
        session.fit_scope(component="model", data_view=None)
        session.prediction_scope(component="model", data_view=None)
        session.evaluate_internal(None, [], label_key="label")
        session.freeze_selection("never", based_on=None)


def runtime_failure_candidate(session):
    session.get_split()
    if True:
        raise RuntimeError("intentional dry-run failure")
    session.fit_scope(component="model", data_view=None)
    session.prediction_scope(component="model", data_view=None)
    session.evaluate_internal(None, [], label_key="label")
    session.freeze_selection("never", based_on=None)


def invalid_terminal_candidate(session):
    terminal = "/data/terminal_holdout/labels.csv"
    views = session.get_split()
    session.fit_scope(component="model", data_view=views.train)
    session.prediction_scope(component="model", data_view=views.validation)
    session.evaluate_internal(views.validation, [], label_key="label")
    session.freeze_selection(terminal, based_on=views.validation)


def _case(name: str):
    if name == "sklearn":
        return (
            "random-classification@1",
            "preflight-cactus",
            "image",
            [
                {"sample_id": f"c-{label}-{index}", "label": label, "x": index}
                for label in (0, 1)
                for index in range(8)
            ],
            sklearn_candidate,
        )
    if name == "boosting":
        return (
            "chronological-regression@1",
            "preflight-taxi",
            "tabular",
            [
                {
                    "sample_id": f"t-{index}",
                    "timestamp": index,
                    "fare": index * 2.0,
                    "x": index,
                }
                for index in range(16)
            ],
            boosting_candidate,
        )
    return (
        "grouped-classification@1",
        "preflight-birds",
        "audio",
        [
            {
                "sample_id": f"b-{group}-{index}",
                "group_id": group,
                "label": [index % 2, (index + 1) % 2],
                "x": index,
            }
            for group in ("a", "b", "c", "d")
            for index in range(4)
        ],
        torch_candidate,
    )


def _prepared(tmp_path: Path, name: str):
    protocol, task, family, records, candidate = _case(name)
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve(protocol),
        task_id=task,
        task_family=family,
        train_view_ref=f"view://{task}/train",
        validation_view_ref=f"view://{task}/internal-validation",
        terminal_view_ref=f"evaluator-only://{task}/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
        adapter_spec={
            "managed": ["boosting", "sklearn"],
            "scope": ["torch"],
            "legacy_ast_positive_proof": False,
            "full_runtime_sdk_required": False,
        },
    )
    _manifest, manifest_path = materialize_data_views(
        records, tmp_path / f"views-{name}", contract, split_id=f"{task}-split"
    )
    return contract, identity, manifest_path, candidate


@pytest.mark.parametrize("name", ["sklearn", "boosting", "torch"])
def test_known_legal_references_pass_full_preflight(tmp_path: Path, name: str) -> None:
    if name == "torch":
        pytest.importorskip("torch")
    contract, identity, manifest_path, candidate = _prepared(tmp_path, name)
    source = inspect.getsource(candidate)
    output = tmp_path / f"preflight-{name}"
    report = ProtocolPreflightRunner(REGISTRY).run(
        candidate,
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=output,
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value
    assert report["terminal_exposure_count"] == 0
    assert report["terminal_score_computed"] is False
    assert report["result_fact_created"] is False
    assert report["closure_hash"]
    assert (output / "PREFLIGHT_EVIDENCE_CLOSURE.json").is_file()
    assert validate_preflight_admission(
        source,
        report_root=output,
        expected_contract_hash=contract.contract_hash,
    )["report_hash"] == report["report_hash"]


def test_static_protocol_violation_blocks_before_candidate_process(tmp_path: Path) -> None:
    contract, identity, manifest_path, _candidate = _prepared(tmp_path, "sklearn")
    source = inspect.getsource(invalid_terminal_candidate)
    static = static_compatibility_check(source, contract)
    assert static["status"] == PreflightStatus.PROTOCOL_VIOLATION.value
    assert any("terminal_path_access" in item for item in static["violations"])
    output = tmp_path / "invalid"
    report = ProtocolPreflightRunner(REGISTRY).run(
        invalid_terminal_candidate,
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=output,
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PROTOCOL_VIOLATION.value
    assert report["violations"] == static["violations"]
    assert report["missing_coverage"] == static["missing_coverage"]
    assert not (output / "collector").exists()
    with pytest.raises(PreflightAdmissionError, match="terminal_path_access") as exc:
        validate_preflight_admission(
            source,
            report_root=output,
            expected_contract_hash=contract.contract_hash,
        )
    assert exc.value.report["status"] == PreflightStatus.PROTOCOL_VIOLATION.value


def test_executor_surfaces_exact_preflight_violations_to_debug_state(
    tmp_path: Path,
) -> None:
    from agents.debug_agent import _protocol_preflight_recovery_guidance
    from engine.executor import Interpreter

    contract, identity, manifest_path, _candidate = _prepared(tmp_path, "sklearn")
    source = inspect.getsource(invalid_terminal_candidate)
    output = tmp_path / "executor-invalid"
    report = ProtocolPreflightRunner(REGISTRY).run(
        invalid_terminal_candidate,
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=output,
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(
                enabled=True,
                report_root=str(output),
                expected_contract_hash=contract.contract_hash,
            ),
        ),
        start_cpu_id="0",
        cpu_number="1",
    )

    result = Interpreter(tmp_path, timeout=5, max_parallel_run=1, cfg=cfg).run(
        source, id="detailed-preflight-block"
    )

    assert result.exc_type == "ProtocolPreflightError"
    assert result.exc_info["preflight_report"]["report_hash"] == report["report_hash"]
    assert any(
        "terminal_path_access" in violation
        for violation in result.exc_info["violations"]
    )
    assert result.protocol_observation["protocol_preflight"]["status"] == (
        PreflightStatus.PROTOCOL_VIOLATION.value
    )
    node = SimpleNamespace(
        protocol_preflight=result.protocol_observation["protocol_preflight"],
        exc_info=result.exc_info,
    )
    guidance = "\n".join(_protocol_preflight_recovery_guidance(node))
    assert "terminal_path_access" in guidance


def test_debug_guidance_ignores_retired_cv_fold_cap() -> None:
    from agents.debug_agent import _protocol_preflight_recovery_guidance

    node = SimpleNamespace(
        protocol_preflight={
            "status": PreflightStatus.PROTOCOL_VIOLATION.value,
            "violations": ["cv_fold_cap:StratifiedKFold:folds=5"],
            "missing_coverage": [],
            "missing_full_runtime_coverage": [],
            "missing_receipts": [],
        },
        exc_info={},
    )

    guidance = "\n".join(_protocol_preflight_recovery_guidance(node))
    assert "cv_fold_cap:StratifiedKFold:folds=5" in guidance
    assert "non-blocking under the current leakage-only Host policy" in guidance
    assert "Preserve the folds" in guidance
    assert "reduce the actual splitter/loop fold count" not in guidance


def test_code_review_active_host_protocol_does_not_restrict_cv_method(
    tmp_path: Path,
) -> None:
    from agents.code_review_agent import _deterministic_contract_audit

    contract, _identity, _manifest, _candidate = _prepared(tmp_path, "sklearn")
    contract_path, _sidecar = write_contract_artifacts(
        contract, tmp_path / "review-contract"
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace()),
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(
                enabled=True,
                contract_path=str(contract_path),
            )
        ),
    )
    audit = _deterministic_contract_audit(
        agent,
        "from sklearn.model_selection import StratifiedKFold\n"
        "splitter = StratifiedKFold(n_splits=5)\n",
    )

    assert audit is not None
    assert not any("cv_fold_cap:" in row for row in audit["violations"])


def test_host_static_preflight_is_leakage_only_not_method_policy(
    tmp_path: Path,
) -> None:
    contract, _identity, _manifest, _candidate = _prepared(tmp_path, "sklearn")
    source = '''import catboost
import torch
from sklearn.model_selection import StratifiedKFold

def candidate(session):
    views = session.get_split()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    models = [object(), object(), object(), object(), object()]
    remote_asset = "https://example.invalid/pretrained.bin"
    checkpoint = torch.load("./working/previous-candidate.pt")
    with session.fit_scope(component="ensemble", data_view=views.train):
        _ = (splitter, models, remote_asset, checkpoint)
    with session.prediction_scope(component="ensemble", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection("ensemble", based_on=views.validation, artifact_hash="0" * 64)
'''

    static = static_compatibility_check(source, contract)

    assert static["status"] == PreflightStatus.PASS.value
    forbidden_method_reasons = (
        "epoch_cap:",
        "cv_fold_cap:",
        "trainable_model_cap:",
        "dataset_wide_per_sample_precompute:",
        "unauthorized_import_roots:",
        "remote_asset_refs:",
        "unverified_local_loads:",
    )
    assert not any(
        str(reason).startswith(forbidden_method_reasons)
        for reason in static["violations"]
    )


def test_inference_bound_contract_requires_full_runtime_inference_scope(
    tmp_path: Path,
) -> None:
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("stratified-log-loss-classification@1"),
        task_id="leaf-classification",
        task_family="tabular",
        train_view_ref="view://leaf/train",
        validation_view_ref="view://leaf/validation",
        terminal_view_ref="evaluator-only://leaf/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
        adapter_spec={
            "managed": ["boosting", "sklearn"],
            "scope": ["torch"],
            "legacy_ast_positive_proof": False,
            "full_runtime_sdk_required": True,
            "inference_view_required": True,
            "inference_view_ref": "view://leaf/inference",
        },
    )
    source = '''from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)

if __name__ == "__main__":
    main()
'''
    static = static_compatibility_check(source, contract)

    assert "inference_scope" in static["missing_full_runtime_coverage"]


def test_inference_bound_contract_rejects_inference_before_selection_freeze(
    tmp_path: Path,
) -> None:
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("stratified-log-loss-classification@1"),
        task_id="leaf-classification",
        task_family="tabular",
        train_view_ref="view://leaf/train",
        validation_view_ref="view://leaf/validation",
        terminal_view_ref="evaluator-only://leaf/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
        adapter_spec={
            "managed": ["boosting", "sklearn"],
            "scope": ["torch"],
            "legacy_ast_positive_proof": False,
            "full_runtime_sdk_required": True,
            "inference_view_required": True,
            "inference_view_ref": "view://leaf/inference",
        },
    )
    source = '''from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    with session.inference_scope(component="model", data_view=views.inference):
        pass
    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)

if __name__ == "__main__":
    main()
'''

    static = static_compatibility_check(source, contract)

    assert "pre_selection_inference:main" in static["violations"]
    assert static["status"] == PreflightStatus.PROTOCOL_VIOLATION.value


def test_inference_bound_contract_accepts_inference_after_selection_freeze(
    tmp_path: Path,
) -> None:
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("stratified-log-loss-classification@1"),
        task_id="leaf-classification",
        task_family="tabular",
        train_view_ref="view://leaf/train",
        validation_view_ref="view://leaf/validation",
        terminal_view_ref="evaluator-only://leaf/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
        adapter_spec={
            "managed": ["boosting", "sklearn"],
            "scope": ["torch"],
            "legacy_ast_positive_proof": False,
            "full_runtime_sdk_required": True,
            "inference_view_required": True,
            "inference_view_ref": "view://leaf/inference",
        },
    )
    source = '''from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)
    with session.inference_scope(component="model", data_view=views.inference):
        pass

if __name__ == "__main__":
    main()
'''

    static = static_compatibility_check(source, contract)

    assert "pre_selection_inference:main" not in static["violations"]
    assert static["status"] == PreflightStatus.PASS.value


def test_inference_helper_definition_before_freeze_is_checked_at_call_site(
    tmp_path: Path,
) -> None:
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("stratified-log-loss-classification@1"),
        task_id="leaf-classification",
        task_family="tabular",
        train_view_ref="view://leaf/train",
        validation_view_ref="view://leaf/validation",
        terminal_view_ref="evaluator-only://leaf/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
        adapter_spec={
            "managed": ["boosting", "sklearn"],
            "scope": ["torch"],
            "legacy_ast_positive_proof": False,
            "full_runtime_sdk_required": True,
            "inference_view_required": True,
            "inference_view_ref": "view://leaf/inference",
        },
    )
    source = '''from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    def write_submission():
        with session.inference_scope(component="model", data_view=views.inference):
            pass
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)
    write_submission()

if __name__ == "__main__":
    main()
'''

    after_freeze = static_compatibility_check(source, contract)
    assert "pre_selection_inference:main" not in after_freeze["violations"]
    assert after_freeze["status"] == PreflightStatus.PASS.value

    before_freeze_source = source.replace(
        '    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)\n'
        "    write_submission()\n",
        "    write_submission()\n"
        '    session.freeze_selection("model", based_on=views.validation, artifact_hash="0" * 64)\n',
    )
    before_freeze = static_compatibility_check(before_freeze_source, contract)
    assert "pre_selection_inference:main" in before_freeze["violations"]
    assert before_freeze["status"] == PreflightStatus.PROTOCOL_VIOLATION.value


def test_dynamic_candidate_source_is_preflighted_before_admission(tmp_path: Path) -> None:
    contract, identity, manifest_path, _candidate = _prepared(tmp_path, "sklearn")
    source = '''def _run(session):
    from sklearn.tree import DecisionTreeClassifier
    from protocol_runtime.adapters import sklearn as managed
    views = session.get_split()
    model = managed.fit(
        session,
        DecisionTreeClassifier(max_depth=1, random_state=7),
        views.train,
        feature_keys=("x",),
        label_key="label",
    )
    predictions = managed.predict(
        session, model, views.validation, feature_keys=("x",)
    )
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection(
        "dynamic", based_on=views.validation, artifact_hash="4" * 64
    )

def candidate(session):
    _run(session)

from protocol_runtime import current_session

def main():
    _run(current_session())

if __name__ == "__main__":
    main()
'''
    report = ProtocolPreflightRunner(REGISTRY).run_source(
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "dynamic-source",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value
    assert validate_preflight_admission(
        source,
        report_root=tmp_path / "dynamic-source",
        expected_contract_hash=contract.contract_hash,
    )["report_hash"] == report["report_hash"]


def test_denoising_image_rmse_candidate_passes_host_preflight(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    asset_root = tmp_path / "source-images"
    asset_root.mkdir()
    records = []
    for index in range(8):
        noisy = asset_root / f"{index}-noisy.png"
        target = asset_root / f"{index}-target.png"
        image_module.new("L", (4, 3), color=index * 10).save(noisy)
        image_module.new("L", (4, 3), color=index * 10).save(target)
        records.append(
            {
                "sample_id": f"{index}.png",
                "_host_assets": {"noisy": str(noisy), "target": str(target)},
            }
        )
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("deterministic-random-regression@1"),
        task_id="denoising-dirty-documents",
        task_family="image",
        train_view_ref="view://denoising/train",
        validation_view_ref="view://denoising/internal-validation",
        terminal_view_ref="evaluator-only://denoising/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
    )
    _manifest, manifest_path = materialize_data_views(
        records,
        tmp_path / "denoising-views",
        contract,
        split_id="denoising-test-split",
    )
    source = '''def candidate(session):
    views = session.get_split()
    with session.fit_scope(component="denoising_model", data_view=views.train) as train_rows:
        if not train_rows:
            raise ValueError("Host training view is empty")
    with session.prediction_scope(component="denoising_model", data_view=views.validation) as validation_rows:
        predictions = [row["assets"]["noisy"] for row in validation_rows]
    session.evaluate_internal(views.validation, predictions, label_key="target")
    session.freeze_selection("host_protocol_dry_run", based_on=views.validation, artifact_hash="0" * 64)

from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="denoising_model", data_view=views.train) as train_rows:
        if not train_rows:
            raise ValueError("Host training view is empty")
    with session.prediction_scope(component="denoising_model", data_view=views.validation) as validation_rows:
        predictions = [row["assets"]["noisy"] for row in validation_rows]
    score = session.evaluate_internal(views.validation, predictions, label_key="target")
    session.freeze_selection("host_protocol_full_runtime", based_on=views.validation, artifact_hash="1" * 64)
    print(score)

if __name__ == "__main__":
    main()
'''
    candidate_only = source.split("from protocol_runtime import current_session", 1)[0]
    candidate_only_static = static_compatibility_check(candidate_only, contract)
    assert candidate_only_static["status"] == PreflightStatus.MISSING_EVIDENCE.value
    assert set(candidate_only_static["missing_full_runtime_coverage"]) == {
        "current_session",
        "evaluator",
        "fit_scope",
        "main_guard",
        "prediction_scope",
        "selection_freeze",
        "split_lineage",
    }
    output = tmp_path / "denoising-preflight"
    report = ProtocolPreflightRunner(REGISTRY).run_source(
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=output,
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value
    assert report["missing_receipts"] == []
    assert report["closure_hash"]


def test_aerial_binary_roc_auc_candidate_passes_host_preflight(
    tmp_path: Path,
) -> None:
    from agents.prompts.impl_guideline import _host_full_runtime_validation_source

    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("stratified-roc-auc-classification@1"),
        task_id="aerial-cactus-identification",
        task_family="image",
        train_view_ref="view://aerial/train",
        validation_view_ref="view://aerial/internal-validation",
        terminal_view_ref="evaluator-only://aerial/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
    )
    records = [
        {"sample_id": f"image-{index}", "label": index % 2, "x": index}
        for index in range(20)
    ]
    _manifest, manifest_path = materialize_data_views(
        records,
        tmp_path / "aerial-views",
        contract,
        split_id="aerial-roc-auc",
    )
    source = _host_full_runtime_validation_source(
        {
            "task_id": "aerial-cactus-identification",
            "label_key": "label",
            "metric_name": "roc_auc",
        }
    )
    report = ProtocolPreflightRunner(REGISTRY).run_source(
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "aerial-preflight",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value
    assert report["missing_receipts"] == []


def test_denoising_image_rmse_accepts_id_wrapped_predictions_but_uses_host_targets(
    tmp_path: Path,
) -> None:
    from PIL import Image
    import numpy as np

    noisy = tmp_path / "noisy.png"
    target = tmp_path / "target.png"
    Image.fromarray(np.full((5, 7), 64, dtype=np.uint8)).save(noisy)
    Image.fromarray(np.full((5, 7), 128, dtype=np.uint8)).save(target)
    records = [
        {
            "sample_id": f"image-{index}",
            "_host_assets": {"noisy": str(noisy), "target": str(target)},
        }
        for index in range(5)
    ]
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("deterministic-random-regression@1"),
        task_id="denoising-dirty-documents",
        task_family="image",
        train_view_ref="view://denoising/wrapped/train",
        validation_view_ref="view://denoising/wrapped/internal-validation",
        terminal_view_ref="evaluator-only://denoising/wrapped/terminal",
        execution_budget={"max_epochs": 1, "max_folds": 1, "max_models": 1, "timeout_seconds": 60},
        collector_spec=identity.collector_spec(),
    )
    _manifest, manifest_path = materialize_data_views(
        records, tmp_path / "wrapped-views", contract, split_id="wrapped"
    )
    source = '''def candidate(session):
    views = session.get_split()
    with session.fit_scope(component="m", data_view=views.train):
        pass
    with session.prediction_scope(component="m", data_view=views.validation) as rows:
        predictions = [(row["sample_id"], row["assets"]["noisy"], "untrusted") for row in rows]
    session.evaluate_internal(views.validation, predictions, label_key="target")
    session.freeze_selection("m", based_on=views.validation, artifact_hash="9" * 64)

from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="m", data_view=views.train):
        pass
    with session.prediction_scope(component="m", data_view=views.validation) as rows:
        predictions = [(row["sample_id"], row["assets"]["noisy"], "untrusted") for row in rows]
    session.evaluate_internal(views.validation, predictions, label_key="target")
    session.freeze_selection("m", based_on=views.validation, artifact_hash="9" * 64)

if __name__ == "__main__":
    main()
'''
    report = ProtocolPreflightRunner(REGISTRY).run_source(
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "wrapped-preflight",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value


def test_static_preflight_allows_final_prediction_after_freeze_but_not_refit(
    tmp_path: Path,
) -> None:
    contract, _identity, _manifest_path, _candidate = _prepared(tmp_path, "boosting")
    source = '''def candidate(session):
    views = session.get_split()
    with session.fit_scope(component="m", data_view=views.train):
        pass
    with session.prediction_scope(component="m", data_view=views.validation):
        pass
    session.evaluate_internal(views.validation, [0], label_key="fare")
    session.freeze_selection("m", based_on=views.validation, artifact_hash="a" * 64)

from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="m", data_view=views.train):
        pass
    with session.prediction_scope(component="m", data_view=views.validation):
        pass
    session.evaluate_internal(views.validation, [0], label_key="fare")
    session.freeze_selection("m", based_on=views.validation, artifact_hash="a" * 64)
    with session.prediction_scope(component="final_submission", data_view=views.validation):
        pass

if __name__ == "__main__":
    main()
'''
    report = static_compatibility_check(source, contract)
    assert report["status"] == PreflightStatus.PASS.value
    refit = source.replace(
        'with session.prediction_scope(component="final_submission", data_view=views.validation):',
        'with session.fit_scope(component="forbidden_refit", data_view=views.train):',
    )
    refit_report = static_compatibility_check(refit, contract)
    assert refit_report["status"] == PreflightStatus.PROTOCOL_VIOLATION.value
    assert any(
        violation.startswith("post_selection_tuning:main:")
        for violation in refit_report["violations"]
    )


def test_static_preflight_rejects_invalid_selection_lifecycle_before_gpu(
    tmp_path: Path,
) -> None:
    contract, _identity, _manifest, _candidate = _prepared(tmp_path, "boosting")
    source = '''def candidate(session):
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="fare")
    session.freeze_selection("dry", based_on=views.validation, artifact_hash="0" * 64)

from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    artifact_hash = "model_hash_" + str(hash(tuple(predictions)))
    session.freeze_selection("real", based_on=views.validation, artifact_hash=artifact_hash)
    session.evaluate_internal(views.validation, predictions, label_key="fare")

if __name__ == "__main__":
    main()
'''
    report = static_compatibility_check(source, contract)
    assert report["status"] == PreflightStatus.PROTOCOL_VIOLATION.value
    assert "selection_before_evaluation:main" in report["violations"]
    assert "post_selection_evaluation:main" in report["violations"]
    assert any(
        value.startswith("invalid_artifact_hash_expression:main:")
        for value in report["violations"]
    )

    checkpoint_source = source.replace(
        '    artifact_hash = "model_hash_" + str(hash(tuple(predictions)))\n',
        '    artifact_hash = "best_model.pt"\n',
    ).replace(
        '    session.freeze_selection("real", based_on=views.validation, artifact_hash=artifact_hash)\n'
        '    session.evaluate_internal(views.validation, predictions, label_key="fare")\n',
        '    session.evaluate_internal(views.validation, predictions, label_key="fare")\n'
        '    session.freeze_selection("real", based_on=views.validation, artifact_hash=artifact_hash)\n',
    )
    checkpoint_report = static_compatibility_check(checkpoint_source, contract)
    assert checkpoint_report["status"] == PreflightStatus.PASS.value


def test_chronological_internal_resplit_and_import_choice_are_method_freedom(
    tmp_path: Path,
) -> None:
    contract, _identity, _manifest, _candidate = _prepared(tmp_path, "boosting")
    source = """
import definitely_not_allowed
from sklearn.model_selection import train_test_split

def run_method(session):
    views = session.get_split()
    train_test_split([], [], shuffle=True)
    session.fit_scope(component='model', data_view=views.train)
    session.prediction_scope(component='model', data_view=views.validation)
    session.evaluate_internal(views.validation, [], label_key='fare')
    session.freeze_selection('x', based_on=views.validation)

def candidate(session):
    run_method(session)

from protocol_runtime import current_session

def main():
    run_method(current_session())

if __name__ == '__main__':
    main()
"""
    report = static_compatibility_check(source, contract)
    assert report["status"] == PreflightStatus.PASS.value
    assert not any("unauthorized_import_roots" in item for item in report["violations"])
    assert not any("chronological_random_resplit" in item for item in report["violations"])


def test_missing_evidence_is_not_protocol_violation(tmp_path: Path) -> None:
    contract, identity, manifest_path, _candidate = _prepared(tmp_path, "sklearn")
    source = inspect.getsource(runtime_missing_evidence_candidate)
    report = ProtocolPreflightRunner(REGISTRY).run(
        runtime_missing_evidence_candidate,
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "missing",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.MISSING_EVIDENCE.value
    assert report["responsible_component"] == "candidate_instrumentation"
    assert "fit_scope" in report["missing_receipts"]
    assert report["repairable"] is True


def test_runtime_failure_is_structured_and_has_no_result(tmp_path: Path) -> None:
    contract, identity, manifest_path, _candidate = _prepared(tmp_path, "sklearn")
    source = inspect.getsource(runtime_failure_candidate)
    report = ProtocolPreflightRunner(REGISTRY).run(
        runtime_failure_candidate,
        source=source,
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "runtime-failure",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.RUNTIME_FAILURE.value
    assert report["runtime_error"]["error_type"] == "RuntimeError"
    assert report["result_fact_created"] is False


def test_contract_mismatch_and_collector_identity_error_are_host_classified(
    tmp_path: Path,
) -> None:
    contract, identity, _manifest_path, candidate = _prepared(tmp_path, "sklearn")
    other_contract, _other_identity, other_manifest, _other_candidate = _prepared(
        tmp_path, "boosting"
    )
    report = ProtocolPreflightRunner(REGISTRY).run(
        candidate,
        source=inspect.getsource(candidate),
        contract=contract,
        identity=identity,
        data_view_manifest_path=other_manifest,
        output_root=tmp_path / "mismatch",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.CONTRACT_MISMATCH.value
    assert report["responsible_component"] == "host_data_view"

    attacker_identity = HostCollectorIdentity.generate()
    report = ProtocolPreflightRunner(REGISTRY).run(
        candidate,
        source=inspect.getsource(candidate),
        contract=contract,
        identity=attacker_identity,
        data_view_manifest_path=_manifest_path,
        output_root=tmp_path / "collector-error",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.COLLECTOR_INTERNAL_ERROR.value
    assert report["responsible_component"] == "host_collector"
    assert other_contract.contract_hash != contract.contract_hash


def test_cache_key_binds_code_contract_image_and_sdk() -> None:
    values = {
        "code_sha256": "a" * 64,
        "contract_hash": "b" * 64,
        "image_digest": "sha256:image-a",
        "sdk_hash": "c" * 64,
    }
    baseline = preflight_cache_key(**values)
    for key in values:
        changed = dict(values)
        changed[key] = values[key] + "-changed"
        assert preflight_cache_key(**changed) != baseline


def test_bounded_repair_preserves_or_changes_method_identity_explicitly() -> None:
    original = "def train(model, data):\n    return model.fit(data)\n"
    instrumentation = (
        "from protocol_runtime import current_session\n"
        "def train(model, data):\n"
        "    current_session().get_split()\n"
        "    return model.fit(data)\n"
    )
    receipt = build_bounded_repair_receipt(
        original,
        instrumentation,
        preflight_status=PreflightStatus.MISSING_EVIDENCE.value,
        repair_kind="instrumentation",
        attempt=1,
        max_attempts=1,
    )
    assert receipt["method_identity_preserved"] is True
    assert receipt["runtime_receipt_fabricated"] is False
    host_entrypoint = original + '''\n\ndef candidate(session):
    views = session.get_split()
    with session.fit_scope(component="model", data_view=views.train):
        pass
    with session.prediction_scope(component="model", data_view=views.validation):
        predictions = []
    session.evaluate_internal(views.validation, predictions, label_key="label")
    session.freeze_selection("dry-run", based_on=views.validation)
'''
    host_receipt = build_bounded_repair_receipt(
        original,
        host_entrypoint,
        preflight_status=PreflightStatus.MISSING_EVIDENCE.value,
        repair_kind="instrumentation",
        attempt=1,
        max_attempts=1,
    )
    assert host_receipt["method_identity_preserved"] is True
    changed = build_bounded_repair_receipt(
        original,
        original.replace("model.fit", "new_model.fit"),
        preflight_status=PreflightStatus.RUNTIME_FAILURE.value,
        repair_kind="budget_simplification",
        attempt=1,
        max_attempts=1,
    )
    assert changed["method_identity_preserved"] is False
    with pytest.raises(ValueError, match="terminal score"):
        build_bounded_repair_receipt(
            original,
            instrumentation,
            preflight_status=PreflightStatus.MISSING_EVIDENCE.value,
            repair_kind="instrumentation",
            attempt=1,
            max_attempts=1,
            terminal_score_observed=True,
        )


def test_node_selection_rejects_explicit_nonpass_preflight() -> None:
    from engine.node_selection import _preflight_selectable

    assert _preflight_selectable(SimpleNamespace(protocol_preflight={}))
    assert _preflight_selectable(
        SimpleNamespace(protocol_preflight={"status": "pass"})
    )
    assert not _preflight_selectable(
        SimpleNamespace(protocol_preflight={"status": "missing_evidence"})
    )
    assert _preflight_selectable(
        SimpleNamespace(
            protocol_preflight={
                "status": "missing_evidence",
                "enforcement_mode": "shadow",
            }
        )
    )


def test_executor_blocks_full_run_without_matching_preflight_report(
    tmp_path: Path,
) -> None:
    from engine.executor import Interpreter

    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(
                enabled=True,
                report_root=str(tmp_path / "reports"),
                expected_contract_hash="a" * 64,
            ),
        ),
        start_cpu_id="0",
        cpu_number="1",
    )
    result = Interpreter(tmp_path, timeout=5, max_parallel_run=1, cfg=cfg).run(
        "print('must not execute')", id="preflight-block"
    )
    assert result.exc_type == "ProtocolPreflightError"
    assert result.exc_info["candidate_subprocess_started"] is False


def test_host_shadow_records_missing_preflight_but_does_not_block_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.executor import Interpreter

    monkeypatch.setattr(
        "engine.executor.validate_preflight_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Host dry-run/admission must not run in shadow mode")
        ),
    )

    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(
                enabled=True,
                report_root=str(tmp_path / "reports"),
                expected_contract_hash="a" * 64,
                agent_controls_protocol_preflight=True,
            ),
        ),
        evaluation_authority=SimpleNamespace(
            mode="shadow",
            protocol_runtime_mode="host_sdk_shadow",
            runtime_protocol_observer_enabled=True,
        ),
        start_cpu_id="0",
        cpu_number="1",
    )
    result = Interpreter(tmp_path, timeout=5, max_parallel_run=1, cfg=cfg).run(
        "print('shadow candidate executed')", id="preflight-shadow"
    )
    assert result.exc_type is None
    assert "shadow candidate executed" in "".join(result.term_out)
    observation = result.protocol_observation["protocol_preflight"]
    assert observation["status"] == "agent_controlled"
    assert observation["enforcement_mode"] == "shadow"
    assert observation["admission_disposition"] == (
        "agent_review_then_execute"
    )
    assert observation["host_dry_run_executed"] is False
