from __future__ import annotations

import copy

from authority.collectors import (
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    StaticActuationCollector,
    RuntimeActuationCollector,
    TrustedCollectorHost,
)
from authority.models import Claim, ClaimType, ProtocolRef, ProtocolSpec
from authority.protocol_registry import ProtocolRegistry


SOURCE_CODE = """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
features = vectorizer.fit_transform(train_text)
loss_name = "log_loss"
param_grid = {"C": [0.5, 1.0]}
ensemble_weights = [1.0]
model = LogisticRegression(C=1.0, max_iter=100)
model.fit(features, labels)
predictions = model.predict_proba(vectorizer.transform(valid_text))
"""


PROTOCOL_REPAIR_CODE = """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from agents.protocol_repair_runtime import ProtocolProvenanceGuard

guard = ProtocolProvenanceGuard()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, valid_idx = next(folds.split(train_text, labels))
guard.register_partition("outer_train", train_idx)
guard.register_partition("outer_holdout", valid_idx)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
features = vectorizer.fit_transform(train_text)
loss_name = "log_loss"
param_grid = {"C": [0.5, 1.0]}
ensemble_weights = [1.0]
model = LogisticRegression(C=1.0, max_iter=100)
model.fit(features, labels)
predictions = model.predict_proba(vectorizer.transform(valid_text))
"""


def build_registry() -> tuple[ProtocolRegistry, ProtocolRef]:
    registry = ProtocolRegistry()
    spec = registry.register(
        ProtocolSpec(
            protocol_id="clean-replay-test",
            version="1",
            task_profile={"family": "tabular"},
            data_split_policy={"kind": "stratified"},
            preprocessing_policy={"fit_scope": "fold_train"},
            evaluator_spec={"name": "log_loss"},
            metric_spec={"direction": "minimize"},
            selection_policy={"freeze_before_holdout": True},
            seed_policy={"pairwise_min_seeds": 3},
            holdout_policy={"terminal_only": True},
            promotion_policy={
                "require_complete_evidence": True,
                "clean_replay": {
                    "allowed_protocol_changes": [
                        "split_api",
                        "preprocessing_scope",
                        "instrumentation",
                    ]
                },
            },
            compatibility_rules={},
        )
    )
    return registry, spec.ref()


def historical_score_claim(
    claim_id: str,
    artifact_id: str,
    protocol_ref: ProtocolRef,
    method_fingerprint: str,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.SCORE,
        subject_artifact_id=artifact_id,
        task_scope={"task_id": "task-a", "task_families": ["tabular"]},
        method_fingerprint=method_fingerprint,
        protocol_ref=protocol_ref,
        statement="Historical score is protocol-invalid until clean replay.",
        source_artifact_refs=[artifact_id],
        boundary={"legacy_static_only": True},
        legacy_status="legacy_static_only",
    )


def trusted_replay_receipts(
    host: TrustedCollectorHost,
    *,
    artifact_id: str,
    protocol_ref: ProtocolRef,
    method_fingerprint: str,
    code_sha256: str,
):
    digest = "a" * 64
    specifications = (
        (
            MethodIdentityCollector,
            {"method_fingerprint": method_fingerprint, "code_sha256": code_sha256},
        ),
        (
            CodeExecutionCollector,
            {"exit_status": 0, "executed_path": artifact_id, "run_hash": "b" * 64},
        ),
        (
            SplitLineageCollector,
            {"partition_hashes": {"train": digest, "valid": "c" * 64}, "overlap_count": 0},
        ),
        (
            FitScopeCollector,
            {"fit_scope_hashes": {"vectorizer": "d" * 64}, "holdout_fit_count": 0},
        ),
        (
            PredictionScopeCollector,
            {"prediction_scope_hashes": {"valid": "e" * 64}, "forbidden_overlap_count": 0},
        ),
        (
            EvaluatorIntegrityCollector,
            {
                "evaluator_hash": "f" * 64,
                "inputs_hash": "1" * 64,
                "metric_direction": "minimize",
                "tampered": False,
            },
        ),
        (
            SelectionFreezeCollector,
            {"candidate_set_hash": "2" * 64, "frozen_before_holdout": True},
        ),
        (
            StaticActuationCollector,
            {"contract_hash": "3" * 64, "checks": {"method_preserved": True}},
        ),
        (
            RuntimeActuationCollector,
            {
                "contract_hash": "3" * 64,
                "event_hashes": ["4" * 64],
                "target_path_executed": True,
                "observations_hash": "5" * 64,
            },
        ),
    )
    return [
        host.collect(
            collector,
            artifact_id=artifact_id,
            run_id="clean-replay-run",
            protocol_ref=protocol_ref,
            source="tests.clean_replay_host",
            payload=copy.deepcopy(payload),
        )
        for collector, payload in specifications
    ]
