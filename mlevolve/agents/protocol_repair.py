"""Generic staged repair for biased evaluation protocols.

The ordinary debug loop is a poor fit for solutions whose model direction is
useful but whose split, fitting, model-selection, or reporting scopes are not
trustworthy.  This module turns such repairs into a fail-closed transaction:
each stage has one narrow contract, preserves the original model design, and
cannot execute on a GPU until every stage is complete.

Nothing here is tied to a dataset or model name.  The stage plan is derived
from task modality, split semantics, and capabilities visible in the code.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
import uuid
from typing import Any

from agents.leakage_audit import build_repair_preservation_contract, code_sha256


PROTOCOL_REPAIR_SCHEMA = "mlevolve_protocol_repair_v1"
PROTOCOL_PLAN_SCHEMA = "mlevolve_protocol_plan_v1"
PROTOCOL_STAGE_AUDIT_SCHEMA = "mlevolve_protocol_stage_audit_v1"
RUNTIME_MARKER = "MLEVOLVE_PROTOCOL_PROVENANCE="

_REPAIRABLE_CODES = {
    "TRANSFORM_FIT_ON_HOLDOUT",
    "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
    "CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE",
    "RESET_INDEX_ORIGINAL_ARRAY_MISALIGNMENT",
    "LLM_TRANSDUCTIVE_CONTAMINATION",
    "LLM_SELECTION_BIAS",
}
_REPAIRABLE_CATEGORIES = {"selection_bias", "transductive_contamination"}
_ISSUE_STAGE = {
    "TRANSFORM_FIT_ON_HOLDOUT": "data_scope",
    "RESET_INDEX_ORIGINAL_ARRAY_MISALIGNMENT": "data_scope",
    "LLM_TRANSDUCTIVE_CONTAMINATION": "data_scope",
    "CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE": "cross_fit",
    "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION": "selection_freeze",
    "LLM_SELECTION_BIAS": "selection_freeze",
}


def _cfg(agent: Any, name: str, default: Any) -> Any:
    config = getattr(getattr(agent, "acfg", None), "protocol_repair", None)
    return getattr(config, name, default) if config is not None else default


def _contains(code: str, pattern: str) -> bool:
    return re.search(pattern, code or "", re.IGNORECASE | re.MULTILINE | re.DOTALL) is not None


def _ast_facts(code: str) -> dict:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {"valid": False, "calls": set(), "assigned": set()}
    calls = set()
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    return {"valid": True, "calls": calls, "assigned": assigned}


def infer_task_profile(task_desc: str, code: str, supplied: dict | None = None) -> dict:
    """Infer only protocol-relevant task traits; never guess a model family."""
    supplied = copy.deepcopy(supplied or {})
    text = f"{task_desc}\n{code}".lower()
    modality = str(supplied.get("modality") or "").lower()
    if not modality:
        if any(token in text for token in ("image", "jpeg", "png", "torchvision", "albumentations")):
            modality = "image"
        elif any(token in text for token in ("text", "tokenizer", "tfidf", "transformer", "bert")):
            modality = "text"
        elif any(token in text for token in ("audio", "spectrogram", "waveform")):
            modality = "audio"
        else:
            modality = "tabular"

    objective = str(supplied.get("objective") or supplied.get("task_type") or "").lower()
    if not objective:
        if any(token in text for token in ("regression", "regressor", "rmse", "mae", "r2_score", "forecast")):
            objective = "regression"
        elif any(token in text for token in ("ranking", "ndcg", "map@", "mean average precision")):
            objective = "ranking"
        elif any(token in text for token in ("survival", "concordance_index")):
            objective = "survival"
        else:
            objective = "classification"

    grouped = any(token in text for token in ("groupkfold", "stratifiedgroupkfold", "group_id", "patient_id"))
    temporal = any(token in text for token in ("timeseriessplit", "time series", "timestamp", "chronological"))
    split_family = "time_ordered" if temporal else "grouped" if grouped else "stratified" if objective == "classification" else "random"
    return {
        **supplied,
        "modality": modality,
        "objective": objective,
        "split_family": split_family,
        "grouped": grouped,
        "temporal": temporal,
    }


def infer_capabilities(code: str) -> dict:
    lower = (code or "").lower()
    model_calls = re.findall(
        r"\b(?:[A-Za-z_][A-Za-z0-9_]*(?:Classifier|Regressor|Regression|Model|Net)|"
        r"AutoModelFor[A-Za-z]+|XGBClassifier|XGBRegressor)\s*\(",
        code or "",
    )
    has_ensemble = bool(
        len(model_calls) >= 2
        or re.search(r"\b(?:ensemble|blend|stack|weighted[_ ]?average|voting)\b", lower)
    )
    has_early_stopping = bool(re.search(r"early_stopping|patience|eval_set", lower))
    has_selection = bool(
        has_ensemble
        or re.search(r"gridsearch|randomizedsearch|optuna|minimize\s*\(|best_(?:weight|param|score)", lower)
    )
    has_stateful_preprocessing = bool(
        re.search(r"(?:vectorizer|scaler|encoder|imputer|selector|pca|svd).*\.fit", lower)
        or re.search(r"\.fit_transform\s*\(", lower)
    )
    has_supervised_secondary_features = bool(
        re.search(r"(?:embedding|feature).*?(?:xgb|lightgbm|logistic|classifier|regressor)", lower, re.DOTALL)
    )
    return {
        "model_component_count": len(model_calls),
        "has_ensemble": has_ensemble,
        "has_early_stopping": has_early_stopping,
        "has_selection": has_selection,
        "has_stateful_preprocessing": has_stateful_preprocessing,
        "has_supervised_secondary_features": has_supervised_secondary_features,
    }


def build_protocol_plan(task_desc: str, code: str, task_profile: dict | None = None) -> dict:
    profile = infer_task_profile(task_desc, code, task_profile)
    capabilities = infer_capabilities(code)
    stages = ["data_scope"]
    if capabilities["has_early_stopping"]:
        stages.append("validation_provenance")
    if capabilities["has_ensemble"] or capabilities["has_supervised_secondary_features"]:
        stages.append("cross_fit")
    if capabilities["has_selection"]:
        stages.append("selection_freeze")
    stages.append("final_holdout")
    return {
        "schema": PROTOCOL_PLAN_SCHEMA,
        "task_profile": profile,
        "capabilities": capabilities,
        "stages": stages,
        "scope_contract": {
            "fit_scope": "outer_train_or_fold_train_only",
            "early_stopping_scope": "inner_validation_only",
            "selection_scope": "outer_train_oof_only",
            "report_scope": "untouched_outer_holdout_once",
            "test_scope": "final_prediction_only",
        },
        "split_contract": {
            "family": profile["split_family"],
            "classification_requires_stratification": profile["objective"] == "classification" and not profile["temporal"],
            "groups_must_not_cross_folds": profile["grouped"],
            "future_must_not_train_past": profile["temporal"],
        },
    }


def is_repairable_protocol_audit(audit: dict, code: str = "") -> bool:
    if not isinstance(audit, dict) or audit.get("status") == "clean":
        return False
    try:
        ast.parse(code or "")
    except SyntaxError:
        return False
    issues = [item for item in audit.get("issues", []) if isinstance(item, dict)]
    if not issues:
        return audit.get("status") == "protocol_biased"
    if any(item.get("category") == "repair_integrity" for item in issues):
        return False
    return any(
        str(item.get("issue_code")) in _REPAIRABLE_CODES
        or str(item.get("category")) in _REPAIRABLE_CATEGORIES
        for item in issues
    )


def ensure_transaction(agent: Any, node: Any) -> dict:
    existing = getattr(node, "protocol_repair", None) or {}
    if existing:
        return existing
    if not bool(_cfg(agent, "enabled", True)):
        return {}
    audit = getattr(node, "leakage_audit", None) or {}
    if not is_repairable_protocol_audit(audit, getattr(node, "code", "")):
        return {}
    contract = copy.deepcopy(
        (getattr(node, "leakage_repair_context", None) or {}).get("preservation_contract", {})
    ) or build_repair_preservation_contract(node.code)
    if contract.get("status") != "frozen":
        return {}
    plan = build_protocol_plan(
        getattr(agent, "task_desc", ""), node.code, getattr(node, "task_profile", None)
    )
    tx = {
        "schema": PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": uuid.uuid4().hex,
        "source_node_id": node.id,
        "source_code_sha256": code_sha256(node.code),
        "preservation_contract": contract,
        "protocol_plan": plan,
        "current_stage_index": 0,
        # Audit attempts count generated children that fail the current
        # protocol contract. Generation attempts count LLM/API failures before
        # a child exists. Neither counter consumes leakage_repair_attempt.
        "stage_attempts": {},
        "stage_generation_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": int(_cfg(agent, "per_stage_attempt_limit", 2)),
        "max_generation_attempts_per_stage": int(
            _cfg(agent, "stage_generation_attempt_limit", 2)
        ),
        "require_runtime_provenance": bool(_cfg(agent, "require_runtime_provenance", True)),
    }
    node.protocol_repair = tx
    audit["repair_mode"] = "staged_protocol_repair"
    audit["protocol_transaction_id"] = tx["transaction_id"]
    return tx


def current_stage(transaction: dict) -> str | None:
    plan = transaction.get("protocol_plan", {})
    stages = list(plan.get("stages") or [])
    index = int(transaction.get("current_stage_index", 0))
    return stages[index] if 0 <= index < len(stages) else None


def is_active(transaction: dict | None) -> bool:
    return bool(
        isinstance(transaction, dict)
        and transaction.get("schema") == PROTOCOL_REPAIR_SCHEMA
        and transaction.get("state") in {"pending", "stage_in_progress", "final_pending"}
        and current_stage(transaction)
    )


def begin_stage_generation(transaction: dict) -> dict:
    """Claim one bounded code-generation attempt for the current stage."""
    tx = copy.deepcopy(transaction)
    stage = current_stage(tx)
    if not stage or tx.get("state") not in {"pending", "final_pending"}:
        raise ValueError("Protocol repair stage is not available for generation")
    attempts = dict(tx.get("stage_generation_attempts") or {})
    attempts[stage] = int(attempts.get(stage, 0)) + 1
    limit = int(tx.get("max_generation_attempts_per_stage", 2))
    if attempts[stage] > limit:
        tx["state"] = "exhausted"
        tx["terminal_reason"] = f"stage_generation_attempts_exhausted:{stage}"
        return tx
    tx["stage_generation_attempts"] = attempts
    tx["state"] = "stage_in_progress"
    tx["active_stage"] = stage
    tx["active_generation_attempt"] = attempts[stage]
    return tx


def record_stage_generation_failure(
    transaction: dict,
    node_id: str,
    reason: str,
) -> dict:
    """Return a failed generation to its stage or exhaust only that stage."""
    tx = copy.deepcopy(transaction)
    stage = current_stage(tx) or tx.get("active_stage")
    attempt = int((tx.get("stage_generation_attempts") or {}).get(stage, 0))
    limit = int(tx.get("max_generation_attempts_per_stage", 2))
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": stage,
        "attempt": attempt,
        "status": "generation_failed",
        "reason": reason,
    })
    tx.pop("active_stage", None)
    tx.pop("active_generation_attempt", None)
    if attempt >= limit:
        tx["state"] = "exhausted"
        tx["terminal_reason"] = f"stage_generation_attempts_exhausted:{stage}"
    else:
        tx["state"] = "final_pending" if stage == "final_holdout" else "pending"
    return tx


def finish_stage_generation(transaction: dict) -> dict:
    """Clear the transient generation lease carried into the child."""
    tx = copy.deepcopy(transaction)
    tx.pop("active_stage", None)
    tx.pop("active_generation_attempt", None)
    if tx.get("state") == "stage_in_progress":
        tx["state"] = (
            "final_pending" if current_stage(tx) == "final_holdout" else "pending"
        )
    return tx


def stage_scope_gate(audit: dict, transaction: dict) -> dict:
    """Require current/earlier protocol issues to be clean, defer only later ones."""
    stage = current_stage(transaction)
    stages = list(transaction.get("protocol_plan", {}).get("stages") or [])
    current_index = stages.index(stage) if stage in stages else len(stages)
    blocking = []
    deferred = []
    for issue in audit.get("issues", []):
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("issue_code") or "")
        category = str(issue.get("category") or "")
        assigned = _ISSUE_STAGE.get(code)
        if category == "repair_integrity":
            blocking.append(issue)
            continue
        if assigned is None:
            # Unknown hard/static failures are never deferred by convention.
            blocking.append(issue)
            continue
        assigned_index = stages.index(assigned) if assigned in stages else len(stages) - 1
        (blocking if assigned_index <= current_index else deferred).append(issue)
    return {
        "status": "clean" if not blocking else "blocked",
        "stage": stage,
        "blocking_issue_codes": [item.get("issue_code") for item in blocking],
        "deferred_issue_codes": [item.get("issue_code") for item in deferred],
    }


def stage_instructions(transaction: dict) -> list[str]:
    stage = current_stage(transaction)
    plan = transaction.get("protocol_plan", {})
    profile = plan.get("task_profile", {})
    common = [
        "Keep every protected model, feature branch, ensemble member, checkpoint, optimizer, loss, and training hyperparameter unchanged.",
        "Modify only split/index propagation, fit/transform scope, early-stopping scope, cross-fitting, selection scope, and final reporting.",
        "Use immutable sample_id values end-to-end; never recover sample identity by slicing arrays by length.",
        "Introduce and maintain ProtocolProvenanceGuard records cumulatively as each scope becomes available.",
        f"Respect split_family={profile.get('split_family')} for modality={profile.get('modality')} objective={profile.get('objective')}.",
        "Return the complete runnable program, not a patch and not pseudocode.",
        "Use these canonical names so the deterministic gate can verify the protocol: sample_ids, outer_train_ids, outer_holdout_ids, inner_train_ids, inner_valid_ids, and oof_predictions.",
    ]
    specific = {
        "data_scope": [
            "Create an explicit outer_train/outer_holdout split before fitting any learned preprocessing or model-selection component.",
            "Carry sample_id through all dense, sparse, image/audio, and label arrays; fit preprocessing on outer_train or fold_train only.",
            "Do not inspect test rows while learning vocabulary, normalization, feature selection, augmentation statistics, or encodings.",
        ],
        "validation_provenance": [
            "Inside each outer-training fold, create a separate inner_train/inner_validation scope for early stopping.",
            "The row receiving an OOF prediction must not participate in model fitting, preprocessing fitting, or early stopping for that prediction.",
        ],
        "cross_fit": [
            "Generate complete OOF predictions/features for every outer_train row using only fold models that did not train or early-stop on that row.",
            "Align every model's OOF output by sample_id, verify exactly-once coverage, and preserve class/output ordering.",
        ],
        "selection_freeze": [
            "Search ensemble weights/hyperparameters only on outer_train OOF predictions, then freeze the selected values before opening outer_holdout.",
            "Do not use outer_holdout or test metrics to revise weights, epochs, thresholds, architecture, or features.",
        ],
        "final_holdout": [
            "Train the frozen design on outer_train, evaluate outer_holdout exactly once, and never tune after that result.",
            "Instrument the protocol with agents.protocol_repair_runtime.ProtocolProvenanceGuard and call emit() after assert_clean().",
            "Record partitions, every learned fit scope, OOF/final prediction scopes, selection scopes, freeze, and final evaluation.",
        ],
    }
    return common + specific.get(str(stage), [])


def _stage_issue(stage: str | None, evidence: str, remediation: str) -> dict:
    return {
        "issue_code": f"PROTOCOL_STAGE_{str(stage or 'UNKNOWN').upper()}_INCOMPLETE",
        "category": "protocol_repair_stage",
        "severity": "high",
        "line": 0,
        "evidence": evidence,
        "remediation": remediation,
        "execution_disposition": "block",
        "detector": "protocol_stage_v1",
    }


def audit_stage(code: str, transaction: dict) -> dict:
    stage = current_stage(transaction)
    plan = transaction.get("protocol_plan", {})
    capabilities = plan.get("capabilities", {})
    profile = plan.get("task_profile", {})
    failures: list[str] = []
    facts = _ast_facts(code)
    calls = facts["calls"]
    if not facts["valid"]:
        failures.append("program is not valid Python")

    if stage == "data_scope":
        if not _contains(code, r"\b(?:outer_)?(?:x_)?train(?:_ids|_idx|_indices)?\b") or not _contains(code, r"\b(?:outer_)?(?:x_)?(?:holdout|test|eval)(?:_ids|_idx|_indices)?\b"):
            failures.append("explicit outer_train and outer_holdout partitions are missing")
        if not _contains(code, r"\b(?:(?:sample|row)_?ids?|indices|all_idx|all_indices)\b"):
            failures.append("stable sample_id/row_id propagation is missing")
        split_calls = {
            "train_test_split", "StratifiedKFold", "StratifiedGroupKFold",
            "GroupKFold", "TimeSeriesSplit", "KFold", "ShuffleSplit",
            "StratifiedShuffleSplit", "GroupShuffleSplit",
        }
        if not (calls & split_calls):
            failures.append("an explicit split primitive is missing")
        if profile.get("temporal") and not _contains(code, r"TimeSeriesSplit|chronolog|sort_values"):
            failures.append("time-ordered task lacks chronological splitting")
        if profile.get("grouped") and not _contains(code, r"GroupKFold|StratifiedGroupKFold|groups\s*="):
            failures.append("grouped task lacks group-isolated splitting")
    elif stage == "validation_provenance":
        if capabilities.get("has_early_stopping") and not _contains(code, r"inner_(?:train|fit).{0,200}inner_(?:val|valid|dev)"):
            failures.append("early stopping is not isolated in an inner split")
        if (
            (capabilities.get("has_ensemble") or capabilities.get("has_supervised_secondary_features"))
            and not _contains(code, r"\boof\w*")
        ):
            failures.append("OOF prediction provenance is missing")
    elif stage == "cross_fit":
        if not (calls & {"StratifiedKFold", "StratifiedGroupKFold", "GroupKFold", "TimeSeriesSplit", "KFold"}):
            failures.append("cross-fitting splitter is missing")
        if not _contains(code, r"\b(?:oof|out_of_fold)\w*\s*\["):
            failures.append("OOF predictions/features are not assigned by validation indices")
        if "record_prediction" not in calls:
            failures.append("prediction provenance is not recorded")
    elif stage == "selection_freeze":
        if not _contains(code, r"(?:best|selected|frozen|final)_?(?:weights?|params?|threshold|config|blend|ratio|coefficients?)"):
            failures.append("selected ensemble/tuning state is not materialized")
        if not _contains(code, r"(?:oof\w*).{0,500}(?:best|selected|optimi|minimize|search)") and not _contains(code, r"(?:best|selected|optimi|minimize|search).{0,500}(?:oof\w*)"):
            failures.append("selection is not demonstrably based on OOF data")
        if "record_selection" not in calls or "freeze" not in calls:
            failures.append("selection scope/freeze provenance is missing")
    elif stage == "final_holdout":
        required = {
            "ProtocolProvenanceGuard": "ProtocolProvenanceGuard",
            "partition registration": "register_partition",
            "fit provenance": "record_fit",
            "prediction provenance": "record_prediction",
            "protocol freeze": "freeze",
            "final evaluation provenance": "record_final_evaluation",
            "runtime assertion": "assert_clean",
            "runtime evidence emission": "emit",
        }
        for label, call_name in required.items():
            if call_name not in calls:
                failures.append(f"{label} is missing")

    return {
        "schema": PROTOCOL_STAGE_AUDIT_SCHEMA,
        "stage": stage,
        "status": "clean" if not failures else "blocked",
        "code_sha256": code_sha256(code),
        "issues": [
            _stage_issue(stage, failure, f"Complete the {stage} contract without changing protected model design.")
            for failure in failures
        ],
    }


def apply_stage_result(transaction: dict, stage_audit: dict, node_id: str) -> dict:
    tx = copy.deepcopy(transaction)
    stage = current_stage(tx)
    attempts = dict(tx.get("stage_attempts") or {})
    attempts[stage] = int(attempts.get(stage, 0)) + 1
    tx["stage_attempts"] = attempts
    passed = stage_audit.get("status") == "clean"
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": stage,
        "attempt": attempts[stage],
        "status": "passed" if passed else "failed",
        "code_sha256": stage_audit.get("code_sha256"),
        "issue_codes": [item.get("issue_code") for item in stage_audit.get("issues", [])],
    })
    if passed:
        tx["current_stage_index"] = int(tx.get("current_stage_index", 0)) + 1
        tx["state"] = "final_pending" if current_stage(tx) == "final_holdout" else "pending"
        if current_stage(tx) is None:
            tx["state"] = "ready_for_execution"
    elif attempts[stage] >= int(tx.get("max_attempts_per_stage", 2)):
        tx["state"] = "exhausted"
        tx["terminal_reason"] = f"stage_attempts_exhausted:{stage}"
    else:
        tx["state"] = "pending"
    return tx


def rollback_final_runtime_failure(transaction: dict, node_id: str, reason: str) -> dict:
    """Return an executed/crashed final node to the final stage, within budget."""
    tx = copy.deepcopy(transaction)
    stages = list(tx.get("protocol_plan", {}).get("stages") or [])
    attempts = int(tx.get("stage_attempts", {}).get("final_holdout", 0))
    tx["current_stage_index"] = max(0, len(stages) - 1)
    tx["state"] = (
        "exhausted"
        if attempts >= int(tx.get("max_attempts_per_stage", 2))
        else "pending"
    )
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": "final_holdout_runtime",
        "attempt": attempts,
        "status": "failed",
        "reason": reason,
    })
    return tx


def runtime_provenance_audit(term_out: str, transaction: dict) -> dict:
    matches = re.findall(rf"(?m)^{re.escape(RUNTIME_MARKER)}(\{{.*\}})$", term_out or "")
    if not matches:
        return {"status": "blocked", "reason": "runtime provenance marker missing"}
    import json
    try:
        payload = json.loads(matches[-1])
    except Exception as exc:
        return {"status": "blocked", "reason": f"runtime provenance is invalid JSON: {exc}"}
    if payload.get("schema") != "mlevolve_protocol_provenance_v1":
        return {"status": "blocked", "reason": "runtime provenance schema mismatch"}
    if payload.get("status") != "clean" or payload.get("violations"):
        return {"status": "blocked", "reason": "; ".join(payload.get("violations") or ["runtime guard rejected protocol"])}
    required_counts = ("partitions", "fits", "predictions", "selections", "final_evaluations")
    if any(int(payload.get("counts", {}).get(key, 0)) <= 0 for key in required_counts):
        return {"status": "blocked", "reason": "runtime provenance is incomplete"}
    return {"status": "clean", "payload_sha256": hashlib.sha256(matches[-1].encode()).hexdigest(), "counts": payload.get("counts", {})}


def runtime_result_as_audit(code: str, result: dict) -> dict:
    """Convert runtime guard evidence into the ordinary leakage-audit shape."""
    if result.get("status") == "clean":
        return {
            "issues": [],
            "detector_status": "complete",
            "runtime_protocol_provenance": result,
        }
    return {
        "issues": [{
            "issue_code": "PROTOCOL_RUNTIME_PROVENANCE_FAILED",
            "category": "protocol_repair_runtime",
            "severity": "critical",
            "line": 0,
            "evidence": str(result.get("reason") or "runtime protocol evidence was not clean"),
            "remediation": "Record real split/fit/prediction/selection/final-evaluation scopes and rerun the final protocol stage.",
            "execution_disposition": "block",
            "detector": "protocol_runtime_v1",
        }],
        "detector_status": "complete",
        "runtime_protocol_provenance": result,
    }
