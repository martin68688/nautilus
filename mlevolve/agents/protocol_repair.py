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

_STATEFUL_PREPROCESSOR_CONSTRUCTORS = {
    "ColumnTransformer",
    "CountVectorizer",
    "DictVectorizer",
    "KBinsDiscretizer",
    "KNNImputer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PCA",
    "Pipeline",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "SelectKBest",
    "SelectPercentile",
    "SimpleImputer",
    "StandardScaler",
    "TfidfTransformer",
    "TfidfVectorizer",
    "TruncatedSVD",
    "VarianceThreshold",
}
_STATEFUL_PREPROCESSOR_NAME = re.compile(
    r"(?:vectorizer|scaler|encoder|imputer|selector|preprocess|transformer|pca|svd|pipeline)",
    re.IGNORECASE,
)


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


def _call_line(tree: ast.AST, call_name: str) -> int | None:
    lines = [
        int(node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _ast_call_name(node) == call_name
    ]
    return min(lines) if lines else None


def _ast_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _root_name(node: ast.AST) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _canonical_partition_failures(tree: ast.AST) -> list[str]:
    """Validate the real runtime signature before an expensive final run."""
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "register_partition"
    ]
    if not calls:
        return []
    names: set[str] = set()
    failures: list[str] = []
    for call in calls:
        if len(call.args) != 2:
            failures.append("register_partition must use exactly (partition_name, sample_ids)")
            continue
        name = call.args[0]
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            # Fold names are commonly expressed as f"fold_{fold}_train".  The
            # runtime accepts those dynamic names; only the two canonical
            # outer partitions must be stable string literals.
            continue
        names.add(name.value)
    if calls and not {"outer_train", "outer_holdout"}.issubset(names):
        failures.append("canonical outer_train and outer_holdout partition registrations are missing")
    return failures


_DEFAULT_STAGE_ATTEMPT_LIMITS = {
    "data_scope": 2,
    "validation_provenance": 2,
    "cross_fit": 5,
    "selection_freeze": 4,
    "final_holdout": 8,
}

_DEFAULT_FINAL_RUNTIME_ATTEMPT_LIMIT = 4


def _stage_attempt_limit(transaction: dict, stage: str | None, *, generation: bool = False) -> int:
    """Return a bounded stage-specific budget with legacy compatibility."""
    key = "stage_generation_attempt_limits" if generation else "stage_attempt_limits"
    limits = transaction.get(key) or {}
    if stage in limits:
        return max(1, int(limits[stage]))
    legacy_key = "max_generation_attempts_per_stage" if generation else "max_attempts_per_stage"
    if legacy_key in transaction:
        return max(1, int(transaction[legacy_key]))
    return int(_DEFAULT_STAGE_ATTEMPT_LIMITS.get(str(stage), 2))


def _ancestor(node: ast.AST, parent: dict[ast.AST, ast.AST], kinds: tuple[type, ...]):
    current = parent.get(node)
    while current is not None:
        if isinstance(current, kinds):
            return current
        current = parent.get(current)
    return None


def _cross_fit_scope_failures(
    tree: ast.AST,
    *,
    outer_holdout_allowed_after: int | None = None,
) -> list[str]:
    """Keep the fixed outer holdout outside the complete OOF transaction.

    Generated repairs are complete-program rewrites.  A later stage therefore
    has to retain the canonical outer split established by ``data_scope``;
    otherwise a model can rename each CV validation fold to ``outer_holdout``
    and satisfy isolated stage checks without preserving a final holdout.
    """
    parent = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    failures = list(_canonical_partition_failures(tree))

    canonical_calls: dict[str, list[ast.Call]] = {"outer_train": [], "outer_holdout": []}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or _ast_call_name(call) != "register_partition":
            continue
        if len(call.args) != 2:
            continue
        name = call.args[0]
        if isinstance(name, ast.Constant) and name.value in canonical_calls:
            canonical_calls[str(name.value)].append(call)

    for name, calls in canonical_calls.items():
        expected_ids = f"{name}_ids"
        for call in calls:
            if _ancestor(call, parent, (ast.For, ast.AsyncFor, ast.While)) is not None:
                failures.append(f"canonical {name} partition is registered inside a fold loop")
            if len(call.args) >= 2 and not (
                isinstance(call.args[1], ast.Name) and call.args[1].id == expected_ids
            ):
                failures.append(
                    f"canonical {name} partition must use the stable {expected_ids} variable"
                )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if {"outer_train_ids", "outer_holdout_ids"} & _assigned_names(node):
            if _ancestor(node, parent, (ast.For, ast.AsyncFor, ast.While)) is not None:
                failures.append("fixed outer partition IDs are reassigned inside a fold loop")

    global_oof_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _ast_call_name(node) == "record_global_oof"
    ]
    for call in global_oof_calls:
        if len(call.args) < 2 or not (
            isinstance(call.args[1], ast.Name) and call.args[1].id == "outer_train_ids"
        ):
            failures.append("global OOF coverage must be recorded against outer_train_ids")

    splitter_types = {
        "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold",
        "TimeSeriesSplit", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit",
    }
    splitter_vars = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Call)
        and _ast_call_name(node.value) in splitter_types
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    splitter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and _root_name(node.func.value) in splitter_vars
        and node.args
    ]
    for call in splitter_calls:
        input_names = {
            child.id.lower()
            for child in ast.walk(call.args[0])
            if isinstance(child, ast.Name)
        }
        if "sample_ids" in input_names or any(
            "outer_holdout" in name or name.startswith("test_")
            for name in input_names
        ):
            failures.append("cross-fit splitter consumes rows outside outer_train")

    forbidden_call_names = {
        "evaluate", "eval", "fit", "fit_transform", "partial_fit", "predict",
        "predict_proba", "score", "transform",
    }
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        call_name = _ast_call_name(call).lower()
        if call_name not in forbidden_call_names and not any(
            token in call_name for token in ("evaluate", "metric", "loss")
        ):
            continue
        loaded = {
            node.id.lower()
            for arg in [*call.args, *(keyword.value for keyword in call.keywords)]
            for node in ast.walk(arg)
            if isinstance(node, ast.Name)
        }
        if any("outer_holdout" in name for name in loaded) and (
            outer_holdout_allowed_after is None
            or int(getattr(call, "lineno", 0) or 0) <= outer_holdout_allowed_after
        ):
            failures.append(
                f"outer_holdout is consumed by {call_name or 'a learned operation'} before protocol freeze"
            )

    return list(dict.fromkeys(failures))


def _cross_fit_preprocessor_failures(tree: ast.AST) -> list[str]:
    """Reject OOF pipelines whose learned preprocessing saw validation rows.

    Only direct, module-level fit calls are classified here.  Helper functions
    remain available for arbitrary modalities, while the runtime provenance
    contract still applies to their callers.
    """
    parent = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    constructors: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not isinstance(node.value, ast.Call):
            continue
        constructor = _ast_call_name(node.value)
        for target in targets:
            if isinstance(target, ast.Name):
                constructors[target.id] = constructor

    global_oof_line = _call_line(tree, "record_global_oof")
    if global_oof_line is None:
        return []

    record_fit_labels = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _ast_call_name(node) == "record_fit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    failures: list[str] = []
    checked_receivers: set[str] = set()
    for call in ast.walk(tree):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr not in {"fit", "fit_transform", "partial_fit"}
            or int(getattr(call, "lineno", 0) or 0) >= global_oof_line
        ):
            continue
        receiver = _root_name(call.func.value)
        constructor = constructors.get(receiver, "")
        if constructor in {"LabelBinarizer", "LabelEncoder"}:
            continue
        if not (
            constructor in _STATEFUL_PREPROCESSOR_CONSTRUCTORS
            or _STATEFUL_PREPROCESSOR_NAME.search(receiver)
        ):
            continue

        current = parent.get(call)
        fold_loop = None
        inside_function = False
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inside_function = True
                break
            if isinstance(current, (ast.For, ast.AsyncFor)):
                iter_text = ast.unparse(current.iter).lower()
                target_text = ast.unparse(current.target).lower()
                if ".split(" in iter_text or (
                    "fold" in target_text
                    and any(token in target_text for token in ("train", "valid", "val"))
                ):
                    fold_loop = current
                    break
            current = parent.get(current)
        if inside_function:
            continue
        if fold_loop is None:
            failures.append(
                f"learned preprocessor {receiver or constructor} is fitted before complete OOF outside the fold loop"
            )
            continue
        if receiver and receiver not in checked_receivers:
            checked_receivers.add(receiver)
            if receiver not in record_fit_labels:
                failures.append(
                    f"fold-local preprocessor {receiver} lacks record_fit provenance using that exact component label"
                )
    return failures


def _assigned_names(node: ast.AST) -> set[str]:
    names = set()
    targets = []
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


_SELECTED_STATE_NAME = re.compile(
    r"(?:best|selected|optimized|frozen).*"
    r"(?:weights?|params?|hyperparameters?|threshold|config|state|blend|ratio|coefficients?)",
    re.IGNORECASE,
)


def _selection_state_lines(tree: ast.AST) -> list[tuple[int, str]]:
    return [
        (int(node.lineno), name)
        for node in ast.walk(tree)
        for name in _assigned_names(node)
        if _SELECTED_STATE_NAME.search(name)
    ]


def _oof_selection_analysis(tree: ast.AST, start: int, end: int) -> dict:
    """Trace OOF data through a metric into the selected/frozen state."""
    objective_tokens = ("loss", "score", "metric", "objective", "evaluate", "minimize", "search", "optimi")
    nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if hasattr(node, "lineno") and start < int(node.lineno) < end
        ),
        key=lambda node: (int(node.lineno), int(getattr(node, "col_offset", 0))),
    )
    oof_derived = {
        node.id
        for node in nodes
        if isinstance(node, ast.Name) and "oof" in node.id.lower()
    }
    metric_results: set[str] = set()
    causal_selection: set[str] = set()
    parent = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }

    def loaded_names(node: ast.AST | None) -> set[str]:
        return {
            child.id
            for child in (ast.walk(node) if node is not None else ())
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    def objective_uses_oof(node: ast.AST | None) -> bool:
        return any(
            isinstance(child, ast.Call)
            and any(token in _ast_call_name(child).lower() for token in objective_tokens)
            and bool(loaded_names(child) & oof_derived)
            for child in (ast.walk(node) if node is not None else ())
        )

    def objective_assignment_uses_oof(targets: set[str], node: ast.AST | None) -> bool:
        """Recognize explicit metric formulas as well as metric helper calls."""
        return bool(
            node is not None
            and any(token in name.lower() for name in targets for token in objective_tokens)
            and loaded_names(node) & oof_derived
        )

    def controlling_if_uses_metric(node: ast.AST) -> bool:
        current = parent.get(node)
        while current is not None:
            if isinstance(current, ast.If) and loaded_names(current.test) & metric_results:
                return True
            current = parent.get(current)
        return False

    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            targets = _assigned_names(node)
            loaded = loaded_names(value)
            if loaded & oof_derived:
                oof_derived.update(targets)
            metric_call = objective_uses_oof(value)
            metric_formula = objective_assignment_uses_oof(targets, value)
            if metric_call or metric_formula:
                metric_results.update(targets)
            selected_targets = {name for name in targets if _SELECTED_STATE_NAME.search(name)}
            if selected_targets and (
                metric_call
                or metric_formula
                or bool(loaded & causal_selection)
                or controlling_if_uses_metric(node)
            ):
                causal_selection.update(selected_targets)
    return {
        "has_metric": bool(metric_results),
        "causal_selection": sorted(causal_selection),
    }


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
        if any(token in text for token in (
            "classification", "classifier", "logisticregression", "crossentropyloss",
            "log_loss", "logloss", "num_labels", "class_label",
        )):
            objective = "classification"
        elif any(token in text for token in ("ranking", "ndcg", "map@", "mean average precision")):
            objective = "ranking"
        elif any(token in text for token in ("survival", "concordance_index")):
            objective = "survival"
        elif any(token in text for token in ("regression", "regressor", "rmse", "mae", "r2_score", "forecast")):
            objective = "regression"
        else:
            objective = "unknown"

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
    has_selection = bool(re.search(
        r"gridsearch|randomizedsearch|optuna|minimize\s*\(|"
        r"(?:best|selected|optimized)[_a-z0-9]*(?:weight|param|hyperparam|threshold|config|blend|ratio|coefficient)",
        lower,
    ))
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
    supported_objectives = {"classification", "regression", "ranking", "survival"}
    if profile.get("objective") not in supported_objectives:
        raise ValueError(
            f"unsupported protocol objective: {profile.get('objective') or 'unknown'}"
        )
    capabilities = infer_capabilities(code)
    stages = ["data_scope"]
    if capabilities["has_early_stopping"]:
        stages.append("validation_provenance")
    if (
        capabilities["has_ensemble"]
        or capabilities["has_supervised_secondary_features"]
        or capabilities["has_selection"]
    ):
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
    try:
        plan = build_protocol_plan(
            getattr(agent, "task_desc", ""), node.code, getattr(node, "task_profile", None)
        )
    except ValueError as exc:
        audit["status"] = "blocked"
        audit["hard_block"] = True
        audit["rank_eligible"] = False
        audit["metric_disposition"] = "invalid"
        audit["memory_disposition"] = "negative_only"
        audit["repair_mode"] = "unsupported_protocol"
        issues = list(audit.get("issues") or [])
        issues.append({
            "issue_code": "UNSUPPORTED_PROTOCOL_PROFILE",
            "category": "protocol_profile",
            "severity": "high",
            "line": 0,
            "evidence": str(exc),
            "remediation": "Provide an explicit supported task_profile or implement a dedicated evaluation protocol.",
            "execution_disposition": "block",
            "detector": "protocol_profile_v1",
        })
        audit["issues"] = issues
        node.leakage_audit = audit
        return {}
    configured_limit = int(_cfg(agent, "per_stage_attempt_limit", 2))
    configured_generation_limit = int(_cfg(agent, "stage_generation_attempt_limit", 2))
    configured_stage_limits = dict(_cfg(agent, "stage_attempt_limits", {}) or {})
    configured_generation_limits = dict(
        _cfg(agent, "stage_generation_attempt_limits", {}) or {}
    )
    stage_attempt_limits = {
        stage: max(
            configured_limit,
            int(configured_stage_limits.get(stage, configured_limit)),
            _DEFAULT_STAGE_ATTEMPT_LIMITS.get(stage, configured_limit),
        )
        for stage in plan.get("stages", [])
    }
    stage_generation_attempt_limits = {
        stage: max(
            configured_generation_limit,
            int(configured_generation_limits.get(
                stage,
                configured_generation_limit,
            )),
            _DEFAULT_STAGE_ATTEMPT_LIMITS.get(stage, configured_generation_limit),
        )
        for stage in plan.get("stages", [])
    }
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
        "stage_generation_failures": {},
        "stage_runtime_attempts": {},
        "history": [],
        "state": "pending",
        # Keep scalar fields so old journals remain readable. New transactions
        # use the maps, allowing complex stages to retry without weakening the
        # budgets of simpler stages.
        "max_attempts_per_stage": configured_limit,
        "max_generation_attempts_per_stage": configured_generation_limit,
        "stage_attempt_limits": stage_attempt_limits,
        "stage_generation_attempt_limits": stage_generation_attempt_limits,
        "final_runtime_attempt_limit": max(
            1,
            int(_cfg(
                agent,
                "final_runtime_attempt_limit",
                _DEFAULT_FINAL_RUNTIME_ATTEMPT_LIMIT,
            )),
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
    failures = dict(tx.get("stage_generation_failures") or {})
    failures[stage] = int(failures.get(stage, 0)) + 1
    tx["stage_generation_failures"] = failures
    limit = _stage_attempt_limit(tx, stage, generation=True)
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": stage,
        "attempt": attempt,
        "status": "generation_failed",
        "reason": reason,
        "feedback": [{
            "issue_code": "PROTOCOL_STAGE_GENERATION_FAILED",
            "evidence": reason,
            "remediation": "Return one complete runnable Python program for the same stage.",
        }],
    })
    tx.pop("active_stage", None)
    tx.pop("active_generation_attempt", None)
    if failures[stage] >= limit:
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
    contract = transaction.get("preservation_contract", {})
    protected_components = sorted((contract.get("component_calls") or {}).keys())
    component_summary = ", ".join(protected_components) if protected_components else "all learned components in the source program"
    common = [
        "Keep every protected model, feature branch, ensemble member, checkpoint, optimizer, loss, and training hyperparameter unchanged.",
        "Modify only split/index propagation, fit/transform scope, early-stopping scope, cross-fitting, selection scope, and final reporting.",
        "Use immutable sample_id values end-to-end; never recover sample identity by slicing arrays by length.",
        "Introduce and maintain ProtocolProvenanceGuard records cumulatively as each scope becomes available.",
        f"Respect split_family={profile.get('split_family')} for modality={profile.get('modality')} objective={profile.get('objective')}.",
        f"The protected learned components for this transaction are: {component_summary}.",
        "Return the complete runnable program, not a patch and not pseudocode.",
        "Use these canonical names so the deterministic gate can verify the protocol: sample_ids, outer_train_ids, outer_holdout_ids, inner_train_ids, inner_valid_ids, and oof_predictions.",
    ]
    specific = {
        "data_scope": [
            "Create an explicit outer_train/outer_holdout split before fitting any learned preprocessing or model-selection component.",
            "Carry sample_id through all dense, sparse, image/audio, and label arrays; fit preprocessing on outer_train or fold_train only.",
            "Do not inspect test rows while learning vocabulary, normalization, feature selection, augmentation statistics, or encodings.",
            "Use the exact runtime signature `protocol_guard.register_partition(\"outer_train\", outer_train_ids)` and likewise for `outer_holdout`; the partition name is always the first argument.",
        ],
        "validation_provenance": [
            "Inside each outer-training fold, create a separate inner_train/inner_validation scope for early stopping.",
            "The row receiving an OOF prediction must not participate in model fitting, preprocessing fitting, or early stopping for that prediction.",
            "Keep explicit `inner_train_ids` and `inner_valid_ids` assignments together in the fold loop; do not rename them to generic train/val arrays.",
        ],
        "cross_fit": [
            "Generate complete OOF predictions/features for every outer_train row using only fold models that did not train or early-stop on that row.",
            "Retain the one canonical outer_train/outer_holdout split created by data_scope. Never relabel each CV fold as a new outer split, and never reassign outer_train_ids or outer_holdout_ids inside a loop.",
            "Run the K-fold splitter on outer_train_ids only. During cross_fit, outer_holdout may be registered but must not be transformed, predicted, evaluated, scored, or otherwise inspected.",
            "Emit raw per-component OOF predictions. Do not tune fold-specific ensemble weights, hyperparameters, thresholds, calibration, or feature choices on inner_valid labels; defer every such choice until selection_freeze has complete global OOF evidence.",
            "Align every model's OOF output by sample_id, verify exactly-once coverage, and preserve class/output ordering.",
            "Use the real runtime API, not a dict or custom imitation: `from agents.protocol_repair_runtime import ProtocolProvenanceGuard`; create `protocol_guard = ProtocolProvenanceGuard()` and register outer_train/outer_holdout partitions.",
            "For every fold and every protected learned predictor, call `protocol_guard.record_prediction(component, inner_train_ids, inner_valid_ids, purpose=\"oof\")` immediately after assigning predictions to `oof_predictions[inner_valid_indices]`.",
            "After verifying exactly-once coverage, call `protocol_guard.record_global_oof(oof_predictions, outer_train_ids)` with the complete aligned OOF matrix; this runtime coverage record is mandatory for selection_freeze.",
            "Also call `protocol_guard.record_fit(component, inner_train_ids, purpose=...)` for each learned preprocessor and model; keep `protocol_guard` alive for the later selection_freeze and final_holdout stages.",
            "Instantiate and fit every learned vectorizer, scaler, encoder, imputer, selector, dimensionality reducer, augmentation-statistics object, and preprocessing pipeline inside the fold loop before producing that fold's OOF rows. Never fit one shared preprocessor on all outer_train rows before cross-fitting.",
            "For each fold-local learned preprocessor, use its exact Python variable name as the stable first argument to `protocol_guard.record_fit`; for example `protocol_guard.record_fit(\"word_vectorizer\", inner_train_ids, purpose=\"fold_preprocess\")`.",
            "Use one stable string literal for each component in both `record_fit` and `record_prediction` (for example `deberta`, `xgboost`, or `logistic_regression`); never append the fold number to the component name. Put fold identity in `purpose` or partition names instead.",
            "The fold's `inner_valid_ids` are prediction-only OOF rows: they must never appear in any model `.fit(...)`, `eval_set`, preprocessing fit, or early-stopping scope.",
            "If any protected component needs early stopping, split only `inner_train_ids` again into `early_stop_train_ids` and `early_stop_eval_ids`; fit on the former, early-stop on the latter, then predict exactly once on `inner_valid_ids`.",
        ],
        "selection_freeze": [
            "Search any ensemble weights, hyperparameters, thresholds, feature choices, or calibration state only on outer_train OOF predictions, then freeze the selected values before opening outer_holdout.",
            "Compute the task metric from candidates derived from the complete raw OOF predictions, update the selected state from those metric values, and never substitute historical/fold-local weights or a no-op search loop.",
            "Do not use outer_holdout or test metrics to revise weights, epochs, thresholds, architecture, or features.",
            "Use the existing real guard object and call `protocol_guard.record_selection(\"protocol_state\", outer_train_ids)` immediately after OOF-only selection, materialize the selected state in a `frozen_protocol_state` value, then call `protocol_guard.freeze()`.",
            "Do not replace `protocol_guard` with a dict, custom class, print statement, or renamed method; the exact `record_selection` and `freeze` calls are mandatory.",
        ],
        "final_holdout": [
            "Train the frozen design on outer_train, evaluate outer_holdout exactly once, and never tune after that result.",
            "The protocol must already be frozen before the first operation that consumes outer_holdout features. Put `protocol_guard.freeze()` before every outer-holdout feature extraction, tokenizer call, transform, prediction, or metric call; merely delaying the metric is not enough.",
            "Generate final holdout predictions directly from the actual outer_holdout feature rows, with the same sample_id order as the outer_holdout labels used by the metric.",
            "Keep outer_holdout predictions separate from external test/submission predictions. Never slice, rename, or reuse test predictions as holdout predictions.",
            "Instrument the protocol with agents.protocol_repair_runtime.ProtocolProvenanceGuard and call emit() after assert_clean().",
            "Record partitions, every learned fit scope, OOF/final prediction scopes, selection scopes, freeze, and final evaluation.",
            "If no earlier selection_freeze stage exists, record the already-fixed design with `protocol_guard.record_selection(\"fixed_protocol_state\", outer_train_ids)` and then call `protocol_guard.freeze()` before final evaluation.",
            "Use the exact final calls `protocol_guard.record_prediction(\"final_predictor\", outer_train_ids, outer_holdout_ids, purpose=\"final\")`, `protocol_guard.record_final_evaluation(outer_holdout_ids)`, `protocol_guard.assert_clean()`, and `protocol_guard.emit()`; do not imitate or rename them.",
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
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    if not facts["valid"]:
        failures.append("program is not valid Python")

    stages = list(plan.get("stages") or [])
    current_index = int(transaction.get("current_stage_index", 0))
    active_stages = stages[: current_index + 1]
    if "cross_fit" in active_stages:
        failures.extend(_cross_fit_scope_failures(
            tree,
            outer_holdout_allowed_after=(
                _call_line(tree, "freeze") if stage == "final_holdout" else None
            ),
        ))

    if stage == "data_scope":
        if not _contains(code, r"\b(?:outer_)?(?:x_)?train(?:_ids|_idx|_indices)?\b") or not _contains(code, r"\b(?:outer_)?(?:x_)?(?:holdout|test|eval)(?:_ids|_idx|_indices)?\b"):
            failures.append("explicit outer_train and outer_holdout partitions are missing")
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
        failures.extend(_canonical_partition_failures(tree))
    elif stage == "validation_provenance":
        if capabilities.get("has_early_stopping") and not _contains(code, r"inner_(?:train|fit).{0,200}inner_(?:val|valid|dev)"):
            failures.append("early stopping is not isolated in an inner split")
    elif stage == "cross_fit":
        if not (calls & {"StratifiedKFold", "StratifiedGroupKFold", "GroupKFold", "TimeSeriesSplit", "KFold"}):
            failures.append("cross-fitting splitter is missing")
        if not _contains(code, r"\b(?:oof|out_of_fold)\w*\s*\["):
            failures.append("OOF predictions/features are not assigned by validation indices")
        if "record_prediction" not in calls:
            failures.append("prediction provenance is not recorded")
        if "record_fit" not in calls:
            failures.append("fit provenance is not recorded")
        if "record_global_oof" not in calls:
            failures.append("global OOF coverage provenance is not recorded")
        else:
            global_oof_line = _call_line(tree, "record_global_oof")
            premature_selection = [
                name
                for line, name in _selection_state_lines(tree)
                if global_oof_line is not None and line < global_oof_line
            ]
            if premature_selection:
                failures.append(
                    "selection state is learned before complete global OOF exists: "
                    + ", ".join(sorted(set(premature_selection)))
                )
        if capabilities.get("has_stateful_preprocessing"):
            failures.extend(_cross_fit_preprocessor_failures(tree))
    elif stage == "selection_freeze":
        if not _contains(
            code,
            r"\b(?:best|selected|frozen|final)(?:_[A-Za-z0-9]+)*_"
            r"(?:weights?|params?|hyperparameters?|threshold|config|state|blend|ratio|coefficients?)\b",
        ):
            failures.append("selected ensemble/tuning state is not materialized")
        if not _contains(code, r"(?:oof\w*).{0,500}(?:best|selected|optimi|minimize|search)") and not _contains(code, r"(?:best|selected|optimi|minimize|search).{0,500}(?:oof\w*)"):
            failures.append("selection is not demonstrably based on OOF data")
        if "record_selection" not in calls or "freeze" not in calls:
            failures.append("selection scope/freeze provenance is missing")
        if "record_global_oof" not in calls:
            failures.append("selection lacks verified global OOF coverage provenance")
        global_oof_line = _call_line(tree, "record_global_oof")
        selection_line = _call_line(tree, "record_selection")
        freeze_line = _call_line(tree, "freeze")
        if (
            global_oof_line is None
            or selection_line is None
            or freeze_line is None
            or not (global_oof_line < selection_line < freeze_line)
        ):
            failures.append("global OOF, selection recording, and freeze are not in causal order")
        selection_analysis = (
            _oof_selection_analysis(tree, global_oof_line, selection_line)
            if global_oof_line is not None and selection_line is not None
            else {"has_metric": False, "causal_selection": []}
        )
        if not selection_analysis["has_metric"]:
            failures.append("selection does not compute a metric/search from OOF predictions")
        elif not selection_analysis["causal_selection"]:
            failures.append("selected protocol state is not causally updated by the OOF metric")
        materialized = [
            name
            for line, name in _selection_state_lines(tree)
            if global_oof_line is not None
            and selection_line is not None
            and global_oof_line < line < selection_line
        ]
        if not materialized:
            failures.append("selected protocol state is not materialized after global OOF scoring")
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
        failures.extend(_canonical_partition_failures(tree))
        freeze_line = _call_line(tree, "freeze")
        final_prediction_lines = [
            int(node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _ast_call_name(node) == "record_prediction"
            and any(
                keyword.arg == "purpose"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "final"
                for keyword in node.keywords
            )
        ]
        final_eval_line = _call_line(tree, "record_final_evaluation")
        assert_line = _call_line(tree, "assert_clean")
        emit_line = _call_line(tree, "emit")
        if (
            freeze_line is None
            or not final_prediction_lines
            or final_eval_line is None
            or assert_line is None
            or emit_line is None
            or not (
                freeze_line
                < min(final_prediction_lines)
                <= max(final_prediction_lines)
                < final_eval_line
                < assert_line
                < emit_line
            )
        ):
            failures.append(
                "protocol freeze, final prediction, one final evaluation, runtime assertion, and emission are not in causal order"
            )
        if len([
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _ast_call_name(node) == "record_final_evaluation"
        ]) != 1:
            failures.append("final evaluation provenance must be recorded exactly once")
        if freeze_line is not None and any(
            line > freeze_line
            for line, _name in _selection_state_lines(tree)
        ):
            failures.append("selected protocol state is modified after protocol freeze")

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
    feedback = [
        {
            "issue_code": item.get("issue_code"),
            "evidence": item.get("evidence"),
            "remediation": item.get("remediation"),
            "line": item.get("line", 0),
        }
        for item in stage_audit.get("issues", [])
        if isinstance(item, dict)
    ]
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": stage,
        "attempt": attempts[stage],
        "status": "passed" if passed else "failed",
        "code_sha256": stage_audit.get("code_sha256"),
        "issue_codes": [item.get("issue_code") for item in stage_audit.get("issues", [])],
        "feedback": feedback,
    })
    if passed:
        tx["current_stage_index"] = int(tx.get("current_stage_index", 0)) + 1
        tx["state"] = "final_pending" if current_stage(tx) == "final_holdout" else "pending"
        if current_stage(tx) is None:
            tx["state"] = "ready_for_execution"
    elif attempts[stage] >= _stage_attempt_limit(tx, stage):
        tx["state"] = "exhausted"
        tx["terminal_reason"] = f"stage_attempts_exhausted:{stage}"
    else:
        tx["state"] = "pending"
    return tx


def rollback_final_runtime_failure(transaction: dict, node_id: str, reason: str) -> dict:
    """Return an executed/crashed final node to the final stage, within budget."""
    tx = copy.deepcopy(transaction)
    stages = list(tx.get("protocol_plan", {}).get("stages") or [])
    stage_attempts = int(tx.get("stage_attempts", {}).get("final_holdout", 0))
    runtime_attempts = dict(tx.get("stage_runtime_attempts") or {})
    runtime_attempts["final_holdout"] = int(runtime_attempts.get("final_holdout", 0)) + 1
    tx["stage_runtime_attempts"] = runtime_attempts
    runtime_limit = max(
        1,
        int(tx.get(
            "final_runtime_attempt_limit",
            _DEFAULT_FINAL_RUNTIME_ATTEMPT_LIMIT,
        )),
    )
    tx["current_stage_index"] = max(0, len(stages) - 1)
    if stage_attempts >= _stage_attempt_limit(tx, "final_holdout"):
        tx["state"] = "exhausted"
        tx["terminal_reason"] = "stage_attempts_exhausted:final_holdout"
    elif runtime_attempts["final_holdout"] >= runtime_limit:
        tx["state"] = "exhausted"
        tx["terminal_reason"] = "runtime_attempts_exhausted:final_holdout"
    else:
        tx["state"] = "pending"
    tx.setdefault("history", []).append({
        "node_id": node_id,
        "stage": "final_holdout_runtime",
        "attempt": runtime_attempts["final_holdout"],
        "stage_attempt": stage_attempts,
        "status": "failed",
        "reason": reason,
        "feedback": [{
            "issue_code": "PROTOCOL_FINAL_RUNTIME_FAILED",
            "evidence": reason,
            "remediation": "Fix the exact runtime exception while preserving every clean protocol stage and protected model component.",
        }],
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
    counts = payload.get("counts", {})
    if any(int(counts.get(key, 0)) <= 0 for key in required_counts):
        return {"status": "blocked", "reason": "runtime provenance is incomplete"}
    stages = set((transaction.get("protocol_plan") or {}).get("stages") or [])
    if "cross_fit" in stages and int(counts.get("global_oof", 0)) != 1:
        return {
            "status": "blocked",
            "reason": "runtime provenance lacks exactly one complete global OOF record",
        }
    return {
        "status": "clean",
        "payload_sha256": hashlib.sha256(matches[-1].encode()).hexdigest(),
        "counts": counts,
    }


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
