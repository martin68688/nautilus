"""Deterministic leakage auditing and persistent audit records.

The LLM leakage reviewer remains useful for task-specific reasoning, but this
module owns the fail-closed checks for code patterns that should never depend
on model judgment. Audit results are deliberately structured so journals,
replay gates, GlobalMemory, and RunForest can enforce the same policy.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


AUDIT_SCHEMA = "mlevolve_leakage_audit_v2"
DETECTOR_VERSION = "deterministic_static_v2"
REGISTRY_SCHEMA = "mlevolve_leakage_registry_record_v2"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

_REGISTRY_LOCK = threading.RLock()

_HOLDOUT_PARTS = {
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "holdout": "holdout",
    "test": "test",
    "dev": "validation",
    "eval": "validation",
    "evaluation": "validation",
    "oos": "holdout",
    "unseen": "holdout",
    "private": "holdout",
}

_STATEFUL_TRANSFORMERS = {
    "CountVectorizer",
    "DictVectorizer",
    "KBinsDiscretizer",
    "KNNImputer",
    "LabelBinarizer",
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

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

_REPAIR_UTILITY_CALLS = {
    "Counter", "DataFrame", "DataLoader", "Dataset", "GradScaler", "Path",
    "Series", "Tensor", "TensorDataset", "load", "loads", "load_state_dict",
    "read_pickle",
}
_REPAIR_MODEL_CALLS = {
    "Adam", "AdamW", "AutoModel", "AutoModelForSequenceClassification",
    "CountVectorizer", "CrossEntropyLoss", "LabelEncoder", "LogisticRegression",
    "StandardScaler", "TfidfTransformer", "TfidfVectorizer", "VarianceThreshold",
    "XGBClassifier", "XGBRegressor",
}
_REPAIR_MODEL_CALL_PATTERN = re.compile(
    r"(?:Attention|Boost|Classifier|Conv\d*d?|Decoder|Dropout|Embedding|Encoder|"
    r"Forest|GRU|LSTM|LayerNorm|Linear|Loss|Model|Net|Optimizer|Regression|Regressor|"
    r"RNN|SVC|SVM|SVR|Transformer)$"
)
_MODEL_LITERAL_PATTERN = re.compile(
    r"(?:deberta|modernbert|roberta|bert|electra|transformer|xgboost|lightgbm)",
    re.IGNORECASE,
)
_TRAINING_HYPERPARAMETER_PATTERN = re.compile(
    r"^(?:batch_size|learning_rate|lr|weight_decay|num_epochs|epochs|max_epochs|"
    r"patience|dropout|hidden_size|num_layers|gradient_accumulation_steps|"
    r"warmup_steps|warmup_ratio|max_length|seed)$",
    re.IGNORECASE,
)
_MODEL_ASSIGNMENT_TARGET_PATTERN = re.compile(
    r"(?:^|_)(?:model|backbone|encoder|decoder|classifier|regressor|estimator|network|net|head|learner)(?:$|_)",
    re.IGNORECASE,
)
_MODEL_IDENTITY_TARGET_PATTERN = re.compile(
    r"(?:model|backbone|encoder|decoder|checkpoint|pretrained|weights?|architecture)(?:_?name|_?path|_?id)?$",
    re.IGNORECASE,
)
_PROTOCOL_GUARD_METHODS = {
    "assert_clean",
    "emit",
    "freeze",
    "record_final_evaluation",
    "record_fit",
    "record_global_oof",
    "record_prediction",
    "record_selection",
    "register_partition",
}


def _is_model_identity_literal(value: str) -> bool:
    if not _MODEL_LITERAL_PATTERN.search(value):
        return False
    stripped = value.strip()
    return bool(stripped and (not re.search(r"\s", stripped) or "/" in stripped))


def _is_local_training_artifact_literal(value: str) -> bool:
    """Return whether a model-like literal is a local output artifact path.

    Protocol repair may need one checkpoint file per fold while preserving the
    same pretrained model identity.  Those files are execution artifacts, not
    replacement model identities.  Remote/vendor identifiers remain protected
    by the stricter model-literal gate below.
    """
    stripped = str(value or "").strip().replace("\\", "/")
    if not stripped or "://" in stripped:
        return False
    path = Path(stripped)
    has_local_prefix = (
        stripped.startswith(("./", "../", "/tmp/", "/workspace/"))
        or "/working/" in f"/{stripped.lstrip('/')}"
    )
    artifact_suffixes = {
        ".bin", ".ckpt", ".csv", ".joblib", ".json", ".npy", ".npz",
        ".parquet", ".pkl", ".pt", ".pth", ".safetensors",
    }
    has_artifact_shape = path.suffix.lower() in artifact_suffixes or bool(
        re.search(r"(?:^|[/_.-])(?:fold|epoch|best|last|checkpoint)(?:$|[/_.-]|\d)", stripped, re.IGNORECASE)
    )
    return has_local_prefix and has_artifact_shape


def code_sha256(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def structural_sha256(code: str) -> str:
    """Hash Python structure while ignoring local identifier renames and formatting."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ""

    class Canonicalizer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.copy_location(ast.Name(id="identifier", ctx=node.ctx), node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = "argument"
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = "function"
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.name = "function"
            return self.generic_visit(node)

    canonical = Canonicalizer().visit(tree)
    ast.fix_missing_locations(canonical)
    payload = ast.dump(canonical, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_repair_protected_call(name: str) -> bool:
    return bool(
        name
        and name not in _REPAIR_UTILITY_CALLS
        and (name in _REPAIR_MODEL_CALLS or _REPAIR_MODEL_CALL_PATTERN.search(name))
    )


def build_repair_preservation_contract(code: str) -> dict[str, Any]:
    """Freeze model-design elements that a leakage-only repair may not remove."""
    source_hash = code_sha256(code)
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return {
            "schema": "mlevolve_repair_preservation_v1",
            "source_code_sha256": source_hash,
            "status": "unavailable",
            "error": f"source AST parse failed at line {exc.lineno}: {exc.msg}",
        }

    semantic_model_call_ids: set[int] = set()
    semantic_model_literals: set[str] = set()
    protocol_component_labels: set[str] = set()
    protocol_metadata_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in _PROTOCOL_GUARD_METHODS:
            protocol_metadata_literals.update(
                child.value
                for arg in node.args
                for child in ast.walk(arg)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
            protocol_metadata_literals.update(
                child.value
                for keyword in node.keywords
                for child in ast.walk(keyword.value)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            )
        if isinstance(node, ast.Call) and _call_name(node.func) in {
            "record_fit", "record_prediction"
        }:
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                protocol_component_labels.add(node.args[0].value)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = [name for target in targets for name in _target_names(target)]
        if isinstance(node.value, ast.Call) and any(
            _MODEL_ASSIGNMENT_TARGET_PATTERN.search(name) for name in target_names
        ):
            semantic_model_call_ids.add(id(node.value))
        if (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and any(_MODEL_IDENTITY_TARGET_PATTERN.search(name) for name in target_names)
        ):
            semantic_model_literals.add(node.value.value)

    component_calls: dict[str, int] = {}
    component_call_hashes: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if _is_repair_protected_call(name) or (
            id(node) in semantic_model_call_ids and name not in _REPAIR_UTILITY_CALLS
        ):
            component_calls[name] = component_calls.get(name, 0) + 1
            component_call_hashes.setdefault(name, []).append(
                structural_sha256(ast.unparse(node))
            )

    architecture_hashes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        class_calls = {
            _call_name(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        }
        bases = " ".join(ast.unparse(base) for base in node.bases)
        if any(_is_repair_protected_call(name) for name in class_calls) or re.search(
            r"(?:nn\.)?Module|BaseEstimator", bases
        ):
            architecture_hashes.append(structural_sha256(ast.unparse(node)))

    training_hyperparameters: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in _target_names(target):
                if _TRAINING_HYPERPARAMETER_PATTERN.fullmatch(name):
                    training_hyperparameters.setdefault(name, []).append(
                        structural_sha256(ast.unparse(node.value))
                    )

    model_literals = sorted({
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value) <= 240
        and _is_model_identity_literal(node.value)
    } | semantic_model_literals)
    return {
        "schema": "mlevolve_repair_preservation_v1",
        "source_code_sha256": source_hash,
        "status": "frozen",
        "component_calls": dict(sorted(component_calls.items())),
        "component_call_hashes": {
            name: sorted(hashes) for name, hashes in sorted(component_call_hashes.items())
        },
        "architecture_hashes": sorted(architecture_hashes),
        "training_hyperparameters": {
            name: sorted(hashes)
            for name, hashes in sorted(training_hyperparameters.items())
        },
        "model_literals": model_literals,
        # Provenance component names describe events; they are not model or
        # checkpoint identities and may be introduced by protocol repair.
        "protocol_component_labels": sorted(protocol_component_labels),
        "protocol_metadata_literals": sorted(protocol_metadata_literals),
    }


def audit_repair_preservation(code: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when a repair removes or rewrites protected model design."""
    if not isinstance(contract, dict) or contract.get("status") != "frozen":
        issue = _issue(
            code=code, node=None,
            issue_code="REPAIR_PRESERVATION_CONTRACT_UNAVAILABLE",
            category="repair_integrity", severity="high",
            evidence=str((contract or {}).get("error") or "No frozen preservation contract is available."),
            remediation="Recreate the repair child from the audited parent and freeze its model-design contract.",
            execution_disposition="block",
        )
        return _summarize_audit(code, [issue])

    current = build_repair_preservation_contract(code)
    if current.get("status") != "frozen":
        issue = _issue(
            code=code, node=None,
            issue_code="REPAIR_PRESERVATION_AUDIT_UNAVAILABLE",
            category="repair_integrity", severity="high",
            evidence=str(current.get("error") or "Repaired code could not be structurally inspected."),
            remediation="Repair the syntax without changing the protected model design.",
            execution_disposition="block",
        )
        return _summarize_audit(code, [issue])

    issues: list[dict[str, Any]] = []
    missing_components = {
        name: int(required) - int(current.get("component_calls", {}).get(name, 0))
        for name, required in contract.get("component_calls", {}).items()
        if int(current.get("component_calls", {}).get(name, 0)) < int(required)
    }
    if missing_components:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_COMPONENT_REMOVED",
            category="repair_integrity", severity="critical",
            evidence=f"Protected model/feature calls were removed or reduced: {missing_components}",
            remediation="Restore every protected model, feature, optimizer, loss, and ensemble component; modify only data/evaluation boundaries.",
            execution_disposition="block",
        ))

    added_components = {
        name: int(count)
        for name, count in current.get("component_calls", {}).items()
        if name not in contract.get("component_calls", {})
    }
    if added_components:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_COMPONENT_ADDED",
            category="repair_integrity", severity="critical",
            evidence=f"Leakage-only repair introduced new learned components: {added_components}",
            remediation="Remove newly introduced models/features; duplicate only exact protected constructors when folds or final refits require another instance.",
            execution_disposition="block",
        ))

    changed_configurations: dict[str, int] = {}
    current_hashes = current.get("component_call_hashes", {})
    for name, required_hashes in contract.get("component_call_hashes", {}).items():
        available = list(current_hashes.get(name, []))
        missing_count = 0
        for required_hash in required_hashes:
            if required_hash in available:
                available.remove(required_hash)
            else:
                missing_count += 1
        if missing_count:
            changed_configurations[name] = missing_count
    if changed_configurations:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_CONFIGURATION_CHANGED",
            category="repair_integrity", severity="critical",
            evidence=f"Protected component constructor/configuration changed: {changed_configurations}",
            remediation="Restore the original component arguments and hyperparameters; only data/evaluation boundaries may change.",
            execution_disposition="block",
        ))

    unapproved_configurations: dict[str, int] = {}
    for name, hashes in current_hashes.items():
        allowed = set(contract.get("component_call_hashes", {}).get(name, []))
        if not allowed:
            continue
        unexpected_count = sum(hash_value not in allowed for hash_value in hashes)
        if unexpected_count:
            unapproved_configurations[name] = unexpected_count
    if unapproved_configurations:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_UNAPPROVED_MODEL_CONFIGURATION_ADDED",
            category="repair_integrity", severity="critical",
            evidence=f"Additional protected component instances changed constructor/configuration: {unapproved_configurations}",
            remediation="Every fold/final duplicate of a protected component must use an exact constructor already frozen by the source contract.",
            execution_disposition="block",
        ))

    changed_hyperparameters: dict[str, int] = {}
    current_hyperparameters = current.get("training_hyperparameters", {})
    for name, required_hashes in contract.get("training_hyperparameters", {}).items():
        available = list(current_hyperparameters.get(name, []))
        missing_count = 0
        for required_hash in required_hashes:
            if required_hash in available:
                available.remove(required_hash)
            else:
                missing_count += 1
        if missing_count:
            changed_hyperparameters[name] = missing_count
    if changed_hyperparameters:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_TRAINING_HYPERPARAMETER_CHANGED",
            category="repair_integrity", severity="critical",
            evidence=f"Protected training hyperparameters changed or disappeared: {changed_hyperparameters}",
            remediation="Restore the original training budget and hyperparameters; leakage repair is not a model-tuning pass.",
            execution_disposition="block",
        ))

    missing_architectures = sorted(
        set(contract.get("architecture_hashes", []))
        - set(current.get("architecture_hashes", []))
    )
    if missing_architectures:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_ARCHITECTURE_CHANGED",
            category="repair_integrity", severity="critical",
            evidence=f"{len(missing_architectures)} protected model class structure(s) no longer match the parent.",
            remediation="Restore the parent model classes exactly; leakage repair belongs in split, fitting, selection, and reporting logic.",
            execution_disposition="block",
        ))

    added_architectures = sorted(
        set(current.get("architecture_hashes", []))
        - set(contract.get("architecture_hashes", []))
    )
    if added_architectures:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_ARCHITECTURE_ADDED",
            category="repair_integrity", severity="critical",
            evidence=f"{len(added_architectures)} new model class structure(s) appeared during leakage-only repair.",
            remediation="Do not introduce replacement architectures during protocol repair.",
            execution_disposition="block",
        ))

    missing_literals = sorted(
        set(contract.get("model_literals", [])) - set(current.get("model_literals", []))
    )
    if missing_literals:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_IDENTITY_CHANGED",
            category="repair_integrity", severity="critical",
            evidence=f"Protected model/checkpoint identifiers were removed: {missing_literals}",
            remediation="Keep the original pretrained model names and checkpoint paths unchanged during leakage repair.",
            execution_disposition="block",
        ))

    added_literals = sorted(
        literal
        for literal in (
            set(current.get("model_literals", []))
            - set(contract.get("model_literals", []))
        )
        if not _is_local_training_artifact_literal(literal)
        and literal not in set(current.get("protocol_component_labels", []))
        and literal not in set(current.get("protocol_metadata_literals", []))
    )
    if added_literals:
        issues.append(_issue(
            code=code, node=None,
            issue_code="REPAIR_MODEL_IDENTITY_ADDED",
            category="repair_integrity", severity="critical",
            evidence=f"Leakage-only repair introduced new model/checkpoint identifiers: {added_literals}",
            remediation="Use only model and checkpoint identifiers frozen by the source contract.",
            execution_disposition="block",
        ))

    return _summarize_audit(code, issues)


def format_repair_preservation_contract(contract: dict[str, Any]) -> str:
    components = contract.get("component_calls", {}) if isinstance(contract, dict) else {}
    literals = contract.get("model_literals", []) if isinstance(contract, dict) else []
    architecture_count = len(contract.get("architecture_hashes", [])) if isinstance(contract, dict) else 0
    hyperparameters = contract.get("training_hyperparameters", {}) if isinstance(contract, dict) else {}
    return (
        f"Protected component calls (minimum counts): {components}; "
        f"protected model/checkpoint identifiers: {literals}; "
        f"protected model class structures: {architecture_count}; "
        f"protected training hyperparameters: {sorted(hyperparameters)}."
    )


def _semantic_taints(name: str) -> set[str]:
    parts = {part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part}
    return {_HOLDOUT_PARTS[part] for part in parts if part in _HOLDOUT_PARTS}


def _root_name(expr: ast.AST | None) -> str:
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else ""


def _call_name(expr: ast.AST | None) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ""


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(_target_names(item))
        return names
    return []


def _expr_taints(expr: ast.AST | None, taints: dict[str, set[str]]) -> set[str]:
    if expr is None:
        return set()
    if isinstance(expr, ast.Name):
        if expr.id in taints:
            return set(taints[expr.id])
        return _semantic_taints(expr.id)
    if isinstance(expr, ast.Attribute):
        return _expr_taints(expr.value, taints)
    if isinstance(expr, ast.Call):
        found: set[str] = set()
        if isinstance(expr.func, ast.Attribute):
            found.update(_expr_taints(expr.func.value, taints))
        for arg in expr.args:
            found.update(_expr_taints(arg, taints))
        for keyword in expr.keywords:
            found.update(_expr_taints(keyword.value, taints))
        return found
    found: set[str] = set()
    for child in ast.iter_child_nodes(expr):
        found.update(_expr_taints(child, taints))
    return found


def _source_line(code: str, node: ast.AST, limit: int = 280) -> str:
    segment = ast.get_source_segment(code, node) or ""
    segment = " ".join(segment.split())
    return segment[:limit]


def _is_explicit_train_partition(target_name: str, value: ast.AST, code: str) -> bool:
    if "train" not in target_name.lower() or not isinstance(value, ast.Subscript):
        return False
    source = " ".join((ast.get_source_segment(code, value) or "").split()).lower()
    return bool(
        re.search(
            r"(?:\.iloc|\.loc)?\[\s*(?::\s*(?:len\(\s*train|n_train|train_size|split_idx)|(?:train|tr)_?(?:idx|indices))",
            source,
        )
    )


def _is_split_assignment(value: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and _call_name(node.func) == "train_test_split"
        for node in ast.walk(value)
    )


def _split_output_taints(names: list[str], value: ast.AST) -> list[set[str]] | None:
    if not _is_split_assignment(value):
        return None
    output: list[set[str]] = []
    for index, name in enumerate(names):
        if index % 2 == 0:
            output.append(set())
        else:
            semantic = _semantic_taints(name)
            output.append(semantic or {"validation"})
    return output


def _issue(
    *,
    code: str,
    node: ast.AST | None,
    issue_code: str,
    category: str,
    severity: str,
    evidence: str,
    remediation: str,
    execution_disposition: str,
) -> dict[str, Any]:
    line = int(getattr(node, "lineno", 0) or 0)
    return {
        "issue_code": issue_code,
        "category": category,
        "severity": severity,
        "line": line,
        "evidence": evidence or (_source_line(code, node) if node is not None else ""),
        "remediation": remediation,
        "execution_disposition": execution_disposition,
        "detector": DETECTOR_VERSION,
    }


def _dedupe_issues(issues: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in issues:
        key = (
            str(item.get("issue_code", "")),
            int(item.get("line", 0) or 0),
            str(item.get("evidence", "")),
        )
        by_key.setdefault(key, item)
    return sorted(by_key.values(), key=lambda item: (int(item.get("line", 0)), str(item.get("issue_code", ""))))


def _nodes_in_scope(scope: ast.AST) -> list[ast.AST]:
    """Walk one lexical scope without mixing local variables from nested scopes."""
    roots = list(getattr(scope, "body", []))
    found: list[ast.AST] = []
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return found


def _summarize_audit(code: str, issues: Iterable[dict[str, Any]], *, detector_status: str = "complete") -> dict[str, Any]:
    issue_list = _dedupe_issues(issues)
    hard_block = any(item.get("execution_disposition") == "block" for item in issue_list)
    has_protocol_bias = any(item.get("category") == "selection_bias" for item in issue_list)
    if hard_block:
        status = "blocked"
        memory_disposition = "quarantine"
        metric_disposition = "reject"
        execution_disposition = "block"
        search_disposition = "blocked"
    elif has_protocol_bias:
        status = "protocol_biased"
        memory_disposition = "negative_only"
        metric_disposition = "protocol_biased"
        execution_disposition = "allow_diagnostic"
        search_disposition = "repair_only"
    elif detector_status != "complete":
        status = "audit_unavailable"
        memory_disposition = "negative_only"
        metric_disposition = "unverified"
        execution_disposition = "allow_diagnostic"
        search_disposition = "provisional"
    elif issue_list:
        status = "warning"
        memory_disposition = "negative_only"
        metric_disposition = "unverified"
        execution_disposition = "allow_diagnostic"
        search_disposition = "repair_only"
    else:
        status = "clean"
        memory_disposition = "positive_eligible"
        metric_disposition = "accept"
        execution_disposition = "allow"
        search_disposition = "normal"

    max_severity = "none"
    for item in issue_list:
        severity = str(item.get("severity", "low"))
        if _SEVERITY_RANK.get(severity, 0) > _SEVERITY_RANK.get(max_severity, 0):
            max_severity = severity

    return {
        "schema": AUDIT_SCHEMA,
        "detector_version": DETECTOR_VERSION,
        "detector_status": detector_status,
        "code_sha256": code_sha256(code),
        "structural_sha256": structural_sha256(code),
        "status": status,
        "max_severity": max_severity,
        "hard_block": hard_block,
        "paper_grade_eligible": status == "clean",
        "metric_disposition": metric_disposition,
        "memory_disposition": memory_disposition,
        "execution_disposition": execution_disposition,
        "search_disposition": search_disposition,
        "rank_eligible": status == "clean",
        "repair_required": status != "clean",
        "issues": issue_list,
    }


def rank_eligible(agent: Any, node: Any) -> bool:
    """Return whether a node may influence certified ranking and artifacts."""
    if getattr(getattr(agent, "acfg", None), "check_data_leakage", False) is not True:
        return bool(
            node is not None
            and getattr(node, "is_buggy", None) is not True
            and getattr(node, "is_valid", None) is not False
        )
    audit = getattr(node, "leakage_audit", None)
    if not isinstance(audit, dict) or not audit:
        return False
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("detector_status") != "complete":
        return False
    if audit.get("code_sha256") != code_sha256(getattr(node, "code", "")):
        return False
    leakage_clean = bool(
        audit.get("status") == "clean"
        and audit.get("metric_disposition") == "accept"
        and audit.get("paper_grade_eligible") is True
    )
    if not leakage_clean:
        return False
    if (
        getattr(node, "draft_role", None) == "novel_exploration"
        and getattr(node, "selected_strategy", None)
    ):
        alignment = getattr(node, "strategy_alignment", None) or {}
        return alignment.get("status") == "verified" and alignment.get("rank_eligible") is True
    return True


def failure_pattern_audit(code: str, patterns: Iterable[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for pattern in patterns:
        original_code = str(pattern.get("issue_code") or "UNKNOWN")
        issues.append(
            _issue(
                code=code,
                node=None,
                issue_code=f"HISTORICAL_FAILURE_PATTERN_MATCH::{original_code}",
                category=str(pattern.get("category") or "historical_failure"),
                severity=str(pattern.get("severity") or "high"),
                evidence=(
                    "Current code structurally matches a previously audited failure. "
                    + str(pattern.get("evidence") or "")
                ),
                remediation=str(pattern.get("remediation") or "Resolve the historical failure before reuse."),
                execution_disposition=str(pattern.get("execution_disposition") or "review"),
            )
        )
    return _summarize_audit(code, issues)


def audit_code(code: str) -> dict[str, Any]:
    """Run deterministic static leakage checks over one Python solution."""
    code = code or ""
    if not code.strip():
        issue = _issue(
            code=code,
            node=None,
            issue_code="EMPTY_CODE_AUDIT_FAILED",
            category="audit_failure",
            severity="high",
            evidence="No executable Python source was supplied for audit.",
            remediation="Generate non-empty executable code before certification.",
            execution_disposition="block",
        )
        return _summarize_audit(code, [issue], detector_status="unavailable")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        issue = _issue(
            code=code,
            node=None,
            issue_code="STATIC_AUDIT_PARSE_FAILED",
            category="audit_failure",
            severity="medium",
            evidence=f"Python AST parsing failed at line {exc.lineno}: {exc.msg}",
            remediation="Repair syntax before the solution can be certified or admitted to memory.",
            execution_disposition="review",
        )
        issue["line"] = int(exc.lineno or 0)
        return _summarize_audit(code, [issue], detector_status="unavailable")

    issues: list[dict[str, Any]] = []
    scopes: list[ast.AST] = [tree]
    scopes.extend(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for scope in scopes:
        scope_nodes = _nodes_in_scope(scope)
        assignments: list[tuple[list[str], ast.AST]] = []
        constructors: dict[str, str] = {}
        for node in scope_nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [name for target in targets for name in _target_names(target)]
                value = node.value
                if names and value is not None:
                    assignments.append((names, value))
                    if isinstance(value, ast.Call):
                        constructor = _call_name(value.func)
                        for name in names:
                            constructors[name] = constructor

        taints: dict[str, set[str]] = {}
        assignments.sort(key=lambda item: int(getattr(item[1], "lineno", 0) or 0))
        for names, value in assignments:
            value_taints = _expr_taints(value, taints)
            split_taints = _split_output_taints(names, value)
            for index, name in enumerate(names):
                if split_taints is not None:
                    taints[name] = set(split_taints[index])
                elif _is_explicit_train_partition(name, value, code):
                    taints[name] = set(value_taints)
                else:
                    taints[name] = _semantic_taints(name) | value_taints

        split_lines = [
            int(node.lineno)
            for node in scope_nodes
            if isinstance(node, ast.Call)
            and _call_name(node.func) in {"train_test_split", "KFold", "StratifiedKFold", "GroupKFold"}
        ]
        first_split_line = min(split_lines) if split_lines else None

        for node in scope_nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in {"fit", "fit_transform", "partial_fit"}:
                continue
            receiver = _root_name(node.func.value)
            constructor = constructors.get(receiver, "")
            transformer_like = constructor in _STATEFUL_TRANSFORMERS or method == "fit_transform"
            fit_taints: set[str] = set()
            for arg in node.args:
                fit_taints.update(_expr_taints(arg, taints))
            for keyword in node.keywords:
                if keyword.arg in {
                    "eval_set", "eval_metric", "validation_data", "validation_set",
                    "val_set", "callbacks",
                }:
                    continue
                fit_taints.update(_expr_taints(keyword.value, taints))
            if fit_taints:
                holdouts = ", ".join(sorted(fit_taints))
                issues.append(
                    _issue(
                        code=code,
                        node=node,
                        issue_code="TRANSFORM_FIT_ON_HOLDOUT",
                        category="transductive_contamination",
                        severity="high",
                        evidence=f"{constructor or receiver}.{method} receives data derived from: {holdouts}. Source: {_source_line(code, node)}",
                        remediation="Fit the transformer on the training partition only, then call transform on validation and test partitions.",
                        execution_disposition="block",
                    )
                )
            elif transformer_like and first_split_line is not None and int(node.lineno) < first_split_line:
                arg_names = {_root_name(arg) for arg in node.args}
                if any(name and "train" in name.lower() for name in arg_names) and constructor not in {"LabelBinarizer", "LabelEncoder"}:
                    issues.append(
                        _issue(
                            code=code,
                            node=node,
                            issue_code="TRANSFORM_FIT_BEFORE_SPLIT",
                            category="validation_contamination",
                            severity="medium",
                            evidence=f"{constructor or receiver}.{method} is called before the first data split. Source: {_source_line(code, node)}",
                            remediation="Split rows first and fit a fresh transformer inside each training split or fold.",
                            execution_disposition="allow_with_warning",
                        )
                    )

    reset_vars = re.findall(
        r"(?m)^\s*(\w+)\s*=\s*[^\n]*\.iloc\[[^\n]+\]\.reset_index\(\s*drop\s*=\s*True\s*\)",
        code,
    )
    for reset_var in reset_vars:
        index_match = re.search(rf"(?m)^\s*(\w+)\s*=\s*{re.escape(reset_var)}\.index(?:\.tolist\(\))?", code)
        if not index_match:
            continue
        index_var = index_match.group(1)
        misuse = re.search(rf"(?m)^\s*\w*(?:label|target|text)\w*\s*=\s*\w+\s*\[\s*{re.escape(index_var)}\s*\]", code, re.IGNORECASE)
        if misuse:
            line = code[: misuse.start()].count("\n") + 1
            issue = _issue(
                code=code,
                node=None,
                issue_code="RESET_INDEX_ORIGINAL_ARRAY_MISALIGNMENT",
                category="target_leakage",
                severity="critical",
                evidence=f"Reset indices from {reset_var} are reused to index a different original label/text array: {' '.join(misuse.group(0).split())}",
                remediation="Use the original split indices directly, or read features and labels from the same split DataFrame after reset_index.",
                execution_disposition="block",
            )
            issue["line"] = line
            issues.append(issue)

    lower = code.lower()
    ensemble_weight_search = bool(
        re.search(r"(?:best|optimal|optimized)[_\s]*(?:ensemble[_\s]*)?weights?", lower)
        and re.search(r"(?:val|valid|validation|dev|holdout|oof)[_\w]*(?:proba|pred)|y_(?:val|valid|dev|holdout)", lower)
        and re.search(r"for\s+\w+\s+in\s+(?:np\.)?(?:arange|linspace)|gridsearch|optuna|minimize\s*\(", lower)
        and re.search(r"log[_\s]?loss|compute_log_loss|brier|score", lower)
    )
    guarded_oof_selection = bool(
        re.search(r"\.record_global_oof\s*\(", lower)
        and re.search(r"\.record_selection\s*\(", lower)
        and re.search(r"\.freeze\s*\(\s*\)", lower)
        and re.search(r"record_selection\s*\([^\n]*(?:outer_train|oof)", lower)
    )
    if ensemble_weight_search and not guarded_oof_selection:
        line_match = re.search(r"(?m)^.*(?:best|optimal|optimized)[_\s]*(?:ensemble[_\s]*)?weights?.*$", code, re.IGNORECASE)
        line = code[: line_match.start()].count("\n") + 1 if line_match else 0
        issue = _issue(
            code=code,
            node=None,
            issue_code="REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
            category="selection_bias",
            severity="medium",
            evidence="Ensemble weights are optimized on validation predictions and the optimized validation loss is reported on the same rows.",
            remediation="Tune weights on an inner/dev split or cross-fitted OOF predictions, then report once on an untouched outer holdout.",
            execution_disposition="allow_with_warning",
        )
        issue["line"] = line
        issues.append(issue)

    if re.search(r"\b(?:last_fold_model|best_fold_model)\b", lower) and re.search(
        r"extract[_\w]*(?:embedding|feature)|(?:embedding|feature)[_\w]*extract", lower
    ) and re.search(r"\b(?:oof|holdout|x_val|val_loader)\b", lower):
        issues.append(
            _issue(
                code=code,
                node=None,
                issue_code="CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE",
                category="target_leakage",
                severity="high",
                evidence="A single selected fold model appears to generate embeddings/features for OOF or holdout rows.",
                remediation="Generate every row's OOF feature with a fold model that did not train on that row.",
                execution_disposition="review",
            )
        )

    return _summarize_audit(code, issues)


def merge_audits(code: str, *audits: dict[str, Any] | None) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    detector_status = "complete"
    detector_rank = {"complete": 0, "llm_unavailable": 1, "registry_unavailable": 2, "registry_corrupt": 3, "unavailable": 4}
    llm_reviews: list[dict[str, Any]] = []
    for audit in audits:
        if not audit:
            continue
        issues.extend(item for item in audit.get("issues", []) if isinstance(item, dict))
        current_status = str(audit.get("detector_status") or "complete")
        if detector_rank.get(current_status, 4) > detector_rank.get(detector_status, 0):
            detector_status = current_status
        llm_reviews.extend(item for item in audit.get("llm_reviews", []) if isinstance(item, dict))
    merged = _summarize_audit(code, issues, detector_status=detector_status)
    if llm_reviews:
        merged["llm_reviews"] = llm_reviews
    return merged


def llm_result_to_audit(code: str, result: dict[str, Any]) -> dict[str, Any]:
    classification = str(result.get("classification") or "").strip().lower()
    has_leakage = bool(result.get("has_leakage"))
    if not classification:
        classification = "hard_leakage" if has_leakage else "clean"
    confidence = str(result.get("confidence") or "low").lower()
    reason = str(result.get("reason") or result.get("leakage_reason") or "")
    if classification == "audit_unavailable":
        audit = _summarize_audit(code, [], detector_status="llm_unavailable")
    elif classification == "clean":
        audit = _summarize_audit(code, [])
    else:
        mapping = {
            "hard_leakage": ("LLM_HARD_LEAKAGE", "target_leakage", "block"),
            "transductive_contamination": ("LLM_TRANSDUCTIVE_CONTAMINATION", "transductive_contamination", "block"),
            "selection_bias": ("LLM_SELECTION_BIAS", "selection_bias", "allow_with_warning"),
            "warning": ("LLM_AUDIT_WARNING", "warning", "review"),
        }
        issue_code, category, disposition = mapping.get(classification, mapping["warning"])
        severity = {"high": "high", "medium": "medium", "low": "low"}.get(confidence, "low")
        issue = _issue(
            code=code,
            node=None,
            issue_code=issue_code,
            category=category,
            severity=severity,
            evidence=reason,
            remediation="Apply the review evidence, rerun deterministic audit, and use an untouched evaluation set before certification.",
            execution_disposition=disposition,
        )
        audit = _summarize_audit(code, [issue])
    audit["llm_reviews"] = [{
        "classification": classification,
        "confidence": confidence,
        "reason": reason,
    }]
    return audit


def format_audit(audit: dict[str, Any], *, heading: str = "LEAKAGE AUDIT") -> str:
    lines = [
        f"{heading}: status={audit.get('status')} severity={audit.get('max_severity')} ",
        f"metric={audit.get('metric_disposition')} memory={audit.get('memory_disposition')}",
    ]
    for item in audit.get("issues", []):
        location = f" line {item.get('line')}" if item.get("line") else ""
        lines.append(f"- [{item.get('issue_code')}]{location}: {item.get('evidence')}")
        if item.get("remediation"):
            lines.append(f"  Fix: {item.get('remediation')}")
    return "\n".join(lines)


def registry_dir_for_agent(agent: Any) -> Path | None:
    workspace = getattr(getattr(agent, "cfg", None), "workspace_dir", None)
    if not workspace:
        return None
    return Path(workspace) / "global_memory" / "leakage_audits"


@contextmanager
def _registry_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _REGISTRY_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def persist_audit(agent: Any, node: Any) -> Path | None:
    audit = getattr(node, "leakage_audit", None) or {}
    digest = str(audit.get("code_sha256") or code_sha256(getattr(node, "code", "")))
    registry_dir = registry_dir_for_agent(agent)
    if registry_dir is None:
        return None
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{digest}.json"
    lock_path = registry_dir / f"{digest}.lock"
    with _registry_file_lock(lock_path):
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        occurrences = list(existing.get("occurrences") or [])
        occurrence = {
            "node_id": str(getattr(node, "id", "")),
            "stage": str(getattr(node, "stage", "")),
            "draft_role": str(getattr(node, "draft_role", "") or ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        occurrence_key = (occurrence["node_id"], occurrence["stage"], occurrence["draft_role"])
        if not any(
            (str(item.get("node_id")), str(item.get("stage")), str(item.get("draft_role"))) == occurrence_key
            for item in occurrences if isinstance(item, dict)
        ):
            occurrences.append(occurrence)
        payload = {
            "schema": REGISTRY_SCHEMA,
            "code_sha256": digest,
            "structural_sha256": audit.get("structural_sha256"),
            "audit": audit,
            "occurrences": occurrences,
        }
        tmp = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return path


def load_registry_audit(agent: Any, digest: str) -> dict[str, Any] | None:
    registry_dir = registry_dir_for_agent(agent)
    if registry_dir is None:
        return None
    path = registry_dir / f"{digest}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") not in {
            "mlevolve_leakage_registry_record_v1", REGISTRY_SCHEMA
        }:
            raise ValueError("unsupported leakage registry schema")
        if str(payload.get("code_sha256") or "") != str(digest):
            raise ValueError("leakage registry hash mismatch")
        audit = payload.get("audit")
        if not isinstance(audit, dict):
            raise ValueError("missing leakage registry audit")
        return audit
    except Exception:
        return {
            "schema": AUDIT_SCHEMA,
            "detector_version": DETECTOR_VERSION,
            "detector_status": "registry_corrupt",
            "status": "audit_unavailable",
            "hard_block": False,
            "paper_grade_eligible": False,
            "memory_disposition": "negative_only",
            "metric_disposition": "unverified",
            "execution_disposition": "allow_diagnostic",
            "search_disposition": "provisional",
            "rank_eligible": False,
            "repair_required": True,
            "issues": [],
        }
