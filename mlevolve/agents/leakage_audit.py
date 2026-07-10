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
import time
from pathlib import Path
from typing import Any, Iterable


AUDIT_SCHEMA = "mlevolve_leakage_audit_v1"
DETECTOR_VERSION = "deterministic_static_v1"

_HOLDOUT_PARTS = {
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "holdout": "holdout",
    "test": "test",
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


def code_sha256(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


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
        return _semantic_taints(expr.id) | taints.get(expr.id, set())
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
        and _call_name(node.func) in {"train_test_split", "split"}
        for node in ast.walk(value)
    )


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
    elif has_protocol_bias:
        status = "protocol_biased"
        memory_disposition = "negative_only"
        metric_disposition = "protocol_biased"
    elif detector_status != "complete":
        status = "audit_unavailable"
        memory_disposition = "negative_only"
        metric_disposition = "unverified"
    elif issue_list:
        status = "warning"
        memory_disposition = "negative_only"
        metric_disposition = "unverified"
    else:
        status = "clean"
        memory_disposition = "positive_eligible"
        metric_disposition = "accept"

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
        "status": status,
        "max_severity": max_severity,
        "hard_block": hard_block,
        "paper_grade_eligible": status == "clean",
        "metric_disposition": metric_disposition,
        "memory_disposition": memory_disposition,
        "issues": issue_list,
    }


def audit_code(code: str) -> dict[str, Any]:
    """Run deterministic static leakage checks over one Python solution."""
    code = code or ""
    if not code.strip():
        return _summarize_audit(code, [], detector_status="complete")

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
        for names, _value in assignments:
            for name in names:
                taints.setdefault(name, set()).update(_semantic_taints(name))
        for _ in range(max(2, len(assignments) + 1)):
            changed = False
            for names, value in assignments:
                value_taints = _expr_taints(value, taints)
                for name in names:
                    assigned_taints = set(value_taints)
                    if _is_split_assignment(value) or _is_explicit_train_partition(name, value, code):
                        assigned_taints.clear()
                    before = len(taints.setdefault(name, set()))
                    taints[name].update(assigned_taints)
                    changed = changed or len(taints[name]) != before
            if not changed:
                break

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
            transformer_like = constructor in _STATEFUL_TRANSFORMERS or (not constructor and method == "fit_transform")
            if not transformer_like:
                continue
            fit_taints: set[str] = set()
            for arg in node.args:
                fit_taints.update(_expr_taints(arg, taints))
            for keyword in node.keywords:
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
            elif first_split_line is not None and int(node.lineno) < first_split_line:
                arg_names = {_root_name(arg) for arg in node.args}
                if any(name and "train" in name.lower() for name in arg_names) and constructor not in {"LabelBinarizer", "LabelEncoder"}:
                    issues.append(
                        _issue(
                            code=code,
                            node=node,
                            issue_code="TRANSFORM_FIT_BEFORE_SPLIT",
                            category="validation_contamination",
                            severity="high",
                            evidence=f"{constructor or receiver}.{method} is called before the first data split. Source: {_source_line(code, node)}",
                            remediation="Split rows first and fit a fresh transformer inside each training split or fold.",
                            execution_disposition="block",
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
        and re.search(r"val(?:idation)?[_\w]*proba|val_probas|y_val", lower)
        and re.search(r"for\s+\w+\s+in\s+(?:np\.)?(?:arange|linspace)|gridsearch", lower)
        and re.search(r"log[_\s]?loss|compute_log_loss", lower)
    )
    if ensemble_weight_search:
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
    llm_reviews: list[dict[str, Any]] = []
    for audit in audits:
        if not audit:
            continue
        issues.extend(item for item in audit.get("issues", []) if isinstance(item, dict))
        if audit.get("detector_status") not in {None, "complete"}:
            detector_status = str(audit.get("detector_status"))
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


def persist_audit(agent: Any, node: Any) -> Path | None:
    audit = getattr(node, "leakage_audit", None) or {}
    digest = str(audit.get("code_sha256") or code_sha256(getattr(node, "code", "")))
    registry_dir = registry_dir_for_agent(agent)
    if registry_dir is None:
        return None
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / f"{digest}.json"
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
    if occurrence not in occurrences:
        occurrences.append(occurrence)
    payload = {
        "schema": "mlevolve_leakage_registry_record_v1",
        "code_sha256": digest,
        "audit": audit,
        "occurrences": occurrences,
    }
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
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
        return payload.get("audit") if isinstance(payload, dict) else None
    except Exception:
        return {
            "schema": AUDIT_SCHEMA,
            "status": "audit_unavailable",
            "hard_block": False,
            "paper_grade_eligible": False,
            "memory_disposition": "negative_only",
            "metric_disposition": "unverified",
            "issues": [],
        }
