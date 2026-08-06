#!/usr/bin/env python3
"""Distill evidence-bound L1 recipes, L2 tactics, and reusable L3 repairs.

L1/L2 admission starts from strict-clean successful RunNodes.  L3 admission is
separate and starts from a complete, genuinely executed
failure -> repair -> clean-success RunForest Transition.  A single clean repair
is enough for full L3 admission; repeated success is retained only as audit
metadata and a small ranking tie-break.  It never creates an evidence tier.
Infrastructure incidents and one-off path/name/syntax
fixes are rejected before the teacher sees them.

The teacher model summarizes already-admitted evidence.  It cannot promote an
ineligible RunNode/Transition, change task identity, or invent evidence IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA = "mlevolve_recipe_sop_bundle_v1"
REPORT_SCHEMA = "mlevolve_recipe_sop_distillation_report_v1"
EVIDENCE_SCHEMA = "mlevolve_recipe_distillation_evidence_v1"
TASK_DIRECTIONS = {
    "aerial-cactus-identification": "maximize",
    "leaf-classification": "minimize",
    "denoising-dirty-documents": "minimize",
    "new-york-city-taxi-fare-prediction": "minimize",
    "spooky-author-identification": "minimize",
}
TASK_DOMAINS = {
    "aerial-cactus-identification": "image_binary_classification",
    "leaf-classification": "multimodal_multiclass_classification",
    "denoising-dirty-documents": "image_restoration",
    "new-york-city-taxi-fare-prediction": "tabular_regression",
    "spooky-author-identification": "text_multiclass_classification",
}
TASK_TYPES = {
    "aerial-cactus-identification": "vision",
    "leaf-classification": "multimodal",
    "denoising-dirty-documents": "vision",
    "new-york-city-taxi-fare-prediction": "tabular",
    "spooky-author-identification": "nlp",
}
RECIPE_SECTIONS = (
    "data_validation",
    "split_validation",
    "feature_representation",
    "model_stack",
    "training_protocol",
    "oof_protocol",
    "ensemble_calibration",
    "final_refit_inference",
    "failure_boundaries",
)
REPAIR_TRIVIA = re.compile(
    r"\b(?:nameerror|filenotfound|undefined|not defined|wrong path|file path|"
    r"api parameter|deprecated|import error|syntax error|collate|dataloader error|"
    r"submission path|variable name|function before|pip install)\b",
    flags=re.I,
)
INFRASTRUCTURE_FAILURE = re.compile(
    r"\b(?:permission denied|cache directory|torch[ ._-]*hub cache|file not found|"
    r"filenotfounderror|no such file or directory|missing (?:file|checkpoint)|"
    r"pod (?:is )?pending|unschedulable|imagepullbackoff|api (?:request )?timeout|"
    r"timed out while (?:calling|connecting)|connection (?:reset|refused)|"
    r"temporary failure in name resolution|rate limit|service unavailable|"
    r"shared memory|/dev/shm|bus error|no space left on device)\b",
    flags=re.I,
)
ONE_OFF_CODE_FAILURE = re.compile(
    r"\b(?:nameerror|syntaxerror|indentationerror|unboundlocalerror|"
    r"referenced before assignment|not defined|undefined variable|misspell(?:ed|ing)|"
    r"typo|hardcoded (?:local )?path|wrong (?:file )?path|duplicate code block|"
    r"merge conflict|forgot to (?:define|import)|importerror|modulenotfounderror)\b",
    flags=re.I,
)
REUSABLE_FAILURE_MARKER = re.compile(
    r"\b(?:[a-z][a-z0-9_]*(?:error|exception)|mismatch|invalid|unsupported|"
    r"data leak(?:age)?|contaminat(?:ion|ed)|continuous (?:format|label)|"
    r"collate|dataloader|tensor|device|index|shape|dtype|metric|roc[ _-]*auc|"
    r"cross[ _-]*validation|fold|scaler|vectorizer|pca|mixup|cutmix|patch)\b",
    flags=re.I,
)
L3_RUNTIME_STAGES = {
    "data_loading",
    "preprocessing",
    "split_validation",
    "feature_extraction",
    "model_forward",
    "training",
    "training_metric",
    "validation",
    "oof",
    "inference",
    "submission",
}


def canonical_task(value: object) -> str:
    task = str(value or "").strip()
    while task.startswith("full-"):
        task = task[len("full-") :]
    return task


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def strict_recipe_eligible(node: Mapping[str, Any]) -> bool:
    audit = node.get("leakage_audit")
    return bool(
        node.get("type") == "RunNode"
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and _finite_number(node.get("metric"))
        and isinstance(audit, Mapping)
        and audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
    )


def _node_code_sha256(node: Mapping[str, Any]) -> str:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return str(node.get("code_sha256") or audit.get("code_sha256") or "")


def _failure_text(node: Mapping[str, Any]) -> str:
    return " ".join(
        _trim(node.get(key), 1800)
        for key in ("analysis", "terminal_excerpt", "plan", "text")
        if node.get(key)
    )


def _repair_text(
    transition: Mapping[str, Any], child: Mapping[str, Any]
) -> str:
    return " ".join(
        _trim(value, 2200)
        for value in (
            child.get("plan"),
            child.get("code_summary"),
            transition.get("text"),
        )
        if value
    )


def repair_transition_eligibility(
    transition: Mapping[str, Any],
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    allowed_tasks: set[str],
) -> tuple[bool, str]:
    """Deterministic L3 admission; the teacher cannot override this gate."""

    if transition.get("type") != "Transition":
        return False, "not_transition"
    if (
        transition.get("outcome") != "debug_fixed"
        or transition.get("parent_buggy") is not True
        or transition.get("child_buggy") is not False
        or "debug" not in str(transition.get("stage_pair") or "")
    ):
        return False, "not_complete_debug_fixed_transition"
    task = canonical_task(transition.get("task"))
    if task not in allowed_tasks:
        return False, "task_not_allowed"
    parent_id = str(transition.get("parent_node_id") or "")
    child_id = str(transition.get("child_node_id") or "")
    parent = nodes_by_id.get(parent_id)
    child = nodes_by_id.get(child_id)
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        return False, "missing_parent_or_child"
    if parent.get("is_buggy") is not True:
        return False, "parent_not_observed_buggy"
    if not strict_recipe_eligible(child):
        return False, "child_not_strict_clean_success"
    if len(_node_code_sha256(parent)) != 64 or len(_node_code_sha256(child)) != 64:
        return False, "missing_before_or_after_code_hash"
    failure = _failure_text(parent)
    if INFRASTRUCTURE_FAILURE.search(failure):
        return False, "infrastructure_failure"
    if ONE_OFF_CODE_FAILURE.search(failure):
        return False, "one_off_code_failure"
    if not failure or not REUSABLE_FAILURE_MARKER.search(failure):
        return False, "no_explicit_reusable_failure_signature"
    repair = _repair_text(transition, child)
    if not repair:
        return False, "missing_repair_action"
    return True, "one_clean_failure_repair_success_transition"


def _repair_evidence_row(
    transition: Mapping[str, Any],
    *,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    source_path: Path,
    source_sha256: str,
    cohort: str,
) -> dict[str, Any]:
    task = canonical_task(transition.get("task"))
    parent_id = str(transition["parent_node_id"])
    child_id = str(transition["child_node_id"])
    parent = nodes_by_id[parent_id]
    child = nodes_by_id[child_id]
    return {
        "transition_id": str(transition.get("id") or ""),
        "task_id": task,
        "task_family": TASK_DOMAINS[task],
        "task_type": TASK_TYPES[task],
        "run_id": str(transition.get("run_id") or transition.get("run_short_id") or ""),
        "stage_pair": str(transition.get("stage_pair") or ""),
        "failure_node_id": parent_id,
        "failure_node_code_sha256": _node_code_sha256(parent),
        "failure_text": _trim(_failure_text(parent), 3000),
        "successful_node_id": child_id,
        "successful_node_code_sha256": _node_code_sha256(child),
        "successful_metric": float(child["metric"]),
        "successful_metric_direction": TASK_DIRECTIONS[task],
        "successful_execution_summary": _trim(
            child.get("analysis") or child.get("terminal_excerpt"), 1800
        ),
        "repair_action_text": _trim(_repair_text(transition, child), 3200),
        "audit_status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "rank_eligible": True,
        "source_cohort": cohort,
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_sha256,
    }


def load_graph_repair_evidence(
    path: Path, *, allowed_tasks: set[str], cohort: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    if not isinstance(nodes, list):
        raise ValueError(f"Graph has no node inventory: {path}")
    nodes_by_id = {
        str(node.get("id") or ""): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("id")
    }
    source_sha = sha256_file(path)
    admitted: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen = 0
    for raw in nodes:
        if not isinstance(raw, Mapping) or raw.get("type") != "Transition":
            continue
        if raw.get("outcome") != "debug_fixed":
            continue
        task = canonical_task(raw.get("task"))
        if task not in allowed_tasks:
            continue
        seen += 1
        eligible, reason = repair_transition_eligibility(
            raw,
            nodes_by_id=nodes_by_id,
            allowed_tasks=allowed_tasks,
        )
        if not eligible:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        admitted.append(
            _repair_evidence_row(
                raw,
                nodes_by_id=nodes_by_id,
                source_path=path,
                source_sha256=source_sha,
                cohort=cohort,
            )
        )
    return admitted, {
        "path": str(path),
        "sha256": source_sha,
        "cohort": cohort,
        "debug_fixed_transition_count": seen,
        "strict_l3_eligible_count": len(admitted),
        "l3_rejection_counts": dict(sorted(rejected.items())),
    }


def _trim(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _signature(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return " ".join(tokens[:120])


def _evidence_row(
    node: Mapping[str, Any], *, source_path: Path, source_sha256: str, cohort: str
) -> dict[str, Any]:
    task = canonical_task(node.get("task"))
    audit = node["leakage_audit"]
    return {
        "node_id": str(node.get("id") or ""),
        "task_id": task,
        "task_domain": TASK_DOMAINS[task],
        "run_id": str(node.get("run_id") or node.get("run_short_id") or ""),
        "stage": str(node.get("stage") or ""),
        "step": node.get("step"),
        "metric": float(node["metric"]),
        "metric_direction": TASK_DIRECTIONS[task],
        "metric_improvement": (
            float(node["metric_improvement"])
            if _finite_number(node.get("metric_improvement"))
            else None
        ),
        "plan": _trim(node.get("plan"), 2400),
        "code_summary": _trim(node.get("code_summary") or node.get("text"), 2200),
        "audit_status": str(audit.get("status")),
        "memory_disposition": str(audit.get("memory_disposition")),
        "paper_grade_eligible": bool(audit.get("paper_grade_eligible")),
        "rank_eligible": bool(audit.get("rank_eligible")),
        "code_sha256": str(audit.get("code_sha256") or ""),
        "source_cohort": cohort,
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_sha256,
    }


def load_graph_evidence(
    path: Path, *, allowed_tasks: set[str], cohort: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    if not isinstance(nodes, list):
        raise ValueError(f"Graph has no node inventory: {path}")
    source_sha = sha256_file(path)
    eligible: list[dict[str, Any]] = []
    task_seen: dict[str, int] = {}
    task_admitted: dict[str, int] = {}
    for raw in nodes:
        if not isinstance(raw, Mapping) or raw.get("type") != "RunNode":
            continue
        task = canonical_task(raw.get("task"))
        if task not in allowed_tasks:
            continue
        task_seen[task] = task_seen.get(task, 0) + 1
        if not strict_recipe_eligible(raw):
            continue
        eligible.append(
            _evidence_row(
                raw, source_path=path, source_sha256=source_sha, cohort=cohort
            )
        )
        task_admitted[task] = task_admitted.get(task, 0) + 1
    return eligible, {
        "path": str(path),
        "sha256": source_sha,
        "cohort": cohort,
        "task_run_node_counts": dict(sorted(task_seen.items())),
        "task_strict_recipe_eligible_counts": dict(sorted(task_admitted.items())),
    }


def load_incremental_evidence(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], None
    path = path.resolve(strict=True)
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("records") if isinstance(doc, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("Incremental evidence must contain a records list")
    admitted = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task = canonical_task(row.get("task_id") or row.get("task"))
        audit = row.get("leakage_audit")
        normalized = {
            "type": "RunNode",
            "id": row.get("node_id") or row.get("id"),
            "task": task,
            "run_id": row.get("run_id"),
            "stage": row.get("stage"),
            "step": row.get("step"),
            "metric": row.get("metric"),
            "metric_improvement": row.get("metric_improvement"),
            "plan": row.get("plan"),
            "code_summary": row.get("code_summary"),
            "is_buggy": row.get("is_buggy"),
            "is_valid": row.get("is_valid"),
            "leakage_audit": audit,
        }
        if task not in TASK_DIRECTIONS or not strict_recipe_eligible(normalized):
            continue
        admitted.append(
            _evidence_row(
                normalized,
                source_path=path,
                source_sha256=sha256_file(path),
                cohort=str(row.get("source_cohort") or "post_freeze_incremental"),
            )
        )
    return admitted, {
        "path": str(path),
        "sha256": sha256_file(path),
        "cohort": "post_freeze_incremental",
        "record_count": len(rows),
        "strict_recipe_eligible_count": len(admitted),
    }


def select_representative_evidence(
    rows: Iterable[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if not rows:
        return []
    direction = str(rows[0]["metric_direction"])
    metric_sorted = sorted(
        rows,
        key=lambda row: (
            -float(row["metric"]) if direction == "maximize" else float(row["metric"]),
            str(row["node_id"]),
        ),
    )
    improvement_sorted = sorted(
        (row for row in rows if _finite_number(row.get("metric_improvement"))),
        key=lambda row: (-float(row["metric_improvement"]), str(row["node_id"])),
    )
    selected: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_summaries: set[str] = set()
    seen_runs: set[str] = set()

    def add(row: dict[str, Any], *, require_new_run: bool = False) -> None:
        if len(selected) >= limit or row["node_id"] in seen_nodes:
            return
        signature = _signature(str(row.get("code_summary") or row.get("plan") or ""))
        if not signature or signature in seen_summaries:
            return
        if require_new_run and row["run_id"] in seen_runs:
            return
        selected.append(row)
        seen_nodes.add(str(row["node_id"]))
        seen_summaries.add(signature)
        seen_runs.add(str(row["run_id"]))

    for row in metric_sorted:
        add(row, require_new_run=True)
    for row in improvement_sorted:
        add(row)
    for row in metric_sorted:
        add(row)
    return selected


SYSTEM_PROMPT = """You are the Recipe Distillation Agent for an automated ML system.

Your job is to consolidate CLEAN successful execution evidence into complete training
recipes, not into debugging trivia.  The caller has already enforced source admission;
you may cite only supplied node_id values and may not decide that another source is clean.

Emit two levels:
1. L1 recipes: complete task-level pipelines suitable for Draft.  Every L1 recipe must
   specify data validation, split/validation design, features or representation, model
   stack, training protocol, OOF/CV policy, ensemble/calibration policy, final refit and
   inference, and important failure boundaries.  Use "not evidenced / single holdout"
   when OOF or ensembling is unsupported.  Do not invent OOF.
2. L2 tactics: reusable model-design or training details tied to one or more L1 method
   families.  They are not standalone recipes.

Forbidden SOP content: file paths, submission paths, NameError fixes, API/deprecation
warnings, undefined variables, import fixes, syntax fixes, DataLoader/collate incidents,
or other one-off repairs.  Those belong to RunForest Debug transitions.

Consolidate duplicates.  Preserve genuinely distinct method families.  Do not combine
different tasks.  Output one JSON object and no prose.
"""

L3_SYSTEM_PROMPT = """You are the L3 Repair Distillation Agent for an automated ML system.

The caller has already admitted only real failure -> repair -> clean-success
RunForest transitions.  You may summarize and consolidate supplied evidence, but
you may not invent or change transition IDs, task identity, failed/successful node
IDs, or execution outcomes.

Emit reusable L3 repairs only.  One successful clean transition is sufficient for
full admission.  Do not label repairs provisional or confirmed: repeated support
is audit metadata only and must not create a reliability tier or broaden applicability.

Each repair must define a stable normalized failure signature, root cause, concrete
repair steps, method family, and runtime stage.  Keep it scoped to the supplied
task and task type.  Never emit infrastructure incidents (node/cache permission,
missing files, Pod scheduling, service/API timeouts, shared-memory/node incidents)
or one-off spelling, variable-name, import, path, syntax, or script-order fixes.
Output one JSON object and no prose.
"""


def build_user_prompt(task: str, evidence: list[dict[str, Any]]) -> str:
    compact = [
        {
            "node_id": row["node_id"],
            "run_id": row["run_id"],
            "stage": row["stage"],
            "metric": row["metric"],
            "metric_direction": row["metric_direction"],
            "metric_improvement": row["metric_improvement"],
            "plan": row["plan"],
            "code_summary": row["code_summary"],
            "source_cohort": row["source_cohort"],
        }
        for row in evidence
    ]
    schema = {
        "recipes": [
            {
                "title": "complete method title",
                "method_family": "stable_snake_case_family",
                "when_to_use": "task conditions and resource assumptions",
                "pipeline": {section: "specific evidenced instruction" for section in RECIPE_SECTIONS},
                "source_node_ids": ["one or more supplied node_id values"],
            }
        ],
        "tactics": [
            {
                "title": "model-design or training tactic",
                "tactic_kind": "feature|model_design|training_protocol|validation_protocol|ensemble",
                "parent_method_families": ["a method_family emitted above"],
                "instruction": "specific evidenced instruction",
                "when_to_use": "condition",
                "source_node_ids": ["supplied node_id values"],
            }
        ],
    }
    post_freeze_ids = [
        row["node_id"]
        for row in evidence
        if str(row.get("source_cohort") or "").startswith("post_freeze")
    ]
    incremental_requirement = ""
    if post_freeze_ids:
        incremental_requirement = (
            "\nAt least one L1 recipe MUST cite at least one of these admitted post-freeze "
            "source_node_ids as corroborating evidence when it matches the same method family; "
            "do not create a duplicate family only to satisfy this rule: "
            + json.dumps(post_freeze_ids, ensure_ascii=False)
            + "\n"
        )
    return (
        f"Task: {task}\nDomain: {TASK_DOMAINS[task]}\nMetric direction: "
        f"{TASK_DIRECTIONS[task]}\n\nCreate 4-6 distinct complete L1 recipes and 3-6 "
        "L2 tactics.  Keep the number of tactics no greater than the number of recipes. "
        "Prefer robust OOF and final-refit designs only when the evidence supports them.\n"
        f"{incremental_requirement}\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Admitted evidence:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def select_repair_evidence(
    rows: Iterable[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["run_id"]), str(row["transition_id"])),
    )
    selected: list[dict[str, Any]] = []
    seen_transitions: set[str] = set()
    seen_failure_fingerprints: set[str] = set()
    # First retain distinct failure descriptions, then backfill.  This is not
    # an eligibility decision; it only keeps the teacher prompt bounded.
    for require_distinct in (True, False):
        for row in rows:
            transition_id = str(row["transition_id"])
            if transition_id in seen_transitions:
                continue
            fingerprint = _signature(str(row.get("failure_text") or ""))
            if require_distinct and fingerprint in seen_failure_fingerprints:
                continue
            selected.append(row)
            seen_transitions.add(transition_id)
            seen_failure_fingerprints.add(fingerprint)
            if len(selected) >= limit:
                return selected
    return selected


def build_repair_user_prompt(task: str, evidence: list[dict[str, Any]]) -> str:
    compact = [
        {
            "transition_id": row["transition_id"],
            "run_id": row["run_id"],
            "stage_pair": row["stage_pair"],
            "failure_node_id": row["failure_node_id"],
            "successful_node_id": row["successful_node_id"],
            "failure_text": _trim(row["failure_text"], 1100),
            "repair_action_text": _trim(row["repair_action_text"], 1300),
            "successful_execution_summary": _trim(
                row["successful_execution_summary"], 650
            ),
            "successful_metric": row["successful_metric"],
        }
        for row in evidence
    ]
    schema = {
        "repairs": [
            {
                "title": "specific reusable repair title",
                "signature_id": "stable/slash_separated/root_cause_id",
                "failure_pattern": "normalized semantic failure condition",
                "root_cause": "why the failure occurs",
                "repair_steps": ["concrete repair step"],
                "when_to_use": "observable preconditions",
                "method_family": "stable_snake_case_family_or_general",
                "runtime_stage": sorted(L3_RUNTIME_STAGES),
                "source_transition_ids": ["one or more supplied transition_id values"],
            }
        ]
    }
    return (
        f"Task: {task}\nTask family: {TASK_DOMAINS[task]}\nTask type: {TASK_TYPES[task]}\n"
        "Create only reusable L3 repairs supported by the supplied transitions. "
        "Prefer 3-10 high-quality distinct repairs; output fewer when evidence is weak.\n\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Admitted repair transitions:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Teacher response did not contain a JSON object")


def _call_teacher_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_message: str,
    user_message: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 12000,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=canonical_bytes(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=600) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return _extract_json(str(content or "")), {
                "model": str(raw.get("model") or model),
                "usage": raw.get("usage") or {},
                "response_id": str(raw.get("id") or ""),
            }
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"teacher HTTP {exc.code}: {detail[:1000]}")
        except (URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = exc
        if attempt < 3:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Teacher call failed: {last_error}")


def call_teacher(
    *, base_url: str, api_key: str, model: str, task: str, evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _call_teacher_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_message=SYSTEM_PROMPT,
        user_message=build_user_prompt(task, evidence),
    )


def call_repair_teacher(
    *, base_url: str, api_key: str, model: str, task: str, evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _call_teacher_json(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_message=L3_SYSTEM_PROMPT,
        user_message=build_repair_user_prompt(task, evidence),
    )


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_task_output(
    *, task: str, raw: Mapping[str, Any], allowed_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    recipes: list[dict[str, Any]] = []
    tactics: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for item in raw.get("recipes") or []:
        if not isinstance(item, Mapping):
            continue
        title = re.sub(
            r"^(?:confirmed|provisional)\s*:\s*",
            "",
            _clean_text(item.get("title")),
            flags=re.I,
        )
        family = re.sub(r"[^a-z0-9]+", "_", _clean_text(item.get("method_family")).lower()).strip("_")
        pipeline = item.get("pipeline") if isinstance(item.get("pipeline"), Mapping) else {}
        source_ids = list(
            dict.fromkeys(
                str(value) for value in (item.get("source_node_ids") or []) if str(value) in allowed_ids
            )
        )
        missing = [section for section in RECIPE_SECTIONS if not _clean_text(pipeline.get(section))]
        reason = ""
        if not title or not family:
            reason = "missing_title_or_method_family"
        elif REPAIR_TRIVIA.search(title):
            reason = "repair_trivia_not_recipe"
        elif not source_ids:
            reason = "no_valid_source_node_ids"
        elif missing:
            reason = "missing_pipeline_sections:" + ",".join(missing)
        elif title.lower() in seen_titles:
            reason = "duplicate_title"
        if reason:
            rejected.append({"kind": "recipe", "title": title, "reason": reason})
            continue
        seen_titles.add(title.lower())
        recipes.append(
            {
                "id": "",
                "type": "SOP",
                "abstraction_level": "L1_recipe",
                "sop_kind": "model_strategy_recipe",
                "task_id": task,
                "task_domain": TASK_DOMAINS[task],
                "decision_stages": ["draft"],
                "title": title,
                "method_family": family,
                "when_to_use": _clean_text(item.get("when_to_use")),
                "pipeline": {section: _clean_text(pipeline[section]) for section in RECIPE_SECTIONS},
                "source_node_ids": source_ids,
                "clean_supporting_node_ids": source_ids,
                "source_admission": "all_sources_strict_clean_positive_eligible",
                "recipe_complete": True,
            }
        )

    families = {row["method_family"] for row in recipes}
    seen_titles.clear()
    for item in raw.get("tactics") or []:
        if not isinstance(item, Mapping):
            continue
        title = _clean_text(item.get("title"))
        parent_families = [
            str(value) for value in (item.get("parent_method_families") or []) if str(value) in families
        ]
        source_ids = list(
            dict.fromkeys(
                str(value) for value in (item.get("source_node_ids") or []) if str(value) in allowed_ids
            )
        )
        instruction = _clean_text(item.get("instruction"))
        reason = ""
        if not title or not instruction:
            reason = "missing_title_or_instruction"
        elif REPAIR_TRIVIA.search(title + " " + instruction):
            reason = "repair_trivia_not_tactic"
        elif not parent_families:
            reason = "no_valid_parent_method_family"
        elif not source_ids:
            reason = "no_valid_source_node_ids"
        elif title.lower() in seen_titles:
            reason = "duplicate_title"
        if reason:
            rejected.append({"kind": "tactic", "title": title, "reason": reason})
            continue
        seen_titles.add(title.lower())
        tactics.append(
            {
                "id": "",
                "type": "SOP",
                "abstraction_level": "L2_tactic",
                "sop_kind": _clean_text(item.get("tactic_kind")) or "training_protocol",
                "task_id": task,
                "task_domain": TASK_DOMAINS[task],
                "decision_stages": ["model_design"],
                "title": title,
                "parent_method_families": list(dict.fromkeys(parent_families)),
                "instruction": instruction,
                "when_to_use": _clean_text(item.get("when_to_use")),
                "source_node_ids": source_ids,
                "clean_supporting_node_ids": source_ids,
                "source_admission": "all_sources_strict_clean_positive_eligible",
            }
        )
    tactics = tactics[: len(recipes)]
    return recipes, tactics, rejected


def normalize_repair_output(
    *,
    task: str,
    raw: Mapping[str, Any],
    evidence_by_transition: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    repairs: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_signatures: set[str] = set()
    for item in raw.get("repairs") or []:
        if not isinstance(item, Mapping):
            continue
        title = re.sub(
            r"^(?:confirmed|provisional)\s*:\s*",
            "",
            _clean_text(item.get("title")),
            flags=re.I,
        )
        signature_id = re.sub(
            r"[^a-z0-9/_-]+",
            "_",
            _clean_text(item.get("signature_id")).lower(),
        ).strip("_/")
        failure_pattern = _clean_text(item.get("failure_pattern"))
        root_cause = _clean_text(item.get("root_cause"))
        repair_steps = [
            _clean_text(value)
            for value in (item.get("repair_steps") or [])
            if _clean_text(value)
        ]
        method_family = re.sub(
            r"[^a-z0-9]+",
            "_",
            _clean_text(item.get("method_family")).lower(),
        ).strip("_") or "general"
        raw_runtime_stages = item.get("runtime_stage")
        if isinstance(raw_runtime_stages, list):
            runtime_stages = list(
                dict.fromkeys(
                    _clean_text(value)
                    for value in raw_runtime_stages
                    if _clean_text(value)
                )
            )
        else:
            runtime_stages = [_clean_text(raw_runtime_stages)] if _clean_text(raw_runtime_stages) else []
        transition_ids = list(
            dict.fromkeys(
                str(value)
                for value in (item.get("source_transition_ids") or [])
                if str(value) in evidence_by_transition
            )
        )
        combined = " ".join(
            [title, failure_pattern, root_cause, *repair_steps]
        )
        reason = ""
        if not title or not signature_id or not failure_pattern or not root_cause:
            reason = "missing_required_repair_fields"
        elif not repair_steps:
            reason = "missing_repair_steps"
        elif not runtime_stages or any(
            value not in L3_RUNTIME_STAGES for value in runtime_stages
        ):
            reason = "invalid_runtime_stage"
        elif not transition_ids:
            reason = "no_valid_source_transition_ids"
        elif INFRASTRUCTURE_FAILURE.search(combined):
            reason = "infrastructure_failure"
        elif ONE_OFF_CODE_FAILURE.search(combined):
            reason = "one_off_code_failure"
        elif signature_id in seen_signatures:
            reason = "duplicate_signature_id"
        if reason:
            rejected.append(
                {
                    "kind": "repair",
                    "title": title,
                    "signature_id": signature_id,
                    "reason": reason,
                }
            )
            continue
        rows = [dict(evidence_by_transition[value]) for value in transition_ids]
        run_ids = list(dict.fromkeys(str(row["run_id"]) for row in rows))
        failure_node_ids = list(
            dict.fromkeys(str(row["failure_node_id"]) for row in rows)
        )
        successful_node_ids = list(
            dict.fromkeys(str(row["successful_node_id"]) for row in rows)
        )
        seen_signatures.add(signature_id)
        repairs.append(
            {
                "id": "",
                "type": "SOP",
                "abstraction_level": "L3_repair",
                "sop_kind": "debug_fix",
                "task_id": task,
                "task_domain": TASK_DOMAINS[task],
                "task_family": TASK_DOMAINS[task],
                "task_type": TASK_TYPES[task],
                "decision_stages": ["debug"],
                "runtime_stage": runtime_stages[0],
                "runtime_stages": runtime_stages,
                "title": title,
                "method_family": method_family,
                "failure_signature": {
                    "id": signature_id,
                    "pattern": failure_pattern,
                    "root_cause": root_cause,
                },
                "repair_action": {
                    "summary": repair_steps[0],
                    "steps": repair_steps,
                },
                "when_to_use": _clean_text(item.get("when_to_use")),
                "supporting_transition_ids": transition_ids,
                "source_transition_ids": transition_ids,
                "failure_node_ids": failure_node_ids,
                "successful_node_ids": successful_node_ids,
                "source_node_ids": list(
                    dict.fromkeys([*failure_node_ids, *successful_node_ids])
                ),
                "distinct_run_ids": run_ids,
                "distinct_run_count": len(run_ids),
                "successful_repair_count": len(rows),
                "evidence_status": "accepted_clean_repair",
                "confidence_prior": 0.60,
                "source_admission": "one_clean_failure_repair_success_transition",
                "infrastructure_failure": False,
                "one_off_code_failure": False,
            }
        )
    return repairs, rejected


def build_artifacts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    all_repair_rows: list[dict[str, Any]] = []
    sources = []
    four_rows, four_source = load_graph_evidence(
        args.fourtask_graph,
        allowed_tasks=set(TASK_DIRECTIONS) - {"spooky-author-identification"},
        cohort="fourtask_graph_v2_frozen",
    )
    spooky_rows, spooky_source = load_graph_evidence(
        args.spooky_graph,
        allowed_tasks={"spooky-author-identification"},
        cohort="spooky_allowlisted_clean_graph",
    )
    incremental_rows, incremental_source = load_incremental_evidence(args.incremental_evidence)
    four_repair_rows, four_repair_source = load_graph_repair_evidence(
        args.fourtask_graph,
        allowed_tasks=set(TASK_DIRECTIONS) - {"spooky-author-identification"},
        cohort="fourtask_graph_v2_frozen",
    )
    spooky_repair_rows, spooky_repair_source = load_graph_repair_evidence(
        args.spooky_graph,
        allowed_tasks={"spooky-author-identification"},
        cohort="spooky_allowlisted_clean_graph",
    )
    all_rows.extend(four_rows)
    all_rows.extend(spooky_rows)
    all_rows.extend(incremental_rows)
    all_repair_rows.extend(four_repair_rows)
    all_repair_rows.extend(spooky_repair_rows)
    four_source.update(
        {
            key: value
            for key, value in four_repair_source.items()
            if key.startswith("debug_")
            or key.startswith("strict_l3_")
            or key.startswith("l3_")
        }
    )
    spooky_source.update(
        {
            key: value
            for key, value in spooky_repair_source.items()
            if key.startswith("debug_")
            or key.startswith("strict_l3_")
            or key.startswith("l3_")
        }
    )
    sources.extend([four_source, spooky_source])
    if incremental_source is not None:
        sources.append(incremental_source)

    unique_by_id: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        unique_by_id[str(row["node_id"])] = row
    admitted = list(unique_by_id.values())
    selected_by_task = {
        task: select_representative_evidence(
            (row for row in admitted if row["task_id"] == task),
            limit=args.max_evidence_per_task,
        )
        for task in TASK_DIRECTIONS
    }
    selected_repairs_by_task = {
        task: select_repair_evidence(
            (row for row in all_repair_rows if row["task_id"] == task),
            limit=args.max_repair_evidence_per_task,
        )
        for task in TASK_DIRECTIONS
    }
    if any(not rows for rows in selected_by_task.values()):
        missing = [task for task, rows in selected_by_task.items() if not rows]
        raise ValueError(f"No admitted evidence for tasks: {missing}")

    evidence_manifest: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "created_at": args.created_at,
        "source_artifacts": sources,
        "strict_admission_rule": (
            "RunNode && !is_buggy && is_valid && finite(metric) && audit.status=clean && "
            "memory_disposition=positive_eligible && paper_grade_eligible && rank_eligible"
        ),
        "strict_l3_admission_rule": (
            "Transition(outcome=debug_fixed,parent_buggy=true,child_buggy=false,stage contains debug) "
            "&& parent observed buggy && child strict-clean successful && before/after code hashes "
            "&& explicit reusable failure marker && !infrastructure && !one-off code/path/name/syntax"
        ),
        "admitted_node_count": len(admitted),
        "admitted_counts_by_task": {
            task: sum(row["task_id"] == task for row in admitted) for task in TASK_DIRECTIONS
        },
        "selected_counts_by_task": {
            task: len(rows) for task, rows in selected_by_task.items()
        },
        "selected_evidence": selected_by_task,
        "admitted_repair_transition_count": len(all_repair_rows),
        "admitted_repair_counts_by_task": {
            task: sum(row["task_id"] == task for row in all_repair_rows)
            for task in TASK_DIRECTIONS
        },
        "selected_repair_counts_by_task": {
            task: len(rows) for task, rows in selected_repairs_by_task.items()
        },
        "selected_repair_evidence": selected_repairs_by_task,
        "manifest_sha256": "",
    }
    evidence_manifest["manifest_sha256"] = payload_hash(evidence_manifest, "manifest_sha256")

    if args.prepare_only:
        return evidence_manifest, {}, {}
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing teacher API key in {args.api_key_env}")

    output_nodes: list[dict[str, Any]] = []
    teacher_calls = []
    rejected = []
    raw_dir = args.output_dir / "raw_responses"
    for task, evidence in selected_by_task.items():
        raw_path = raw_dir / f"{task}.json"
        if raw_path.exists() and task not in set(args.refresh_task):
            print(f"[recipe-distill] reuse {task}: {raw_path}", flush=True)
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            call = {
                "model": args.model,
                "usage": {},
                "response_id": "",
                "resumed_from_frozen_raw_response": True,
            }
        else:
            print(
                f"[recipe-distill] call {task}: evidence={len(evidence)} model={args.model}",
                flush=True,
            )
            raw, call = call_teacher(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                task=task,
                evidence=evidence,
            )
            write_json(raw_path, raw)
        recipes, tactics, task_rejected = normalize_task_output(
            task=task,
            raw=raw,
            allowed_ids={row["node_id"] for row in evidence},
        )
        if len(recipes) < 3:
            raise ValueError(f"Teacher produced fewer than three valid recipes for {task}")
        print(
            f"[recipe-distill] accepted {task}: recipes={len(recipes)} tactics={len(tactics)} "
            f"rejected={len(task_rejected)}",
            flush=True,
        )
        output_nodes.extend(recipes)
        output_nodes.extend(tactics)
        teacher_calls.append(
            {
                "task_id": task,
                "call_kind": "l1_l2_recipe",
                **call,
                "recipe_count": len(recipes),
                "tactic_count": len(tactics),
            }
        )
        rejected.extend({"task_id": task, **row} for row in task_rejected)

    for task, evidence in selected_repairs_by_task.items():
        if not evidence:
            continue
        raw_path = raw_dir / f"{task}.l3.json"
        if raw_path.exists() and not args.refresh_repairs:
            print(f"[recipe-distill] reuse L3 {task}: {raw_path}", flush=True)
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            call = {
                "model": args.model,
                "usage": {},
                "response_id": "",
                "resumed_from_frozen_raw_response": True,
            }
        else:
            print(
                f"[recipe-distill] call L3 {task}: transitions={len(evidence)} model={args.model}",
                flush=True,
            )
            raw, call = call_repair_teacher(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                task=task,
                evidence=evidence,
            )
            write_json(raw_path, raw)
        evidence_by_transition = {
            str(row["transition_id"]): row for row in evidence
        }
        repairs, task_rejected = normalize_repair_output(
            task=task,
            raw=raw,
            evidence_by_transition=evidence_by_transition,
        )
        print(
            f"[recipe-distill] accepted L3 {task}: repairs={len(repairs)} "
            f"rejected={len(task_rejected)}",
            flush=True,
        )
        output_nodes.extend(repairs)
        teacher_calls.append(
            {
                "task_id": task,
                "call_kind": "l3_repair",
                **call,
                "repair_count": len(repairs),
            }
        )
        rejected.extend({"task_id": task, **row} for row in task_rejected)

    counters = {
        task: {"L1_recipe": 0, "L2_tactic": 0, "L3_repair": 0}
        for task in TASK_DIRECTIONS
    }
    for node in output_nodes:
        task = node["task_id"]
        level = node["abstraction_level"]
        counters[task][level] += 1
        prefix = {
            "L1_recipe": "recipe",
            "L2_tactic": "tactic",
            "L3_repair": "repair",
        }[level]
        node["id"] = f"{prefix}::{task}::{counters[task][level]:03d}"

    l1_count = sum(node["abstraction_level"] == "L1_recipe" for node in output_nodes)
    l2_count = sum(node["abstraction_level"] == "L2_tactic" for node in output_nodes)
    l3_count = sum(node["abstraction_level"] == "L3_repair" for node in output_nodes)
    total = len(output_nodes)
    methodology_total = l1_count + l2_count
    l1_ratio = l1_count / methodology_total if methodology_total else 0.0
    if l1_ratio < 0.40:
        raise ValueError(f"Recipe ratio gate failed: {l1_ratio:.3f} < 0.40")
    if any(
        REPAIR_TRIVIA.search(node["title"])
        for node in output_nodes
        if node["abstraction_level"] in {"L1_recipe", "L2_tactic"}
    ):
        raise ValueError("Repair trivia survived L1/L2 SOP normalization")
    post_freeze_selected_ids = {
        row["node_id"]
        for rows in selected_by_task.values()
        for row in rows
        if str(row.get("source_cohort") or "").startswith("post_freeze")
    }
    cited_source_ids = {
        source_id for node in output_nodes for source_id in node["source_node_ids"]
    }
    post_freeze_cited_ids = sorted(post_freeze_selected_ids & cited_source_ids)
    if post_freeze_selected_ids and not post_freeze_cited_ids:
        raise ValueError("No admitted post-freeze evidence was cited by the distilled SOPs")

    bundle: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_version": args.bundle_version,
        "created_at": args.created_at,
        "teacher": {
            "base_url": args.base_url,
            "model_requested": args.model,
            "temperature": 0.2,
            "calls": teacher_calls,
        },
        "evidence_manifest_sha256": evidence_manifest["manifest_sha256"],
        "routing_contract": {
            "draft": "L1_recipe only",
            "model_design": "L2_tactic only after L1 method-family freeze",
            "improve": "RunForest metric_improved Transition only",
            "debug": (
                "L3_repair hard task gate (exact task first; same task type only when exact is absent; "
                "cross-task-type forbidden) then supporting clean failure-to-repair Transition"
            ),
        },
        "nodes": output_nodes,
        "bundle_sha256": "",
    }
    bundle["bundle_sha256"] = payload_hash(bundle, "bundle_sha256")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "bundle_version": args.bundle_version,
        "bundle_sha256": bundle["bundle_sha256"],
        "evidence_manifest_sha256": evidence_manifest["manifest_sha256"],
        "node_count": total,
        "abstraction_counts": {
            "L1_recipe": l1_count,
            "L2_tactic": l2_count,
            "L3_repair": l3_count,
        },
        "abstraction_ratios": {
            "L1_recipe": l1_count / total if total else 0.0,
            "L2_tactic": l2_count / total if total else 0.0,
            "L3_repair": l3_count / total if total else 0.0,
        },
        "complete_recipe_count": sum(bool(node.get("recipe_complete")) for node in output_nodes),
        "complete_recipe_ratio": l1_ratio,
        "task_counts": counters,
        "repair_trivia_in_l1_l2_count": 0,
        "accepted_clean_repair_l3_count": sum(
            node.get("evidence_status") == "accepted_clean_repair"
            for node in output_nodes
        ),
        "l3_evidence_tiering_enabled": False,
        "post_freeze_selected_evidence_count": len(post_freeze_selected_ids),
        "post_freeze_cited_evidence_count": len(post_freeze_cited_ids),
        "post_freeze_cited_evidence_ids": post_freeze_cited_ids,
        "rejected_output_count": len(rejected),
        "rejected_outputs": rejected,
        "quality_gates": {
            "l1_recipe_ratio_at_least_0_40": l1_ratio >= 0.40,
            "all_l1_sections_complete": all(
                all(_clean_text(node["pipeline"].get(section)) for section in RECIPE_SECTIONS)
                for node in output_nodes
                if node["abstraction_level"] == "L1_recipe"
            ),
            "all_source_ids_admitted": True,
            "no_repair_trivia_in_l1_l2": True,
            "all_l3_have_complete_evidence_bindings": all(
                node.get("failure_signature", {}).get("id")
                and node.get("repair_action", {}).get("steps")
                and node.get("supporting_transition_ids")
                and node.get("failure_node_ids")
                and node.get("successful_node_ids")
                and node.get("distinct_run_ids")
                and node.get("task_family")
                and node.get("method_family")
                and node.get("runtime_stage")
                for node in output_nodes
                if node["abstraction_level"] == "L3_repair"
            ),
            "all_l3_exclude_infrastructure_and_one_off_code_failures": all(
                node.get("infrastructure_failure") is False
                and node.get("one_off_code_failure") is False
                for node in output_nodes
                if node["abstraction_level"] == "L3_repair"
            ),
            "task_isolation_preserved": True,
            "post_freeze_incremental_evidence_cited": bool(post_freeze_cited_ids)
            if post_freeze_selected_ids
            else True,
        },
        "report_sha256": "",
    }
    report["report_sha256"] = payload_hash(report, "report_sha256")
    return evidence_manifest, bundle, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourtask-graph", type=Path, required=True)
    parser.add_argument("--spooky-graph", type=Path, required=True)
    parser.add_argument("--incremental-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--bundle-version", default="recipe-sop-v1")
    parser.add_argument("--base-url", default="http://localhost:56542/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--api-key-env", default="RECIPE_DISTILL_API_KEY")
    parser.add_argument("--max-evidence-per-task", type=int, default=32)
    parser.add_argument("--max-repair-evidence-per-task", type=int, default=24)
    parser.add_argument("--refresh-task", action="append", default=[])
    parser.add_argument("--refresh-repairs", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    evidence, bundle, report = build_artifacts(args)
    write_json(args.output_dir / "evidence_manifest.json", evidence)
    if bundle:
        write_json(args.output_dir / "recipe_sops.json", bundle)
        write_json(args.output_dir / "distillation_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "schema": evidence["schema"],
                    "admitted_node_count": evidence["admitted_node_count"],
                    "admitted_counts_by_task": evidence["admitted_counts_by_task"],
                    "selected_counts_by_task": evidence["selected_counts_by_task"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
