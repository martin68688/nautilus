"""Agent-owned semantic protocol review with bounded method-preserving repair.

This is intentionally separate from Host Protocol Preflight.  The Agent reads
the actual program entrypoint, may repair protocol/data-flow defects, and then
returns the final source to the normal execution path.  In the End2End shadow
profile, unresolved findings are retained as observations and never converted
into a Host receipt-based admission veto.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

from llm import FunctionSpec, query
from agents import leakage_audit
from agents.coder.diff_coder import SearchReplacePatcher
from agents.prompts.impl_guideline import get_host_protocol_contract_from_agent
from utils.response import wrap_code


AGENT_PROTOCOL_REVIEW_SCHEMA = "mlevolve_agent_semantic_protocol_review_v1"

AGENT_PROTOCOL_REVIEW_SPEC = FunctionSpec(
    name="review_actual_training_protocol",
    description=(
        "Review the actual executable training path and, when necessary, "
        "return a narrow method-preserving SEARCH/REPLACE repair."
    ),
    json_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["clean", "revise", "uncertain"],
            },
            "reason": {"type": "string", "maxLength": 3000},
            "actual_entrypoint": {"type": "string", "maxLength": 500},
            "findings": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 1000},
            },
            "revised_code": {
                "type": "string",
                "description": (
                    "Empty unless status=revise. Otherwise one or more raw "
                    "SEARCH/REPLACE blocks; never a full rewritten program."
                ),
            },
        },
        "required": [
            "status",
            "reason",
            "actual_entrypoint",
            "findings",
            "revised_code",
        ],
    },
)


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _host_sdk_static_observations(source: str) -> list[dict[str, Any]]:
    """Surface narrow SDK misuse facts to the repair Agent without gating execution."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    split_names = {"views", "split"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get_split"
        ):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        split_names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )

    def is_handle(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"train", "validation", "inference"}
            and isinstance(node.value, ast.Name)
            and node.value.id in split_names
        )

    iterated_parameters: dict[str, tuple[list[str], set[str]]] = {}
    for definition in ast.walk(tree):
        if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = [
            argument.arg
            for argument in (
                *definition.args.posonlyargs,
                *definition.args.args,
            )
        ]
        parameters = set(positional) | {
            argument.arg for argument in definition.args.kwonlyargs
        }
        iterated: set[str] = set()
        for child in ast.walk(definition):
            iterator = None
            if isinstance(child, (ast.For, ast.AsyncFor, ast.comprehension)):
                iterator = child.iter
            if isinstance(iterator, ast.Name) and iterator.id in parameters:
                iterated.add(iterator.id)
        if iterated:
            iterated_parameters[definition.name] = (positional, iterated)

    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(code: str, node: ast.AST, message: str) -> None:
        key = (code, int(getattr(node, "lineno", 0) or 0))
        if key in seen:
            return
        seen.add(key)
        observations.append(
            {"code": code, "line": key[1], "message": message}
        )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in iterated_parameters
        ):
            positional, iterated = iterated_parameters[node.func.id]
            supplied = {
                positional[index]: value
                for index, value in enumerate(node.args)
                if index < len(positional)
            }
            supplied.update(
                {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            for parameter in sorted(iterated):
                value = supplied.get(parameter)
                if isinstance(value, ast.Constant) and value.value is None:
                    add(
                        "none_passed_to_iterated_parameter",
                        node,
                        f"{node.func.id} iterates parameter {parameter!r}, but this call passes None; repair the helper/call data flow before execution.",
                    )

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"iter", "len", "list", "set", "tuple"}
            and any(is_handle(arg) for arg in node.args)
        ):
            add(
                "data_view_handle_direct_use",
                node,
                "DataViewHandle is a capability token and cannot be iterated, sized, or materialized directly; consume the rows yielded by the matching session scope.",
            )
        elif isinstance(node, ast.For) and is_handle(node.iter):
            add(
                "data_view_handle_direct_use",
                node,
                "A DataViewHandle cannot be used as a for-loop iterable; iterate only the rows yielded by fit_scope, prediction_scope, or inference_scope.",
            )
        elif isinstance(node, ast.comprehension) and is_handle(node.iter):
            add(
                "data_view_handle_direct_use",
                node,
                "A DataViewHandle cannot be used as a comprehension iterable; iterate only scope-yielded rows.",
            )

        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        prediction_context = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "prediction_scope"
            for item in node.items
        )
        if not prediction_context:
            continue
        for statement in node.body:
            for child in ast.walk(statement):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "evaluate_internal"
                ):
                    add(
                        "evaluate_inside_prediction_scope",
                        child,
                        "Exit prediction_scope before calling session.evaluate_internal so the Host can seal prediction evidence first.",
                    )

    return observations


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prompt(agent: Any, source: str, repair_attempt: int) -> dict[str, Any]:
    contract = get_host_protocol_contract_from_agent(agent)
    static_observations = _host_sdk_static_observations(source)
    return {
        "Role": (
            "You are the semantic protocol reviewer immediately before GPU execution. "
            "Review the code that really runs, not comments, claimed receipts, or a dummy "
            "candidate(session) function that the __main__ path never calls."
        ),
        "Task": str(agent.task_desc),
        "Frozen Host protocol": json.dumps(
            contract, sort_keys=True, ensure_ascii=False, indent=2
        ),
        "Repair attempt already used": str(repair_attempt),
        "Host SDK static observations": static_observations,
        "Required proof": [
            "Identify the actual __main__ or top-level executable path.",
            "The actual fit must consume train_rows yielded by session.fit_scope on views.train; it must not reopen ./input/train.csv or rediscover/resplit the public training set.",
            "Internal-validation predictions must be produced from rows yielded by session.prediction_scope on views.validation, in Host row order.",
            "views.train, views.validation, and views.inference are DataViewHandle capability tokens, not iterables; never call list/len/iter on them or loop over them directly.",
            "Exit prediction_scope before session.evaluate_internal; evaluation requires prediction evidence sealed when that context exits.",
            "The reported score must come from session.evaluate_internal on that same validation view and prediction array.",
            "After every repair, recheck helper signatures and call sites: never pass None to a parameter that the helper iterates or otherwise dereferences.",
            "session.freeze_selection must happen after evaluation/model choice and before final inference.",
            "Final submission inference must consume views.inference after selection freeze.",
            "Import/module initialization must not train, read task data, create working artifacts, or perform other candidate side effects.",
        ],
        "Repair boundary": [
            "Preserve the model family, architecture, losses, optimizer, hyperparameters, epochs, folds, augmentation, ensemble, and feature design.",
            "Change only data-view plumbing, evaluation/freeze ordering, entrypoint structure, or import-time side effects required by the protocol.",
            "If status=revise, return exact raw <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks. Never return the full program and never use markdown fences.",
            "Use status=clean only with positive data-flow evidence. Use uncertain when the path cannot be proven and no safe narrow patch can be written.",
            "If Host SDK static observations is non-empty, status=clean is invalid: return status=revise with a narrow patch that clears every observation.",
        ],
        "Actual complete candidate source": wrap_code(source),
    }


def run(agent: Any, node: Any) -> tuple[str, dict[str, Any]]:
    """Return reviewed source and a durable non-blocking semantic trace."""

    settings = getattr(getattr(agent, "acfg", None), "protocol_preflight", None)
    enabled = bool(
        settings is not None
        and getattr(settings, "agent_semantic_review_enabled", False)
    )
    source = str(node.code or "")
    initial_hash = _sha256(source)
    if not enabled:
        return source, {
            "schema": AGENT_PROTOCOL_REVIEW_SCHEMA,
            "enabled": False,
            "initial_code_sha256": initial_hash,
            "final_code_sha256": initial_hash,
            "final_status": "disabled",
            "execution_disposition": "continue",
            "attempts": [],
        }

    max_repairs = max(
        0, int(getattr(settings, "agent_semantic_max_repair_attempts", 2))
    )
    max_reviews = max(
        max_repairs + 1,
        int(
            getattr(
                settings,
                "agent_semantic_max_review_attempts",
                max_repairs + 3,
            )
        ),
    )
    preservation = leakage_audit.build_repair_preservation_contract(source)
    attempts: list[dict[str, Any]] = []
    repairs_applied = 0
    final_status = "uncertain"
    final_reason = "review did not return a conclusive result"

    for review_index in range(max_reviews):
        static_observations = _host_sdk_static_observations(source)
        try:
            compile(source, f"<agent-protocol-review:{node.id}>", "exec")
        except SyntaxError as error:
            final_status = "syntax_error"
            final_reason = f"{error.msg} at line {error.lineno}"
            attempts.append(
                {
                    "review_index": review_index,
                    "code_sha256": _sha256(source),
                    "status": "syntax_error",
                    "reason": final_reason,
                    "patch_applied": False,
                }
            )
            break

        try:
            response = cast(
                dict[str, Any],
                query(
                    system_message=_prompt(agent, source, repairs_applied),
                    user_message=None,
                    func_spec=AGENT_PROTOCOL_REVIEW_SPEC,
                    model=(
                        str(getattr(settings, "model", "") or "")
                        or str(agent.acfg.feedback.model)
                    ),
                    temperature=float(
                        getattr(settings, "agent_semantic_temperature", 0.0)
                    ),
                    max_tokens=int(
                        getattr(settings, "agent_semantic_max_tokens", 4096)
                    ),
                    cfg=agent.cfg,
                ),
            )
        except Exception as error:  # preserved as an observation, never a veto
            final_status = "unavailable"
            final_reason = f"{type(error).__name__}: {error}"
            attempts.append(
                {
                    "review_index": review_index,
                    "code_sha256": _sha256(source),
                    "status": final_status,
                    "reason": final_reason,
                    "patch_applied": False,
                }
            )
            break

        status = str(response.get("status") or "uncertain").lower()
        reason = str(response.get("reason") or "")
        patch = str(response.get("revised_code") or "")
        record: dict[str, Any] = {
            "review_index": review_index,
            "code_sha256": _sha256(source),
            "status": status,
            "reason": reason,
            "actual_entrypoint": str(response.get("actual_entrypoint") or ""),
            "findings": list(map(str, response.get("findings") or [])),
            "patch_supplied": bool(patch.strip()),
            "patch_applied": False,
            "host_sdk_static_observations": static_observations,
        }
        attempts.append(record)
        final_status = status
        final_reason = reason

        if status == "clean":
            if patch.strip():
                record["consistency_error"] = "clean review supplied a patch"
                final_status = "uncertain"
                final_reason = record["consistency_error"]
            elif static_observations:
                record["consistency_error"] = (
                    "clean review contradicted Host SDK static observations"
                )
                final_status = "uncertain"
                final_reason = record["consistency_error"]
                if review_index + 1 < max_reviews:
                    record["review_retried"] = True
                    continue
            break
        if status == "uncertain" and review_index + 1 < max_reviews:
            record["review_retried"] = True
            continue
        if status != "revise" or not patch.strip():
            break
        if repairs_applied >= max_repairs:
            final_status = "repair_exhausted"
            final_reason = "Agent requested another repair after the bounded repair budget"
            break
        if "<<<<<<< SEARCH" not in patch and "< SEARCH" not in patch:
            record["patch_error"] = "repair was not SEARCH/REPLACE format"
            final_status = "repair_invalid"
            final_reason = record["patch_error"]
            break

        try:
            patched, count = SearchReplacePatcher().apply_patch(
                patch, source, strict=False
            )
            if count <= 0 or not patched or patched == source:
                raise ValueError("no SEARCH block matched the exact candidate source")
            compile(patched, f"<agent-protocol-repair:{node.id}>", "exec")
            preservation_audit = leakage_audit.audit_repair_preservation(
                patched, preservation
            )
            record["preservation_status"] = preservation_audit.get("status")
            record["preservation_issue_codes"] = [
                str(item.get("issue_code") or "")
                for item in preservation_audit.get("issues", [])
            ]
            if preservation_audit.get("status") != "clean":
                raise ValueError("repair changed the frozen method design")
        except Exception as error:
            record["patch_error"] = f"{type(error).__name__}: {error}"
            final_status = "repair_invalid"
            final_reason = record["patch_error"]
            break

        record["patch_applied"] = True
        record["patched_code_sha256"] = _sha256(patched)
        source = patched.strip()
        repairs_applied += 1

    report = {
        "schema": AGENT_PROTOCOL_REVIEW_SCHEMA,
        "enabled": True,
        "reviewer": "llm_agent",
        "host_receipt_admission_authority": False,
        "initial_code_sha256": initial_hash,
        "final_code_sha256": _sha256(source),
        "code_changed": initial_hash != _sha256(source),
        "repair_budget": max_repairs,
        "review_budget": max_reviews,
        "repairs_applied": repairs_applied,
        "final_status": final_status,
        "final_reason": final_reason,
        "execution_disposition": "observe_then_execute",
        "attempts": attempts,
    }
    _atomic_json(
        Path(agent.cfg.log_dir) / "agent_semantic_review" / f"{node.id}.json",
        report,
    )
    return source, report


__all__ = [
    "AGENT_PROTOCOL_REVIEW_SCHEMA",
    "AGENT_PROTOCOL_REVIEW_SPEC",
    "_host_sdk_static_observations",
    "run",
]
