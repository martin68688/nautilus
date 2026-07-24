from __future__ import annotations

import ast
import hashlib
from typing import Any, Mapping

from schema import sha256_json


CODE_ARTIFACT_SCHEMA = "decision_admissibility_tier1_code_artifact_v1"
CODE_EXECUTION_RECEIPT_SCHEMA = (
    "decision_admissibility_tier1_code_execution_receipt_v1"
)
STATIC_ACTUATION_RECEIPT_SCHEMA = (
    "decision_admissibility_tier1_static_actuation_receipt_v1"
)
RUNTIME_ACTUATION_RECEIPT_SCHEMA = (
    "decision_admissibility_tier1_runtime_actuation_receipt_v1"
)

CONTROLLED_ACTION_UTILITY = {
    "oracle": 1.0,
    "neutral": 0.5,
    "mismatch": 0.25,
    "forbidden": 0.0,
}


def _literal(value: Any) -> ast.expr:
    return ast.parse(repr(value), mode="eval").body


def _assign(name: str, value: Any) -> ast.Assign:
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=_literal(value),
    )


def canonical_action_program(
    *,
    episode_id: str,
    stage: str,
    protocol_id: str,
    selected_action_id: str,
    config_patch: Mapping[str, Any],
    protocol_legal: bool,
) -> dict[str, Any]:
    """Build code for the current selected node without embedding memory metadata."""

    patch = dict(config_patch)
    patch_hash = sha256_json(patch)
    events = [
        {
            "event": "controlled_action_executed",
            "episode_id": episode_id,
            "stage": stage,
            "action_id": selected_action_id,
        },
        *[
            {
                "event": "config_applied",
                "key": key,
                "value_hash": sha256_json(patch[key]),
            }
            for key in sorted(patch)
        ],
        {
            "event": "protocol_legality_observed",
            "protocol_id": protocol_id,
            "protocol_legal": bool(protocol_legal),
        },
    ]
    runtime_result = {
        "execution_succeeded": True,
        "episode_id": episode_id,
        "stage": stage,
        "selected_action_id": selected_action_id,
        "config_patch_hash": patch_hash,
        "protocol_id": protocol_id,
        "protocol_legal": bool(protocol_legal),
        "event_count": len(events),
    }
    module = ast.Module(
        body=[
            _assign(
                "NODE_METADATA",
                {
                    "episode_id": episode_id,
                    "stage": stage,
                    "protocol_id": protocol_id,
                    "selected_action_id": selected_action_id,
                },
            ),
            _assign("CONFIG_PATCH", patch),
            _assign("CONFIG_PATCH_HASH", patch_hash),
            _assign("RUNTIME_EVENTS", events),
            _assign("RUNTIME_RESULT", runtime_result),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    normalized_ast = ast.dump(module, include_attributes=False, annotate_fields=True)
    artifact: dict[str, Any] = {
        "schema": CODE_ARTIFACT_SCHEMA,
        "episode_id": episode_id,
        "stage": stage,
        "protocol_id": protocol_id,
        "selected_action_id": selected_action_id,
        "config_patch": patch,
        "config_patch_hash": patch_hash,
        "protocol_legal": bool(protocol_legal),
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "normalized_ast_sha256": hashlib.sha256(
            normalized_ast.encode("utf-8")
        ).hexdigest(),
        "memory_metadata_embedded": False,
        "artifact_hash": "",
    }
    artifact["artifact_hash"] = sha256_json(
        {key: value for key, value in artifact.items() if key != "artifact_hash"}
    )
    return artifact


def execute_canonical_action(artifact: Mapping[str, Any]) -> dict[str, Any]:
    source = str(artifact["source"])
    observed_ast = ast.parse(source, mode="exec")
    observed_ast_hash = hashlib.sha256(
        ast.dump(
            observed_ast,
            include_attributes=False,
            annotate_fields=True,
        ).encode("utf-8")
    ).hexdigest()
    namespace: dict[str, Any] = {"__builtins__": {}}
    error = ""
    try:
        compiled = compile(observed_ast, "<tier1-controlled-node>", "exec")
        exec(compiled, namespace, namespace)
    except Exception as exception:  # Host execution errors are evidence, not hidden.
        error = f"{type(exception).__name__}: {exception}"[:1000]
    events = namespace.get("RUNTIME_EVENTS")
    result = namespace.get("RUNTIME_RESULT")
    expected_patch_hash = artifact["config_patch_hash"]
    event_patch_keys = {
        str(row.get("key") or "")
        for row in events or []
        if isinstance(row, Mapping) and row.get("event") == "config_applied"
    }
    expected_patch_keys = set((artifact.get("config_patch") or {}).keys())
    passed = bool(
        not error
        and observed_ast_hash == artifact["normalized_ast_sha256"]
        and isinstance(events, list)
        and isinstance(result, Mapping)
        and result.get("execution_succeeded") is True
        and result.get("selected_action_id") == artifact["selected_action_id"]
        and result.get("config_patch_hash") == expected_patch_hash
        and event_patch_keys == expected_patch_keys
    )
    receipt: dict[str, Any] = {
        "schema": CODE_EXECUTION_RECEIPT_SCHEMA,
        "artifact_hash": artifact["artifact_hash"],
        "source_sha256": artifact["source_sha256"],
        "normalized_ast_sha256": observed_ast_hash,
        "compile_and_exec_attempted": True,
        "execution_passed": passed,
        "error": error,
        "runtime_events": events if isinstance(events, list) else [],
        "runtime_events_hash": sha256_json(events if isinstance(events, list) else []),
        "runtime_result": dict(result) if isinstance(result, Mapping) else {},
        "runtime_result_hash": sha256_json(
            dict(result) if isinstance(result, Mapping) else {}
        ),
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    return receipt


def static_actuation_receipt(
    *,
    request_id: str,
    condition: str,
    memory: Mapping[str, Any] | None,
    recommended_action: Mapping[str, Any] | None,
    selected_action: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    memory_exposed = memory is not None
    recommended_action_id = str(
        (memory or {}).get("recommended_action_id") or ""
    )
    selected_action_id = str(selected_action.get("action_id") or "")
    recommended_patch_hash = (
        sha256_json((recommended_action or {}).get("config_patch") or {})
        if recommended_action is not None
        else ""
    )
    action_alignment = bool(
        memory_exposed and recommended_action_id == selected_action_id
        and str((recommended_action or {}).get("action_id") or "")
        == recommended_action_id
    )
    patch_alignment = bool(
        action_alignment
        and recommended_patch_hash
        and recommended_patch_hash == artifact.get("config_patch_hash")
    )
    code_binding = bool(
        artifact.get("selected_action_id") == selected_action_id
        and artifact.get("config_patch") == selected_action.get("config_patch")
        and artifact.get("memory_metadata_embedded") is False
    )
    passed = action_alignment and patch_alignment and code_binding
    receipt: dict[str, Any] = {
        "schema": STATIC_ACTUATION_RECEIPT_SCHEMA,
        "request_id": request_id,
        "condition": condition,
        "memory_exposed": memory_exposed,
        "memory_id": str((memory or {}).get("memory_id") or ""),
        "recommended_action_id": recommended_action_id,
        "selected_action_id": selected_action_id,
        "action_alignment": action_alignment,
        "recommended_patch_hash": recommended_patch_hash,
        "selected_patch_hash": artifact.get("config_patch_hash", ""),
        "patch_alignment": patch_alignment,
        "code_binding": code_binding,
        "static_actuation_passed": passed,
        "agent_self_report_required": False,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    return receipt


def runtime_actuation_receipt(
    *,
    static_receipt: Mapping[str, Any],
    code_execution_receipt: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_result = code_execution_receipt.get("runtime_result") or {}
    selected_patch_executed = bool(
        code_execution_receipt.get("execution_passed") is True
        and runtime_result.get("selected_action_id") == artifact.get("selected_action_id")
        and runtime_result.get("config_patch_hash") == artifact.get("config_patch_hash")
    )
    passed = bool(
        static_receipt.get("static_actuation_passed") is True
        and selected_patch_executed
    )
    receipt: dict[str, Any] = {
        "schema": RUNTIME_ACTUATION_RECEIPT_SCHEMA,
        "request_id": static_receipt["request_id"],
        "condition": static_receipt["condition"],
        "static_receipt_hash": static_receipt["receipt_hash"],
        "code_execution_receipt_hash": code_execution_receipt["receipt_hash"],
        "selected_patch_executed": selected_patch_executed,
        "runtime_actuation_passed": passed,
        "code_execution_is_not_historical_actuation": True,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    return receipt


def controlled_action_utility(action: Mapping[str, Any]) -> float:
    if action.get("protocol_legal") is not True:
        return 0.0
    return CONTROLLED_ACTION_UTILITY.get(str(action.get("role") or ""), 0.0)
