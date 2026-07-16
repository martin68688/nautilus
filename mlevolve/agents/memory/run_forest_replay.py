"""Fail-closed loading of exact source code from clean RunForest journals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agents.memory.external_skill_memory import resolve_memory_path
from agents.leakage_audit import audit_code, load_registry_audit, merge_audits
from fixed_holdout.mode import bypass_protocol_gates


REPLAY_SCHEMA = "run-forest-replay-targets-v1"


def _short_run_id(run_id: str) -> str:
    parts = str(run_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(run_id)


def _metric_value(raw_node: dict[str, Any]) -> tuple[float | None, bool | None]:
    metric = raw_node.get("metric")
    if isinstance(metric, dict):
        value = metric.get("value")
        return (float(value), bool(metric.get("maximize"))) if value is not None else (None, None)
    if isinstance(metric, (int, float)):
        return float(metric), None
    return None, None


def requires_protocol_repair(agent: Any, target_audit_status: str) -> bool:
    """Return whether a historical candidate must enter the internal repair flow."""
    return (
        target_audit_status == "candidate_replay"
        and not bypass_protocol_gates(agent.cfg)
    )


def validate_candidate_audit(
    target: dict[str, Any],
    replay_audit: dict[str, Any],
    *,
    external_holdout_mode: bool,
) -> None:
    """Validate old repair metadata only when it remains execution-authoritative."""
    if external_holdout_mode:
        return
    expected_issues = {
        str(item) for item in target.get("known_issue_codes", []) if item
    }
    detected_issues = {
        str(item.get("issue_code")) for item in replay_audit.get("issues", [])
    }
    if not expected_issues:
        raise ValueError("candidate_replay requires explicit known_issue_codes")
    if not expected_issues.issubset(detected_issues):
        missing = sorted(expected_issues - detected_issues)
        raise ValueError(f"Replay repair seed audit does not reproduce known issues: {missing}")
    if replay_audit.get("status") == "clean" or replay_audit.get("repair_required") is not True:
        raise ValueError("candidate_replay unexpectedly passed the fresh leakage audit")


def load_exact_replay(agent: Any) -> dict[str, Any]:
    """Load a verified replay or an immutable, non-executable repair seed."""
    policy = getattr(agent.acfg, "draft_role_policy", None)
    targets_value = str(getattr(policy, "replay_targets_path", "") or "")
    if not targets_value:
        raise ValueError("memory_reproduction requires draft_role_policy.replay_targets_path")

    layer = getattr(agent, "external_skill_memory", None)
    if layer is None or not hasattr(layer, "graph") or not hasattr(layer, "graph_path"):
        raise ValueError("memory_reproduction requires an initialized RunForest memory layer")
    meta = layer.graph.get("meta") or {}
    if meta.get("leak_verified") is not True or meta.get("paper_grade") is not True:
        raise ValueError("memory_reproduction requires a clean-certified RunForest graph")

    targets_path = resolve_memory_path(targets_value, base_dir=Path(layer.graph_path).parent)
    if not targets_path.exists():
        raise FileNotFoundError(f"Replay target manifest not found: {targets_path}")
    manifest = json.loads(targets_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"Unsupported replay target schema: {manifest.get('schema')}")

    task_id = str(getattr(agent.cfg, "exp_id", "") or "")
    matches = [item for item in manifest.get("targets", []) if str(item.get("task_id")) == task_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one replay target for {task_id}; found {len(matches)}")
    target = matches[0]
    run_id = str(target.get("run_id") or "")
    original_node_id = str(target.get("original_node_id") or "")
    run_short = _short_run_id(run_id)
    if run_short not in set(meta.get("source_runs") or []):
        raise ValueError(f"Replay run {run_short} is not present in the clean graph source set")
    if any(run_short.startswith(str(prefix)) for prefix in meta.get("blocked_run_prefixes") or []):
        raise ValueError(f"Replay run {run_short} matches a blocked run prefix")

    graph_node_id = f"run::{run_id}::node::{original_node_id}"
    graph_node = layer.nodes.get(graph_node_id)
    if not graph_node or graph_node.get("type") != "RunNode":
        raise ValueError(f"Replay RunNode is missing from graph: {graph_node_id}")
    if str(graph_node.get("task")) != task_id or graph_node.get("is_buggy") is True or graph_node.get("is_valid") is False:
        raise ValueError("Replay RunNode does not satisfy task/validity requirements")

    run_record = layer.nodes.get(f"run::{run_id}")
    if not run_record:
        raise ValueError(f"Replay Run record is missing from graph: {run_id}")
    repo_root = Path(__file__).resolve().parents[3]
    allowed_root = (repo_root / "mlevolve" / "runs").resolve()
    journal_path = (repo_root / str(run_record.get("journal_path") or "")).resolve()
    if not journal_path.is_relative_to(allowed_root):
        raise ValueError(f"Replay journal escapes clean runs directory: {journal_path}")
    if not journal_path.exists():
        raise FileNotFoundError(f"Replay journal not found: {journal_path}")

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    raw_nodes = [node for node in journal.get("nodes", []) if str(node.get("id")) == original_node_id]
    if len(raw_nodes) != 1:
        raise ValueError(f"Expected one source node {original_node_id}; found {len(raw_nodes)}")
    raw_node = raw_nodes[0]
    code = raw_node.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Replay source node has no executable code")

    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    expected_hash = str(target.get("code_sha256") or "")
    graph_hash = str(graph_node.get("code_sha256") or "")
    if not expected_hash or code_sha256 != expected_hash:
        raise ValueError("Replay source code does not match target manifest hash")
    if not graph_hash or code_sha256 != graph_hash:
        raise ValueError("Replay source code does not match RunForest graph hash")

    static_audit = audit_code(code)
    registry_audit = load_registry_audit(agent, code_sha256)
    replay_audit = merge_audits(code, static_audit, registry_audit)
    issue_codes = [str(item.get("issue_code")) for item in replay_audit.get("issues", [])]
    target_audit_status = str(target.get("audit_status") or "")
    if target_audit_status not in {"verified_clean", "candidate_replay"}:
        raise ValueError(f"Unsupported replay audit status: {target_audit_status}")

    historical_requires_repair = target_audit_status == "candidate_replay"
    external_holdout_mode = bypass_protocol_gates(agent.cfg)
    requires_repair = requires_protocol_repair(agent, target_audit_status)
    if historical_requires_repair:
        if (
            not external_holdout_mode
            and not bool(getattr(getattr(agent, "acfg", None), "check_data_leakage", False))
        ):
            raise ValueError("candidate_replay requires deterministic leakage auditing to be enabled")
        validate_candidate_audit(
            target,
            replay_audit,
            external_holdout_mode=external_holdout_mode,
        )
    else:
        if replay_audit.get("hard_block"):
            raise ValueError(
                "Verified replay source failed deterministic leakage audit: " + ", ".join(issue_codes)
            )
        if replay_audit.get("paper_grade_eligible") is not True:
            raise ValueError(
                "Verified replay source is not paper-grade eligible: " + ", ".join(issue_codes)
            )

    metric, maximize = _metric_value(raw_node)
    expected_metric = target.get("historical_metric")
    if expected_metric is not None and (metric is None or abs(metric - float(expected_metric)) > 1e-12):
        raise ValueError("Replay source metric does not match target manifest")
    if raw_node.get("is_buggy") is True or raw_node.get("is_valid") is False:
        raise ValueError("Replay source journal node is buggy or invalid")

    from authority.adapters.mlevolve.replay_gate import authorize_replay_source
    if not authorize_replay_source(
        agent,
        artifact_id=graph_node_id,
        code_sha256=code_sha256,
        audit=replay_audit,
        source_run_id=run_id,
        repair_seed=requires_repair,
    ):
        raise ValueError("Replay source lacks CODE_SEED authority under the active protocol")

    sop_ids = [str(ref) for ref in target.get("sop_ids", [])]
    for sop_id in sop_ids:
        if sop_id not in layer.nodes or layer.nodes[sop_id].get("type") != "SOP":
            raise ValueError(f"Replay SOP is missing from clean graph: {sop_id}")
    transition_ids = [
        node_id for node_id, node in layer.nodes.items()
        if node.get("type") == "Transition" and node.get("child_node_id") == graph_node_id
    ]
    source_ref_ids = list(dict.fromkeys([graph_node_id, *transition_ids, *sop_ids]))
    replay_source = {
        "selection_mode": "audited_target_manifest_v2",
        "task_id": task_id,
        "run_id": run_id,
        "original_node_id": original_node_id,
        "graph_node_id": graph_node_id,
        "journal_path": str(Path("mlevolve/runs") / journal_path.relative_to(allowed_root)),
        "historical_metric": metric,
        "maximize": maximize,
        "code_sha256": code_sha256,
        "sop_ids": sop_ids,
        "selection_basis": str(target.get("selection_basis") or ""),
        "leakage_audit": replay_audit,
        "target_audit_status": target_audit_status,
        "historical_requires_protocol_repair": historical_requires_repair,
        "evaluation_authority": (
            "fixed_hidden_holdout_terminal_only"
            if external_holdout_mode
            else "internal_protocol_audit"
        ),
        "requires_repair": requires_repair,
        "repair_seed_only": requires_repair,
        "known_issue_codes": sorted({str(item) for item in target.get("known_issue_codes", []) if item}),
    }
    if requires_repair:
        replay_status = "blocked_exact_source_repair_seed"
        adoption_mode = "blocked_exact_source_repair_seed"
        requirement = (
            "Preserve this historical source byte-for-byte as a non-executable repair seed. "
            "Create a child that fixes only the audited data/evaluation protocol while preserving "
            "the complete model, feature, ensemble, checkpoint, and training design."
        )
    elif historical_requires_repair and external_holdout_mode:
        replay_status = "exact_source_loaded_fixed_holdout"
        adoption_mode = "exact_code_replay_fixed_holdout"
        requirement = (
            "Execute the historical source byte-for-byte in the label-isolated train view. "
            "Its historical and self-reported validation metrics are search-only; only the "
            "terminal external fixed-holdout evaluator may rank this submission."
        )
    else:
        replay_status = "exact_source_loaded"
        adoption_mode = "exact_code_replay"
        requirement = "Execute the audited source code byte-for-byte without LLM redesign or code review."
    return {
        "code": code,
        "plan": str(raw_node.get("plan") or "Exact source from an audited RunForest target."),
        "source_ref_ids": source_ref_ids,
        "replay_source": replay_source,
        "leakage_audit": replay_audit,
        "requires_repair": requires_repair,
        "replay_status": replay_status,
        "adoption_mode": adoption_mode,
        "role_contract": {
            "role": "memory_reproduction",
            "requirement": requirement,
            "source": replay_source,
        },
    }
