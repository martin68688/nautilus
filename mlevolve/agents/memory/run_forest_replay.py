"""Fail-closed loading of exact source code from clean RunForest journals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agents.memory.external_skill_memory import resolve_memory_path
from agents.leakage_audit import audit_code, load_registry_audit, merge_audits
from fixed_holdout.mode import bypass_protocol_gates


REPLAY_SCHEMA_V1 = "run-forest-replay-targets-v1"
REPLAY_SCHEMA_V2 = "run-forest-replay-research-targets-v2"
# Backward-compatible public name used by older tests and release tooling.
REPLAY_SCHEMA = REPLAY_SCHEMA_V1


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _task_replay_portfolio(
    manifest: dict[str, Any], task_id: str
) -> dict[str, Any]:
    """Validate and return one Replay Research portfolio for ``task_id``.

    Version 2 deliberately separates score-oriented history from the
    architecture-diverse execution frontier.  A target can appear in both
    ordered lists, but every identity and source hash remains singular.
    """

    portfolios = [
        dict(item)
        for item in manifest.get("portfolios", [])
        if str(item.get("task_id") or "") == task_id
    ]
    if len(portfolios) != 1:
        raise ValueError(
            f"Expected exactly one replay research portfolio for {task_id}; "
            f"found {len(portfolios)}"
        )
    portfolio = portfolios[0]
    targets = [
        dict(item)
        for item in manifest.get("targets", [])
        if str(item.get("task_id") or "") == task_id
    ]
    if not targets:
        raise ValueError(f"Replay research portfolio for {task_id} has no targets")
    target_ids = [str(item.get("target_id") or "") for item in targets]
    if any(not value for value in target_ids) or len(target_ids) != len(set(target_ids)):
        raise ValueError("Replay research target IDs must be non-empty and unique per task")
    by_id = dict(zip(target_ids, targets))
    anchor_target_id = str(portfolio.get("anchor_target_id") or "")
    if anchor_target_id not in by_id:
        raise ValueError("Replay research anchor_target_id is missing from task targets")

    ordered_fields = (
        "score_frontier_target_ids",
        "diverse_frontier_target_ids",
    )
    for field in ordered_fields:
        values = [str(value) for value in (portfolio.get(field) or [])]
        if not values or len(values) > 5:
            raise ValueError(f"{field} must contain 1..5 target IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{field} contains duplicate target IDs")
        unknown = sorted(set(values) - set(by_id))
        if unknown:
            raise ValueError(f"{field} cites unknown targets: {unknown}")
        portfolio[field] = values
    if anchor_target_id not in portfolio["score_frontier_target_ids"]:
        raise ValueError("Replay research anchor must be present in score frontier")
    if anchor_target_id not in portfolio["diverse_frontier_target_ids"]:
        raise ValueError("Replay research anchor must be present in diverse frontier")

    code_hashes: set[str] = set()
    architecture_signatures: set[str] = set()
    for target_id, target in by_id.items():
        code_sha256 = str(target.get("code_sha256") or "")
        if not _is_sha256(code_sha256):
            raise ValueError(f"Replay research target {target_id} has invalid code_sha256")
        if code_sha256 in code_hashes:
            raise ValueError("Replay research targets must be deduplicated by code_sha256")
        code_hashes.add(code_sha256)
        protocol = str(target.get("validation_protocol") or "")
        authority = str(target.get("metric_authority") or "")
        architecture = str(target.get("architecture_signature") or "")
        if not protocol or not authority or not architecture:
            raise ValueError(
                f"Replay research target {target_id} must bind validation protocol, "
                "metric authority, and architecture signature"
            )
        if target_id in portfolio["diverse_frontier_target_ids"]:
            if architecture in architecture_signatures:
                raise ValueError(
                    "Replay research diverse frontier must be deduplicated by "
                    "architecture_signature"
                )
            architecture_signatures.add(architecture)
        allowed_actions = [str(value) for value in (target.get("allowed_actions") or [])]
        if not allowed_actions:
            raise ValueError(f"Replay research target {target_id} has no allowed_actions")
        target["allowed_actions"] = allowed_actions
        by_id[target_id] = target

    portfolio["task_id"] = task_id
    portfolio["anchor_target_id"] = anchor_target_id
    portfolio["targets_by_id"] = by_id
    return portfolio


def is_historical_replay_anchor(node: Any) -> bool:
    """Identify immutable exact replays that must not receive adoption rewrites."""

    source = dict(getattr(node, "replay_source", None) or {})
    return bool(
        source.get("historical_anchor_only") is True
        and source.get("exact_replay_execution", True) is True
        and str(getattr(node, "replay_status", "") or "")
        in {
            "historical_exact_anchor_loaded",
            "historical_exact_research_loaded",
        }
    )


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


def _journal_relative_to_runs(journal_value: str) -> Path:
    """Return the portion of a graph journal path below ``mlevolve/runs``."""

    raw = Path(str(journal_value or ""))
    if raw.is_absolute() or not raw.parts or ".." in raw.parts:
        raise ValueError(f"Replay journal path is unsafe: {journal_value}")
    parts = raw.parts
    if parts[:2] == ("mlevolve", "runs"):
        parts = parts[2:]
    elif parts[:1] == ("runs",):
        parts = parts[1:]
    else:
        raise ValueError(
            "Replay journal must be recorded below mlevolve/runs: "
            f"{journal_value}"
        )
    if not parts:
        raise ValueError(f"Replay journal path has no file component: {journal_value}")
    return Path(*parts)


def _resolve_replay_journal(
    policy: Any,
    *,
    repo_root: Path,
    journal_value: str,
) -> tuple[Path, Path]:
    """Resolve a journal from an explicit historical-run root or this checkout.

    End-to-end releases contain code and manifests but intentionally do not
    duplicate historical ``runs`` directories.  ``replay_runs_root`` binds
    exact replay to the persistent, read-only artifact store instead of the
    release checkout.  The old in-checkout location remains the default for
    legacy configs and unit tests.
    """

    relative = _journal_relative_to_runs(journal_value)
    configured_root = str(getattr(policy, "replay_runs_root", "") or "").strip()
    if configured_root:
        roots = [resolve_memory_path(configured_root, base_dir=repo_root)]
    else:
        roots = [(repo_root / "mlevolve" / "runs").resolve()]

    attempted: list[str] = []
    for root in roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError(f"Replay journal escapes configured runs root: {candidate}")
        attempted.append(str(candidate))
        if candidate.is_file():
            return candidate, resolved_root
    raise FileNotFoundError(
        "Replay journal not found in configured artifact root: "
        + ", ".join(attempted)
    )


def requires_protocol_repair(agent: Any, target_audit_status: str) -> bool:
    """Return whether a historical candidate must enter the internal repair flow."""
    return (
        target_audit_status == "candidate_replay"
        and not bypass_protocol_gates(agent.cfg)
    )


def full_runtime_migration_report(agent: Any, code: str) -> dict[str, Any]:
    """Describe why an otherwise legal replay needs a derived Host-SDK child.

    Historical RunForest programs predate the Contract-bound ``main()``
    lifecycle.  They remain valid method evidence, but must not be submitted as
    byte-exact executable candidates when Host SDK preflight is enabled.
    """

    preflight = getattr(getattr(agent, "acfg", None), "protocol_preflight", None)
    if not bool(getattr(preflight, "enabled", False)):
        return {}
    contract_path = str(getattr(preflight, "contract_path", "") or "")
    if not contract_path:
        return {}

    from authority.protocol_execution_contract import read_contract_artifact
    from protocol_runtime.preflight import static_compatibility_check

    contract = read_contract_artifact(contract_path)
    if not bool(contract.adapter_spec.get("full_runtime_sdk_required", False)):
        return {}
    static = static_compatibility_check(code, contract)
    missing = list(static.get("missing_full_runtime_coverage") or [])
    if not missing:
        return {}
    return {
        "schema": "mlevolve_replay_full_runtime_migration_v1",
        "status": "required",
        "contract_hash": contract.contract_hash,
        "code_sha256": str(static["code_sha256"]),
        "static_report_hash": str(static["static_report_hash"]),
        "missing_full_runtime_coverage": sorted(map(str, missing)),
        "source_execution_allowed": False,
        "derived_candidate_required": True,
    }


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


def load_exact_replay(
    agent: Any,
    *,
    target_id: str | None = None,
) -> dict[str, Any]:
    """Load one verified replay or immutable, non-executable repair seed.

    Version-1 manifests retain their exact single-target behavior.  Version 2
    selects the declared anchor by default and permits callers to request one
    exact target identity from the validated research portfolio.
    """
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
    task_id = str(getattr(agent.cfg, "exp_id", "") or "")
    manifest = json.loads(targets_path.read_text(encoding="utf-8"))
    manifest_schema = str(manifest.get("schema") or "")
    portfolio: dict[str, Any] = {}
    if manifest_schema == REPLAY_SCHEMA_V1:
        if target_id:
            raise ValueError("Version-1 replay manifests do not support target_id selection")
        matches = [
            dict(item)
            for item in manifest.get("targets", [])
            if str(item.get("task_id")) == task_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one replay target for {task_id}; found {len(matches)}"
            )
        target = matches[0]
    elif manifest_schema == REPLAY_SCHEMA_V2:
        portfolio = _task_replay_portfolio(manifest, task_id)
        selected_target_id = str(target_id or portfolio["anchor_target_id"])
        target = dict((portfolio["targets_by_id"] or {}).get(selected_target_id) or {})
        if not target:
            raise ValueError(
                f"Replay research target {selected_target_id!r} is not declared for {task_id}"
            )
    else:
        raise ValueError(f"Unsupported replay target schema: {manifest_schema}")
    source_kind = str(target.get("source_kind") or "runforest_journal")
    if source_kind not in {"runforest_journal", "recipe_implementation_capsule"}:
        raise ValueError(f"Unsupported replay source_kind: {source_kind}")
    run_id = str(target.get("run_id") or "")
    original_node_id = str(target.get("original_node_id") or "")
    run_short = _short_run_id(run_id)
    if any(run_short.startswith(str(prefix)) for prefix in meta.get("blocked_run_prefixes") or []):
        raise ValueError(f"Replay run {run_short} matches a blocked run prefix")

    if source_kind == "runforest_journal":
        source_runs = {str(value) for value in (meta.get("source_runs") or [])}
        source_run_shorts = {_short_run_id(value) for value in source_runs}
        if run_id not in source_runs and run_short not in source_run_shorts:
            raise ValueError(
                f"Replay run {run_short} is not present in the clean graph source set"
            )
        graph_node_id = f"run::{run_id}::node::{original_node_id}"
    else:
        graph_node_id = str(target.get("graph_node_id") or "")
        if not graph_node_id:
            raise ValueError(
                "recipe_implementation_capsule replay requires graph_node_id"
            )
    graph_node = layer.nodes.get(graph_node_id)
    if not graph_node or graph_node.get("type") != "RunNode":
        raise ValueError(f"Replay RunNode is missing from graph: {graph_node_id}")
    if str(graph_node.get("task")) != task_id or graph_node.get("is_buggy") is True or graph_node.get("is_valid") is False:
        raise ValueError("Replay RunNode does not satisfy task/validity requirements")

    if source_kind == "runforest_journal":
        run_record = layer.nodes.get(f"run::{run_id}")
        if not run_record:
            raise ValueError(f"Replay Run record is missing from graph: {run_id}")
        repo_root = Path(__file__).resolve().parents[3]
        journal_record_path = str(run_record.get("journal_path") or "")
        journal_path, journal_runs_root = _resolve_replay_journal(
            policy,
            repo_root=repo_root,
            journal_value=journal_record_path,
        )

        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        raw_nodes = [
            node
            for node in journal.get("nodes", [])
            if str(node.get("id")) == original_node_id
        ]
        if len(raw_nodes) != 1:
            raise ValueError(
                f"Expected one source node {original_node_id}; found {len(raw_nodes)}"
            )
        raw_node = raw_nodes[0]
        journal_reference = str(
            Path("mlevolve/runs") / journal_path.relative_to(journal_runs_root)
        )
    else:
        capsule = graph_node.get("implementation_capsule")
        if not isinstance(capsule, dict):
            raise ValueError(
                "Recipe replay source has no frozen implementation capsule: "
                f"{graph_node_id}"
            )
        if str(capsule.get("node_id") or "") != graph_node_id:
            raise ValueError("Recipe replay implementation capsule node mismatch")
        capsule_raw_node_id = str(capsule.get("source_raw_node_id") or "")
        if capsule_raw_node_id and capsule_raw_node_id != original_node_id:
            raise ValueError("Recipe replay implementation capsule raw-node mismatch")
        source_journal = str(capsule.get("source_journal") or "")
        if not source_journal:
            raise ValueError("Recipe replay implementation capsule has no source journal")
        portable_marker = "experiments/"
        marker_index = source_journal.find(portable_marker)
        journal_reference = (
            source_journal[marker_index:]
            if marker_index >= 0
            else source_journal
        )
        metric_direction = str(graph_node.get("metric_direction") or "")
        raw_node = {
            "id": original_node_id,
            "code": capsule.get("code"),
            "plan": graph_node.get("plan"),
            "metric": {
                "value": graph_node.get("metric"),
                "maximize": bool(
                    target.get("maximize")
                    if "maximize" in target
                    else metric_direction == "maximize"
                ),
            },
            "is_buggy": graph_node.get("is_buggy"),
            "is_valid": graph_node.get("is_valid"),
        }
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
        source_execution_verified=True,
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
    full_runtime_migration = (
        full_runtime_migration_report(agent, code)
        if not requires_repair and not external_holdout_mode
        else {}
    )
    replay_source = {
        "selection_mode": "audited_target_manifest_v2",
        "manifest_schema": manifest_schema,
        "source_kind": source_kind,
        "task_id": task_id,
        "run_id": run_id,
        "original_node_id": original_node_id,
        "graph_node_id": graph_node_id,
        "journal_path": journal_reference,
        "journal_artifact_root": (
            str(journal_runs_root) if source_kind == "runforest_journal" else ""
        ),
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
        "requires_full_runtime_migration": bool(full_runtime_migration),
        "full_runtime_migration": full_runtime_migration,
        "execution_seed_only": bool(full_runtime_migration),
    }
    if manifest_schema == REPLAY_SCHEMA_V2:
        replay_target_id = str(target.get("target_id") or "")
        is_anchor = replay_target_id == str(portfolio.get("anchor_target_id") or "")
        score_ids = list(portfolio.get("score_frontier_target_ids") or [])
        diverse_ids = list(portfolio.get("diverse_frontier_target_ids") or [])
        replay_source.update(
            {
                "target_id": replay_target_id,
                "target_role": "anchor" if is_anchor else "research",
                "portfolio_roles": [
                    role
                    for role, members in (
                        ("score_top5", score_ids),
                        ("diverse_frontier_top5", diverse_ids),
                    )
                    if replay_target_id in members
                ],
                "score_rank": (
                    score_ids.index(replay_target_id) + 1
                    if replay_target_id in score_ids
                    else None
                ),
                "diverse_rank": (
                    diverse_ids.index(replay_target_id) + 1
                    if replay_target_id in diverse_ids
                    else None
                ),
                "validation_protocol": str(target.get("validation_protocol") or ""),
                "metric_authority": str(target.get("metric_authority") or ""),
                "architecture_signature": str(
                    target.get("architecture_signature") or ""
                ),
                "method_family": str(target.get("method_family") or ""),
                "allowed_actions": list(target.get("allowed_actions") or []),
                "component_blueprint": dict(target.get("component_blueprint") or {}),
                "exact_replay_eligible": bool(
                    target.get("exact_replay_eligible", False)
                ),
                "historical_anchor_only": True,
                "exact_replay_execution": False,
            }
        )
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
    elif full_runtime_migration:
        replay_status = "blocked_legacy_full_runtime_seed"
        adoption_mode = "derived_full_runtime_migration_seed"
        requirement = (
            "Treat the byte-exact historical program as a non-executable method seed. "
            "Create a derived child that preserves its model, feature, ensemble, checkpoint, "
            "loss, optimizer, and training design while moving real fit, validation prediction, "
            "Host evaluation, and selection freeze into the frozen full-runtime SDK lifecycle. "
            "The child must receive a fresh code hash, Preflight, execution receipts, and score."
        )
    else:
        replay_status = "exact_source_loaded"
        adoption_mode = "exact_code_replay"
        requirement = "Execute the audited source code byte-for-byte without LLM redesign or code review."
    if manifest_schema == REPLAY_SCHEMA_V2 and not requires_repair and not full_runtime_migration:
        is_anchor = replay_source.get("target_role") == "anchor"
        replay_status = (
            "historical_exact_anchor_loaded"
            if is_anchor
            else "historical_exact_research_loaded"
        )
        adoption_mode = (
            "replay_exact_anchor" if is_anchor else "replay_exact_diverse"
        )
        replay_source["exact_replay_execution"] = True
        requirement = (
            "Execute the hash-bound historical anchor byte-for-byte before research."
            if is_anchor
            else "Execute this independently verified diverse historical source byte-for-byte; "
            "do not blend or transplant components in the exact replay step."
        )
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


def _read_replay_manifest(agent: Any) -> tuple[dict[str, Any], str]:
    policy = getattr(agent.acfg, "draft_role_policy", None)
    targets_value = str(getattr(policy, "replay_targets_path", "") or "")
    if not targets_value:
        raise ValueError("memory_reproduction requires draft_role_policy.replay_targets_path")
    layer = getattr(agent, "external_skill_memory", None)
    if layer is None or not hasattr(layer, "graph_path"):
        raise ValueError("memory_reproduction requires an initialized RunForest memory layer")
    path = resolve_memory_path(targets_value, base_dir=Path(layer.graph_path).parent)
    if not path.is_file():
        raise FileNotFoundError(f"Replay target manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8")), str(
        getattr(agent.cfg, "exp_id", "") or ""
    )


def _portfolio_receipt(
    portfolio: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the durable metadata-only receipt stored on the anchor node."""

    target_rows = []
    ordered_ids = list(
        dict.fromkeys(
            [
                *list(portfolio.get("score_frontier_target_ids") or []),
                *list(portfolio.get("diverse_frontier_target_ids") or []),
            ]
        )
    )
    for target_id in ordered_ids:
        replay = results[target_id]
        source = dict(replay.get("replay_source") or {})
        target_rows.append(
            {
                key: source.get(key)
                for key in (
                    "target_id",
                    "target_role",
                    "portfolio_roles",
                    "score_rank",
                    "diverse_rank",
                    "graph_node_id",
                    "code_sha256",
                    "historical_metric",
                    "maximize",
                    "validation_protocol",
                    "metric_authority",
                    "architecture_signature",
                    "method_family",
                    "allowed_actions",
                    "component_blueprint",
                    "target_audit_status",
                    "known_issue_codes",
                    "exact_replay_eligible",
                    "requires_repair",
                )
            }
        )
    return {
        "schema": "mlevolve_replay_research_portfolio_receipt_v1",
        "task_id": str(portfolio.get("task_id") or ""),
        "anchor_target_id": str(portfolio.get("anchor_target_id") or ""),
        "score_frontier_target_ids": list(
            portfolio.get("score_frontier_target_ids") or []
        ),
        "diverse_frontier_target_ids": list(
            portfolio.get("diverse_frontier_target_ids") or []
        ),
        "metric_comparison_policy": str(
            portfolio.get("metric_comparison_policy") or ""
        ),
        "fusion_weight_policy": str(portfolio.get("fusion_weight_policy") or ""),
        "targets": target_rows,
    }


def load_replay_research_portfolio(agent: Any) -> dict[str, Any]:
    """Load and verify the v2 anchor plus its bounded Top-K research set.

    Every target goes through the same task, graph, source-code hash, audit,
    and Authority checks as the original exact replay.  Targets carrying audit
    warnings remain visible as repair/reference evidence but are not put on the
    deterministic exact-execution queue.
    """

    manifest, task_id = _read_replay_manifest(agent)
    schema = str(manifest.get("schema") or "")
    if schema == REPLAY_SCHEMA_V1:
        anchor = load_exact_replay(agent)
        return {
            "schema": REPLAY_SCHEMA_V1,
            "anchor": anchor,
            "results": {},
            "receipt": {},
            "exact_research_target_ids": [],
            "strategy_target_ids": [],
        }
    if schema != REPLAY_SCHEMA_V2:
        raise ValueError(f"Unsupported replay target schema: {schema}")

    portfolio = _task_replay_portfolio(manifest, task_id)
    ordered_ids = list(
        dict.fromkeys(
            [
                *list(portfolio.get("score_frontier_target_ids") or []),
                *list(portfolio.get("diverse_frontier_target_ids") or []),
            ]
        )
    )
    results = {
        target_id: load_exact_replay(agent, target_id=target_id)
        for target_id in ordered_ids
    }
    anchor_id = str(portfolio["anchor_target_id"])
    exact_ids = [
        target_id
        for target_id in portfolio["diverse_frontier_target_ids"]
        if target_id != anchor_id
        and bool(
            (results[target_id].get("replay_source") or {}).get(
                "exact_replay_eligible"
            )
        )
        and results[target_id].get("replay_status")
        == "historical_exact_research_loaded"
    ]
    return {
        "schema": REPLAY_SCHEMA_V2,
        "anchor": results[anchor_id],
        "results": results,
        "receipt": _portfolio_receipt(portfolio, results),
        "exact_research_target_ids": exact_ids,
        "strategy_target_ids": list(portfolio["diverse_frontier_target_ids"]),
    }


def replay_research_strategy_cards(agent: Any) -> list[dict[str, Any]]:
    """Return bounded, code-bearing Strategy cards from the verified portfolio."""

    results = dict(getattr(agent, "_replay_research_results", {}) or {})
    ordered_ids = list(getattr(agent, "_replay_research_strategy_target_ids", []) or [])
    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    limit = max(0, int(getattr(policy, "replay_research_strategy_slots", 0) or 0))
    max_chars = max(
        1000,
        int(getattr(policy, "replay_research_card_max_chars", 12000) or 12000),
    )
    cards: list[dict[str, Any]] = []
    for target_id in ordered_ids:
        replay = dict(results.get(target_id) or {})
        source = dict(replay.get("replay_source") or {})
        if not replay or source.get("target_role") == "anchor":
            continue
        code = str(replay.get("code") or "")
        code_view = code
        truncated = False
        if len(code_view) > max_chars:
            half = max_chars // 2
            code_view = (
                code_view[:half]
                + "\n...[hash-bound source truncated for Strategy attention budget]...\n"
                + code_view[-half:]
            )
            truncated = True
        cards.append(
            {
                "memory_id": f"replay-target::{target_id}",
                "candidate_id": f"replay-target::{target_id}",
                "source": "verified_replay_research_portfolio",
                "router_visibility": "replay_research_portfolio",
                "title": f"Replay Research target {target_id}",
                "method_family": source.get("method_family"),
                "architecture_signature": source.get("architecture_signature"),
                "metric": source.get("historical_metric"),
                "historical_metric": source.get("historical_metric"),
                "validation_protocol": source.get("validation_protocol"),
                "metric_protocol": source.get("validation_protocol"),
                "metric_authority": source.get("metric_authority"),
                "metric_claim_status": "within_bound_protocol_only",
                "code_sha256": source.get("code_sha256"),
                "graph_node_id": source.get("graph_node_id"),
                "source_memory_ids": list(replay.get("source_ref_ids") or []),
                "allowed_actions": list(source.get("allowed_actions") or []),
                "component_blueprint": dict(source.get("component_blueprint") or {}),
                "target_audit_status": source.get("target_audit_status"),
                "known_issue_codes": list(source.get("known_issue_codes") or []),
                "exact_replay_eligible": bool(source.get("exact_replay_eligible")),
                "requires_repair": bool(source.get("requires_repair")),
                "code_visibility": (
                    "hash_bound_complete_source_loaded_bounded_prompt_view"
                    if truncated
                    else "hash_bound_complete_source_in_prompt"
                ),
                "source_code": code_view,
                "source_code_truncated": truncated,
                "text": json.dumps(
                    source.get("component_blueprint") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        if len(cards) >= limit:
            break
    return cards
