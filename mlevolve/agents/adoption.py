"""Adoption tracking helper: record which memory entries were injected into a node's prompt.

Side-channel ONLY:
- appends to node.adoption_log AFTER the prompt is sent and the node is registered
- never touches the prompt string
- no-op when adoption tracking is disabled (agent.adoption_tracking_enabled=False) or ref_ids empty

This lets the post-run analyzer (analysis/adoption_tracker.py) correlate "which memory
entries each node actually saw" with "what the generated code does", without polluting the
LLM prompt (memory ids never appear in any prompt text).
"""
import time
import logging
import copy
import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def _routing_candidate_id(row):
    if not isinstance(row, dict):
        return ""
    for key in ("candidate_id", "id", "sop_id", "node_id", "transition_id"):
        value = str(row.get(key) or "")
        if value:
            return value
    return ""


def _serialize_layered_routing_trace(pack, ref_ids):
    """Normalize a layered Dynamic pack into the shared routing trace schema.

    Layered retrieval predates the ten-system adapter and has richer,
    stage-specific pack shapes.  Persist the exact Prompt-visible refs passed
    to ``log_adoption`` and derive suppression only from candidates the router
    actually observed.  This is observational bookkeeping; it never changes
    retrieval or the already-sent Prompt.
    """

    visible_ids = list(dict.fromkeys(str(value) for value in ref_ids if value))
    visible_set = set(visible_ids)
    raw_rows = []
    seen_raw = set()
    for key in (
        "pre_gate_raw_candidates",
        "strategy_candidates",
        "direct_sop_candidates",
        "tree_candidate_details",
        "selected_tactics",
    ):
        for row in pack.get(key) or []:
            if not isinstance(row, dict):
                continue
            candidate_id = _routing_candidate_id(row)
            if not candidate_id or candidate_id in seen_raw:
                continue
            normalized = copy.deepcopy(row)
            normalized["candidate_id"] = candidate_id
            normalized["candidate_source"] = key
            normalized["final_prompt_visible"] = candidate_id in visible_set
            raw_rows.append(normalized)
            seen_raw.add(candidate_id)

    navigation = [
        copy.deepcopy(row)
        for row in pack.get("navigation_trace") or []
        if isinstance(row, dict)
    ]
    for row in navigation:
        candidate_id = _routing_candidate_id(row)
        if not candidate_id or candidate_id in seen_raw:
            continue
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_source": "navigation_trace",
                "selection_state": str(row.get("selection_state") or ""),
                "retrieval_channel": str(row.get("retrieval_channel") or ""),
                "final_prompt_visible": candidate_id in visible_set,
            }
        )
        seen_raw.add(candidate_id)

    details = {}
    for row in raw_rows:
        details[str(row["candidate_id"])] = row
    final_candidates = []
    for candidate_id in visible_ids:
        detail = copy.deepcopy(details.get(candidate_id) or {})
        detail["candidate_id"] = candidate_id
        detail["final_prompt_visible"] = True
        final_candidates.append(detail)

    suppressed = {}
    for row in raw_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in visible_set:
            suppressed[candidate_id] = {
                "candidate_id": candidate_id,
                "reason": str(
                    row.get("gate_reason")
                    or row.get("selection_state")
                    or "not_selected_for_prompt"
                ),
                "candidate_source": str(row.get("candidate_source") or ""),
            }
    for row in (pack.get("execution_safety_gate") or {}).get("rejected") or []:
        candidate_id = _routing_candidate_id(row)
        if candidate_id and candidate_id not in visible_set:
            suppressed[candidate_id] = {
                "candidate_id": candidate_id,
                "reason": str(row.get("reason") or "execution_safety_gate"),
                "candidate_source": "execution_safety_gate",
            }

    raw_hash = hashlib.sha256(
        json.dumps(
            raw_rows,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    task_profile = pack.get("task_profile") or {}
    return {
        "schema": "mlevolve_memory_routing_trace_v1",
        "memory_pack_schema": str(pack.get("schema") or ""),
        "algorithm_version": str(pack.get("algorithm_version") or ""),
        "system_id": "dynamic_hybrid",
        "stage_route": copy.deepcopy(pack.get("stage_route") or {}),
        "target_task_id": str(
            pack.get("target_task_id") or task_profile.get("task_id") or ""
        ),
        "candidate_pool_hash": raw_hash,
        "candidate_pool_source": "layered_strategy_observed_candidates",
        "raw_pool_observed": True,
        "raw_candidates": raw_rows,
        "selected_candidates": final_candidates,
        "suppressed_candidates": list(suppressed.values()),
        "final_prompt_candidate_ids": visible_ids,
        "final_prompt_candidates": final_candidates,
        "selected_sop_gateway_ids": [
            _routing_candidate_id(row)
            for row in pack.get("selected_sop_gateways") or []
            if _routing_candidate_id(row)
        ],
        "strategy_selection": copy.deepcopy(pack.get("strategy_selection") or {}),
        "gateway_selection": copy.deepcopy(pack.get("gateway_selection") or {}),
        "l3_agent_match": copy.deepcopy(pack.get("l3_agent_match") or {}),
        "navigation_trace": navigation,
        "visible_clause_ids": list(pack.get("visible_clause_ids") or []),
        "prompt_token_count": int(pack.get("prompt_token_count") or 0),
        "prompt_token_count_available": "prompt_token_count" in pack,
        "prompt_truncated": bool(pack.get("prompt_truncated")),
        "visibility_safety_gate": copy.deepcopy(
            pack.get("visibility_safety_gate") or {}
        ),
        "unauthorized_prompt_exposure": int(
            pack.get("unauthorized_prompt_exposure") or 0
        ),
        "memory_snapshot_bound_but_not_exposed": bool(
            pack.get("memory_snapshot_bound_but_not_exposed")
        ),
        "memory_bundle": copy.deepcopy(pack.get("memory_bundle") or {}),
        "observational_only": True,
    }


def _serialize_exact_replay_trace(node, ref_ids):
    """Record direct historical-code selection without calling it Prompt use."""

    source = copy.deepcopy(getattr(node, "replay_source", None) or {})
    graph_node_id = str(source.get("graph_node_id") or "")
    raw = [
        {
            "candidate_id": graph_node_id,
            "candidate_source": "frozen_exact_replay_target",
            "source_kind": str(source.get("source_kind") or ""),
            "historical_metric": source.get("historical_metric"),
            "code_sha256": str(source.get("code_sha256") or ""),
            "selected_for_direct_execution": True,
            "final_prompt_visible": False,
        }
    ] if graph_node_id else []
    pool_hash = hashlib.sha256(
        json.dumps(
            raw,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "mlevolve_memory_routing_trace_v1",
        "memory_pack_schema": "exact_replay_target_v1",
        "algorithm_version": "exact_code_replay_v1",
        "system_id": "dynamic_hybrid",
        "stage_route": {
            "stage": "draft",
            "route": "exact_code_replay",
            "control": "memory_reproduction",
        },
        "role_policy_abstention": {
            "status": "abstain",
            "reason": "draft_origin_policy_uses_exact_code_replay_not_router_prompt",
            "draft_only": True,
        },
        "target_task_id": str(source.get("task_id") or ""),
        "candidate_pool_hash": pool_hash,
        "candidate_pool_source": "frozen_exact_replay_target",
        "raw_pool_observed": True,
        "raw_candidates": raw,
        "selected_candidates": copy.deepcopy(raw),
        "suppressed_candidates": [],
        # The implementation is loaded as executable code, not shown to an
        # LLM through the Prompt.  Keep these fields empty by construction.
        "final_prompt_candidate_ids": [],
        "final_prompt_candidates": [],
        "direct_code_replay": True,
        "direct_replay_source_ref_ids": list(
            dict.fromkeys(str(value) for value in ref_ids if value)
        ),
        "replay_source": source,
        "navigation_trace": [],
        "visible_clause_ids": [],
        "prompt_token_count": 0,
        "prompt_token_count_available": True,
        "prompt_truncated": False,
        "observational_only": True,
    }


def _hybrid_trace_for_ref(pack, ref_id):
    matches = []
    for item in pack.get("navigation_trace", []):
        if not isinstance(item, dict):
            continue
        if (
            item.get("candidate_id") == ref_id
            or item.get("gateway_sop_id") == ref_id
            or ref_id in (item.get("supporting_transition_ids") or [])
            or ref_id in (item.get("expanded_candidate_ids") or [])
        ):
            matches.append(item)
    if matches:
        return matches[-1]
    if ref_id in (pack.get("evidence_refs") or []):
        return {
            "retrieval_channel": "evidence",
            "candidate_class": None,
            "gateway_sop_id": None,
            "supporting_transition_ids": [],
            "selection_reason": "verified evidence attached to an expanded transition",
            "selection_state": "injected",
        }
    if ref_id in (pack.get("failure_patterns") or []):
        return {
            "retrieval_channel": "risk_warning",
            "candidate_class": None,
            "gateway_sop_id": None,
            "supporting_transition_ids": [],
            "selection_reason": "failure pattern injected as warning-only evidence",
            "selection_state": "injected",
        }
    return {
        "retrieval_channel": "hybrid_pack",
        "candidate_class": None,
        "gateway_sop_id": None,
        "supporting_transition_ids": [],
        "selection_reason": "referenced by the injected hybrid pack",
        "selection_state": "injected",
    }


def log_adoption(
    node: "SearchNode",
    agent,
    source: str,
    ref_ids,
    stage: str,
    adoption_mode: str = "prompt_injection",
) -> None:
    """Append adoption records to node.adoption_log.

    Args:
        node: the SearchNode whose prompt had memory injected.
        agent: AgentSearch instance (reads adoption_tracking_enabled).
        source: "methodology" | "global_memory" | "skillgraph" | custom external-memory source.
        ref_ids: list of memory entry ids injected into this node's prompt.
        stage: "draft" | "improve" | "debug".
    """
    layer = getattr(agent, "external_skill_memory", None)
    ref_ids = [ref_id for ref_id in (ref_ids or []) if ref_id]
    pack = {}
    if source == "run_forest_stage_hybrid_memory" and layer is not None:
        getter = getattr(layer, "current_navigation_pack", None)
        if callable(getter):
            pack = getter()
            from agents.memory.end2end_memory_system import canonical_stage

            expected_stage = canonical_stage(stage)
            raw_pack_stage = str(
                (pack.get("stage_route") or {}).get("stage") or ""
            )
            pack_stage = canonical_stage(raw_pack_stage)
            if pack.get("schema") == "experiment_r_memory_pack_v1":
                if pack_stage != expected_stage:
                    raise RuntimeError(
                        "Current Router pack stage does not match generated node: "
                        f"pack={pack_stage!r} node={expected_stage!r}"
                    )
            node.memory_navigation_trace = copy.deepcopy(
                pack.get("navigation_trace", [])
            )
            if bool(getattr(layer, "experiment_r_enabled", False)):
                snapshot = getattr(layer, "memory_snapshot", None)
                base = getattr(snapshot, "base_bundle", None)
                run_identity = getattr(
                    getattr(agent, "cfg", None), "run_identity", None
                )
                pool = pack.get("candidate_pool") or {}
                # Experiment-R packs historically used a Mapping with an
                # embedded pool_identity.  Cross-task transfer deliberately
                # exposes a score-free list of portable candidates instead.
                # Preserve either shape without treating a list like a dict.
                pool_identity = (
                    pool.get("pool_identity") or {}
                    if isinstance(pool, dict)
                    else {}
                )
                node.memory_routing_trace = {
                    "schema": "mlevolve_memory_routing_trace_v1",
                    "memory_pack_schema": str(pack.get("schema") or ""),
                    "algorithm_version": str(pack.get("algorithm_version") or ""),
                    "system_id": "dynamic_hybrid",
                    "stage_route": copy.deepcopy(pack.get("stage_route") or {}),
                    "node_stage": expected_stage,
                    "node_stage_raw": str(stage),
                    "pack_stage": pack_stage,
                    "pack_stage_raw": raw_pack_stage,
                    "target_task_id": str(pack.get("target_task_id") or ""),
                    "memory_pool_sha256": str(
                        pack.get("memory_pool_sha256")
                        or pool_identity.get("memory_pool_sha256")
                        or ""
                    ),
                    "candidate_pool_hash": str(pack.get("candidate_pool_hash") or ""),
                    "candidate_pool_source": str(pack.get("candidate_pool_source") or ""),
                    "qualification_checkpoint_id": str(
                        pack.get("qualification_checkpoint_id") or ""
                    ),
                    "qualification_candidate_pool_artifact_sha256": str(
                        pack.get("qualification_candidate_pool_artifact_sha256") or ""
                    ),
                    "ranking_contract": str(pack.get("ranking_contract") or ""),
                    "live_query_used_for_candidate_pool": bool(
                        pack.get("live_query_used_for_candidate_pool", True)
                    ),
                    "candidate_pool_identity": copy.deepcopy(
                        pool_identity
                    ),
                    "raw_candidates": copy.deepcopy(pool),
                    "selected_candidates": copy.deepcopy(
                        pack.get("selected_items")
                        or pack.get("selected_candidates")
                        or []
                    ),
                    "budget_contract": copy.deepcopy(pack.get("budget_contract") or {}),
                    "safety_gate": copy.deepcopy(
                        pack.get("safety_gate")
                        or pack.get("visibility_safety_gate")
                        or {}
                    ),
                    "memory_transfer": copy.deepcopy(
                        pack.get("memory_transfer") or {}
                    ),
                    "final_prompt_candidate_ids": list(
                        pack.get("final_prompt_candidate_ids") or []
                    ),
                    "final_prompt_candidates": copy.deepcopy(
                        pack.get("final_prompt_candidates") or []
                    ),
                    # Evidence Resolver runs after the Retrieval Judge and may
                    # open hash-bound parent/child code that is intentionally
                    # absent from the first-layer prompt.  Keep both the
                    # receipt and opened evidence in the durable Journal trace;
                    # otherwise the live Strategy can consume the evidence but
                    # the run cannot prove which selected IDs were resolved.
                    "evidence_resolver": copy.deepcopy(
                        pack.get("evidence_resolver") or {}
                    ),
                    "resolved_evidence": copy.deepcopy(
                        pack.get("resolved_evidence") or []
                    ),
                    "retrieval_agent": copy.deepcopy(
                        pack.get("retrieval_agent") or {}
                    ),
                    "router_activation": copy.deepcopy(
                        pack.get("router_activation") or {}
                    ),
                    "visible_clause_ids": list(pack.get("visible_clause_ids") or []),
                    "prompt_token_count": int(pack.get("prompt_token_count") or 0),
                    "prompt_truncated": bool(pack.get("prompt_truncated")),
                    "memory_snapshot_bound_but_not_exposed": bool(
                        pack.get("memory_snapshot_bound_but_not_exposed")
                    ),
                    "base_bundle_id": str(getattr(base, "bundle_id", "") or ""),
                    "base_manifest_sha256": str(
                        getattr(base, "manifest_sha256", "") or ""
                    ),
                    "memory_snapshot_sha256": str(
                        getattr(snapshot, "snapshot_sha256", "") or ""
                    ),
                    "session_overlay_path": str(
                        getattr(snapshot, "session_overlay_path", "") or ""
                    ),
                    "production_binding_sha256": str(
                        getattr(run_identity, "memory_bundle_binding_sha256", "") or ""
                    ),
                    "current_file_sha256": str(
                        getattr(run_identity, "memory_current_sha256", "") or ""
                    ),
                }
                adapter = getattr(agent, "evaluation_authority", None)
                attester = getattr(adapter, "attest_experiment_r_candidate_pool", None)
                if callable(attester):
                    attester(node)
            elif pack.get("schema") == "mlevolve_end2end_memory_pack_v1":
                node.memory_routing_trace = {
                    "schema": "mlevolve_memory_routing_trace_v1",
                    "memory_pack_schema": str(pack.get("schema") or ""),
                    "algorithm_version": str(
                        pack.get("algorithm_version") or ""
                    ),
                    "system_id": str(pack.get("system_id") or ""),
                    "stage_route": copy.deepcopy(pack.get("stage_route") or {}),
                    "target_task_id": str(pack.get("target_task_id") or ""),
                    "candidate_pool_hash": str(
                        pack.get("candidate_pool_hash") or ""
                    ),
                    "candidate_pool_source": str(
                        pack.get("candidate_pool_source") or ""
                    ),
                    "raw_pool_observed": bool(pack.get("raw_pool_observed")),
                    "raw_candidates": copy.deepcopy(
                        pack.get("candidate_pool") or []
                    ),
                    "selected_candidates": copy.deepcopy(
                        pack.get("selected_candidates") or []
                    ),
                    "suppressed_candidates": copy.deepcopy(
                        pack.get("suppressed_candidates") or []
                    ),
                    "final_prompt_candidate_ids": list(
                        pack.get("final_prompt_candidate_ids") or []
                    ),
                    "final_prompt_candidates": copy.deepcopy(
                        pack.get("final_prompt_candidates") or []
                    ),
                    "visible_clause_ids": list(
                        pack.get("visible_clause_ids") or []
                    ),
                    "prompt_token_count": int(
                        pack.get("prompt_token_count") or 0
                    ),
                    "prompt_truncated": bool(pack.get("prompt_truncated")),
                    "visibility_safety_gate": copy.deepcopy(
                        pack.get("visibility_safety_gate") or {}
                    ),
                    "unauthorized_prompt_exposure": int(
                        pack.get("unauthorized_prompt_exposure") or 0
                    ),
                    "memory_snapshot_bound_but_not_exposed": bool(
                        pack.get("memory_snapshot_bound_but_not_exposed")
                    ),
                    "memory_bundle": copy.deepcopy(pack.get("memory_bundle") or {}),
                }
            elif pack.get("schema") in {
                "layered_strategy_memory_pack_v1",
                "stage_hybrid_memory_pack_v1",
                "layered_model_design_tactics_v1",
            }:
                node.memory_routing_trace = _serialize_layered_routing_trace(
                    pack, ref_ids
                )
        if (
            not getattr(node, "memory_routing_trace", None)
            and (getattr(node, "replay_source", None) or {}).get(
                "graph_node_id"
            )
        ):
            node.memory_routing_trace = _serialize_exact_replay_trace(
                node, ref_ids
            )
    if not ref_ids:
        return
    visibility_pack = (
        getattr(agent, "methodology_visibility_pack", None)
        if source == "methodology"
        else None
    )
    if visibility_pack is None and layer is not None:
        getter = getattr(layer, "current_visibility_pack", None)
        if callable(getter):
            visibility_pack = getter()
    adapter = getattr(agent, "evaluation_authority", None)
    exposure_recorder = getattr(adapter, "record_prompt_exposure", None)
    candidate_exposure_recorder = getattr(
        adapter, "record_memory_candidate_exposure", None
    )
    if (
        pack.get("schema") in {
            "mlevolve_end2end_memory_pack_v1",
            "experiment_r_memory_pack_v1",
            "layered_strategy_memory_pack_v1",
            "stage_hybrid_memory_pack_v1",
            "layered_model_design_tactics_v1",
            "mlevolve_cross_task_transfer_pack_v1",
            "mlevolve_cross_task_transfer_pack_v2",
        }
        and callable(candidate_exposure_recorder)
    ):
        try:
            candidate_exposure_recorder(
                node=node,
                candidates=(
                    pack.get("final_prompt_candidates")
                    or (getattr(node, "memory_routing_trace", None) or {}).get(
                        "final_prompt_candidates"
                    )
                    or []
                ),
                request_id=str(
                    getattr(visibility_pack, "request_id", "") or ""
                ),
            )
        except Exception as error:
            logger.warning(
                "Failed to record End2End candidate exposure for node %s: %s",
                getattr(node, "id", "unknown"),
                type(error).__name__,
            )
    elif callable(exposure_recorder) and visibility_pack is not None:
        try:
            exposure_recorder(
                node=node,
                visibility_pack=visibility_pack,
                injected_ref_ids=ref_ids,
            )
        except Exception as error:
            # Exposure bookkeeping must never fabricate adoption or silently
            # alter the already-sent prompt. Fail closed for later writeback.
            logger.warning(
                "Failed to record ExperienceContract exposure for node %s: %s",
                getattr(node, "id", "unknown"),
                type(error).__name__,
            )
    if not getattr(agent, "adoption_tracking_enabled", False):
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for rid in ref_ids:
        trace = _hybrid_trace_for_ref(pack, rid)
        record = {
            "source": source,
            "ref_id": rid,
            "stage": stage,
            "injected_at": ts,
            "adoption_mode": adoption_mode,
            "adoption_outcome": (
                "rejected_after_inspection"
                if adoption_mode == "strategy_candidate_inspection"
                and trace.get("selection_state") == "rejected"
                else "pending_analysis"
            ),
        }
        for key in (
            "retrieval_channel",
            "candidate_class",
            "gateway_sop_id",
            "supporting_transition_ids",
            "selection_reason",
            "selection_state",
        ):
            record[key] = copy.deepcopy(trace.get(key))
        node.adoption_log.append(record)
