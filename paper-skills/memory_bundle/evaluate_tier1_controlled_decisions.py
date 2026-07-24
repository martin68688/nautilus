from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_tier1_controlled_episodes import (
    AGENT_SEEDS,
    CONDITIONS,
    DEFAULT_LEGACY_EPISODE_ROOT,
)
from run_tier1_controlled_decisions import parse_agent_response
from schema import sha256_json
from tier1_controlled_runtime import (
    canonical_action_program,
    controlled_action_utility,
    execute_canonical_action,
    runtime_actuation_receipt,
    static_actuation_receipt,
)
from verify_tier1_controlled_episodes import verify_packet


EVALUATION_REPORT_SCHEMA = "decision_admissibility_tier1_evaluation_report_v2"
DECISION_RECEIPT_SCHEMA = "decision_admissibility_tier1_decision_receipt_v1"
COUNTERFACTUAL_RECEIPT_SCHEMA = (
    "decision_admissibility_tier1_counterfactual_receipt_v1"
)
SYSTEM_COMPOSITION_SCHEMA = "decision_admissibility_tier1_system_composition_v2"

SYSTEM_ROUTES = {
    "no_memory": "NM",
    "flat_relevance": "F10",
    "stage_router_only": "F10",
    "global_validity_bit": "NM",
    "authority_only": "F01",
    "full_decision_admissibility": "F11",
    "post_prompt_claim_tags": "F10",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_exclusive(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def _validate_hash(payload: Mapping[str, Any], hash_field: str) -> bool:
    return payload.get(hash_field) == sha256_json(
        {key: value for key, value in payload.items() if key != hash_field}
    )


def _load_bound_inputs(
    packet_root: Path,
    generation_root: Path,
    *,
    legacy_episode_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    packet_verification = verify_packet(
        packet_root,
        legacy_episode_root=legacy_episode_root,
    )
    if not packet_verification["verified"]:
        raise ValueError(
            f"Tier-1 packet verification failed: {packet_verification['errors']}"
        )
    packet_manifest = _read_json(packet_root / "manifest.json")
    episodes = _read_jsonl(packet_root / str(packet_manifest["episode_file"]))
    request_plan = _read_json(generation_root / "request_plan.json")
    generation_report = _read_json(generation_root / "run_report.json")
    response_path = generation_root / str(generation_report.get("responses_file") or "")
    responses = _read_jsonl(response_path)

    if not _validate_hash(request_plan, "request_plan_hash"):
        raise ValueError("Tier-1 request plan hash mismatch")
    if not _validate_hash(generation_report, "run_hash"):
        raise ValueError("Tier-1 generation report hash mismatch")
    if request_plan["packet_manifest_hash"] != packet_manifest["manifest_hash"]:
        raise ValueError("Generation plan is bound to a different packet manifest")
    if request_plan["packet_episode_file_sha256"] != packet_manifest[
        "episode_file_sha256"
    ]:
        raise ValueError("Generation plan is bound to a different episode file")
    if generation_report["request_plan_hash"] != request_plan["request_plan_hash"]:
        raise ValueError("Generation report is bound to a different request plan")
    if generation_report["packet_manifest_hash"] != packet_manifest["manifest_hash"]:
        raise ValueError("Generation report is bound to a different packet")
    runner_source = Path(__file__).resolve().with_name(
        "run_tier1_controlled_decisions.py"
    )
    if _sha256_file(runner_source) != generation_report.get("runner_source_sha256"):
        raise ValueError("Generation runner source hash mismatch")
    if _sha256_file(response_path) != generation_report["responses_file_sha256"]:
        raise ValueError("Generated response JSONL hash mismatch")
    if len(responses) != generation_report["response_count"]:
        raise ValueError("Generated response count mismatch")

    requests = {row["request_id"]: row for row in request_plan["requests"]}
    if len(requests) != len(request_plan["requests"]):
        raise ValueError("Duplicate request IDs in request plan")
    response_ids = [row.get("request_id") for row in responses]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Duplicate request IDs in generated responses")
    if set(response_ids) != set(requests):
        raise ValueError("Generated response IDs do not match the frozen request plan")
    raw_hashes = generation_report.get("raw_response_file_hashes") or {}
    if set(raw_hashes) != {
        f"{request_id}.json" for request_id in requests
    }:
        raise ValueError("Raw response file set does not match the request plan")
    responses_by_id = {row["request_id"]: row for row in responses}
    for request_id, request in requests.items():
        raw_path = generation_root / "raw_responses" / f"{request_id}.json"
        if _sha256_file(raw_path) != raw_hashes[raw_path.name]:
            raise ValueError(f"Raw response file hash mismatch: {request_id}")
        raw_record = _read_json(raw_path)
        if not _validate_hash(raw_record, "record_hash"):
            raise ValueError(f"Raw response record hash mismatch: {request_id}")
        if raw_record.get("identity_hash") != request.get("identity_hash"):
            raise ValueError(f"Raw response identity mismatch: {request_id}")
        parsed = parse_agent_response(raw_record.get("raw_response"), request)
        if parsed != raw_record.get("parsed_response"):
            raise ValueError(f"Raw response parse binding mismatch: {request_id}")
        expected_response = {
            "request_id": request_id,
            "source": responses_by_id[request_id]["source"],
            **raw_record,
        }
        if responses_by_id[request_id] != expected_response:
            raise ValueError(f"Response JSONL/raw cache mismatch: {request_id}")
    return packet_manifest, episodes, request_plan, generation_report, responses


def _selected_action(episode: Mapping[str, Any], action_id: str) -> dict[str, Any]:
    matches = [
        dict(action)
        for action in episode.get("action_candidates") or []
        if action.get("action_id") == action_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Selected action {action_id} is not unique in {episode.get('episode_id')}"
        )
    return matches[0]


def _safe_code_name(request_id: str) -> str:
    return request_id.replace("::", "__").replace(":", "_") + ".py"


def _base_decision_receipt(
    *,
    episode: Mapping[str, Any],
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = response["parsed_response"]
    selected = _selected_action(episode, parsed["selected_action_id"])
    condition = str(request["condition"])
    memory = None if condition == "NM" else episode["memory_cells"][condition]
    recommended_action = (
        None
        if memory is None
        else _selected_action(episode, memory["recommended_action_id"])
    )
    artifact = canonical_action_program(
        episode_id=episode["episode_id"],
        stage=episode["stage"],
        protocol_id=episode["protocol_id"],
        selected_action_id=selected["action_id"],
        config_patch=selected["config_patch"],
        protocol_legal=selected["protocol_legal"],
    )
    execution = execute_canonical_action(artifact)
    static = static_actuation_receipt(
        request_id=request["request_id"],
        condition=condition,
        memory=memory,
        recommended_action=recommended_action,
        selected_action=selected,
        artifact=artifact,
    )
    runtime = runtime_actuation_receipt(
        static_receipt=static,
        code_execution_receipt=execution,
        artifact=artifact,
    )
    current_node_id = f"controlled-node::{request['request_id']}"
    receipt: dict[str, Any] = {
        "schema": DECISION_RECEIPT_SCHEMA,
        "request_id": request["request_id"],
        "episode_id": episode["episode_id"],
        "source_task_id": episode["source_task_id"],
        "source_run_id": episode["source_run_id"],
        "source_episode_id": episode["source_episode_id"],
        "target_task_id": episode["target_task_id"],
        "stage": episode["stage"],
        "domain": episode["domain"],
        "condition": condition,
        "agent_replicate_id": request["agent_replicate_id"],
        "selected_action_id": selected["action_id"],
        "selected_action_role": selected["role"],
        "selected_config_patch": selected["config_patch"],
        "selected_config_patch_hash": artifact["config_patch_hash"],
        "protocol_legal": selected["protocol_legal"],
        "controlled_action_utility": controlled_action_utility(selected),
        "oracle_action_selected": selected["action_id"] == episode["oracle_action_id"],
        "memory_exposed": memory is not None,
        "memory_id": str((memory or {}).get("memory_id") or ""),
        "memory_recommended_action_id": str(
            (memory or {}).get("recommended_action_id") or ""
        ),
        "granularity_match": (
            None if memory is None else memory["granularity_match"]
        ),
        "authority_valid": None if memory is None else memory["authority_valid"],
        "agent_reported_memory_influence": parsed["memory_influence"],
        "agent_reported_memory_refs": parsed["memory_refs_used"],
        "agent_rationale_sha256": hashlib.sha256(
            parsed["rationale"].encode("utf-8")
        ).hexdigest(),
        "code_artifact": {
            key: value for key, value in artifact.items() if key != "source"
        },
        "code_file": f"code/{_safe_code_name(request['request_id'])}",
        "code_execution_receipt": execution,
        "static_actuation_receipt": static,
        "runtime_actuation_receipt": runtime,
        "counterfactual_receipt_hash": "",
        "current_run_node": {
            "subject_artifact_id": current_node_id,
            "recordable": execution["execution_passed"],
            "code_execution_receipt_hash": execution["receipt_hash"],
        },
        "promote_result_path": {
            "subject_artifact_id": current_node_id,
            "historical_actuation_required": False,
            "derived_from_refs": [],
            "controlled_positive_result_candidate": bool(
                execution["execution_passed"]
                and selected["protocol_legal"]
                and selected["role"] == "oracle"
            ),
            "production_result_fact_written": False,
            "reason": "controlled_tier1_has_no_real_target_metric",
        },
        "publish_adoption_path": {
            "eligible": False,
            "historical_memory_ref": "",
            "requires_static_and_runtime_actuation": True,
        },
        "publish_causal_path": {
            "eligible": False,
            "historical_memory_ref": "",
            "requires_counterfactual_actuation": True,
        },
        "receipt_hash": "",
    }
    return receipt, artifact


def _attach_counterfactuals(
    receipts: list[dict[str, Any]],
    *,
    require_full_matrix: bool,
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in receipts:
        key = (row["episode_id"], int(row["agent_replicate_id"]))
        by_pair[key][row["condition"]] = row
    counterfactuals: list[dict[str, Any]] = []
    for key, conditions in sorted(by_pair.items()):
        if require_full_matrix and set(conditions) != set(CONDITIONS):
            raise ValueError(f"Incomplete condition matrix for {key}: {sorted(conditions)}")
        baseline = conditions.get("NM")
        for condition, row in sorted(conditions.items()):
            if condition == "NM" or baseline is None:
                continue
            action_changed = row["selected_action_id"] != baseline["selected_action_id"]
            code_changed = (
                row["code_artifact"]["source_sha256"]
                != baseline["code_artifact"]["source_sha256"]
            )
            runtime_changed = (
                row["code_execution_receipt"]["runtime_events_hash"]
                != baseline["code_execution_receipt"]["runtime_events_hash"]
            )
            changed = action_changed or code_changed or runtime_changed
            static_passed = row["static_actuation_receipt"][
                "static_actuation_passed"
            ]
            runtime_passed = row["runtime_actuation_receipt"][
                "runtime_actuation_passed"
            ]
            counterfactual_passed = bool(static_passed and runtime_passed and changed)
            authority_valid = row["authority_valid"] is True
            granularity_match = row["granularity_match"] is True
            adoption_eligible = bool(
                authority_valid
                and granularity_match
                and static_passed
                and runtime_passed
            )
            causal_eligible = bool(adoption_eligible and counterfactual_passed)
            outcome_delta = (
                row["controlled_action_utility"]
                - baseline["controlled_action_utility"]
            )
            counterfactual: dict[str, Any] = {
                "schema": COUNTERFACTUAL_RECEIPT_SCHEMA,
                "episode_id": row["episode_id"],
                "agent_replicate_id": row["agent_replicate_id"],
                "condition": condition,
                "memory_request_id": row["request_id"],
                "memory_off_request_id": baseline["request_id"],
                "memory_id": row["memory_id"],
                "action_changed": action_changed,
                "code_changed": code_changed,
                "runtime_events_changed": runtime_changed,
                "historical_memory_changed_current_node": changed,
                "static_actuation_passed": static_passed,
                "runtime_actuation_passed": runtime_passed,
                "counterfactual_actuation_passed": counterfactual_passed,
                "controlled_outcome_delta": outcome_delta,
                "controlled_efficacy_improved": outcome_delta > 0,
                "real_target_metric_claimed": False,
                "receipt_hash": "",
            }
            counterfactual["receipt_hash"] = sha256_json(
                {
                    field: value
                    for field, value in counterfactual.items()
                    if field != "receipt_hash"
                }
            )
            row["counterfactual_receipt_hash"] = counterfactual["receipt_hash"]
            row["publish_adoption_path"] = {
                "eligible": adoption_eligible,
                "historical_memory_ref": (
                    row["memory_id"] if adoption_eligible else ""
                ),
                "requires_static_and_runtime_actuation": True,
                "authority_valid": authority_valid,
                "granularity_match": granularity_match,
            }
            row["publish_causal_path"] = {
                "eligible": causal_eligible,
                "historical_memory_ref": row["memory_id"] if causal_eligible else "",
                "requires_counterfactual_actuation": True,
                "counterfactual_actuation_passed": counterfactual_passed,
                "controlled_efficacy_improved": outcome_delta > 0,
                "real_target_metric_claimed": False,
            }
            counterfactuals.append(counterfactual)
    for row in receipts:
        row["receipt_hash"] = sha256_json(
            {key: value for key, value in row.items() if key != "receipt_hash"}
        )
    return counterfactuals


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _cell_metrics(
    receipts: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counterfactual_by_request = {
        row["memory_request_id"]: row for row in counterfactuals
    }
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in receipts if row["condition"] == condition]
        paired = [
            counterfactual_by_request[row["request_id"]]
            for row in rows
            if row["request_id"] in counterfactual_by_request
        ]
        output[condition] = {
            "decision_count": len(rows),
            "oracle_action_count": sum(row["oracle_action_selected"] for row in rows),
            "oracle_action_rate": _ratio(
                sum(row["oracle_action_selected"] for row in rows), len(rows)
            ),
            "mean_controlled_action_utility": (
                sum(float(row["controlled_action_utility"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "static_actuation_count": sum(
                row["static_actuation_receipt"]["static_actuation_passed"]
                for row in rows
            ),
            "runtime_actuation_count": sum(
                row["runtime_actuation_receipt"]["runtime_actuation_passed"]
                for row in rows
            ),
            "paired_counterfactual_count": len(paired),
            "action_change_count": sum(row["action_changed"] for row in paired),
            "code_change_count": sum(row["code_changed"] for row in paired),
            "counterfactual_actuation_count": sum(
                row["counterfactual_actuation_passed"] for row in paired
            ),
        }
    return output


def _compose_systems(
    receipts: Sequence[Mapping[str, Any]],
    episodes_by_id: Mapping[str, Mapping[str, Any]],
    *,
    require_full_matrix: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in receipts:
        by_pair[(row["episode_id"], int(row["agent_replicate_id"]))][
            row["condition"]
        ] = row
    composed: list[dict[str, Any]] = []
    for (episode_id, replicate), conditions in sorted(by_pair.items()):
        if require_full_matrix and set(conditions) != set(CONDITIONS):
            raise ValueError(f"Cannot compose systems from incomplete pair: {episode_id}:{replicate}")
        baseline = conditions.get("NM")
        if baseline is None:
            continue
        episode = episodes_by_id[episode_id]
        invalid_gateway_challenge = any(
            cell.get("authority_valid") is False
            for cell in (episode.get("memory_cells") or {}).values()
        )
        for system, route in SYSTEM_ROUTES.items():
            selected = conditions.get(route)
            if selected is None:
                continue
            invalid_exposure = route in {"F00", "F10"}
            action_changed = selected["selected_action_id"] != baseline["selected_action_id"]
            retained_valid_use = bool(
                route == "F11"
                and selected["runtime_actuation_receipt"][
                    "runtime_actuation_passed"
                ]
            )
            composed.append(
                {
                    "schema": SYSTEM_COMPOSITION_SCHEMA,
                    "system": system,
                    "episode_id": episode_id,
                    "agent_replicate_id": replicate,
                    "routed_condition": route,
                    "selected_action_id": selected["selected_action_id"],
                    "oracle_action_selected": selected["oracle_action_selected"],
                    "controlled_action_utility": selected[
                        "controlled_action_utility"
                    ],
                    "invalid_claim_gateway_exposure": invalid_gateway_challenge,
                    "unauthorized_prompt_exposure": invalid_exposure,
                    "invalid_claim_changed_action": bool(
                        invalid_exposure and action_changed
                    ),
                    "valid_matched_knowledge_retained": retained_valid_use,
                    "stage_mismatch_changed_action": bool(
                        route == "F01" and action_changed
                    ),
                    "composition_is_host_routing_over_frozen_responses": True,
                }
            )
        oracle_action = _selected_action(episode, episode["oracle_action_id"])
        composed.append(
            {
                "schema": SYSTEM_COMPOSITION_SCHEMA,
                "system": "oracle",
                "episode_id": episode_id,
                "agent_replicate_id": replicate,
                "routed_condition": "HOST_ORACLE",
                "selected_action_id": oracle_action["action_id"],
                "oracle_action_selected": True,
                "controlled_action_utility": controlled_action_utility(oracle_action),
                "invalid_claim_gateway_exposure": invalid_gateway_challenge,
                "unauthorized_prompt_exposure": False,
                "invalid_claim_changed_action": False,
                "valid_matched_knowledge_retained": True,
                "stage_mismatch_changed_action": False,
                "composition_is_host_routing_over_frozen_responses": True,
            }
        )
    summaries: dict[str, Any] = {}
    for system in (*SYSTEM_ROUTES, "oracle"):
        rows = [row for row in composed if row["system"] == system]
        invalid_opportunities = sum(
            row["invalid_claim_gateway_exposure"] for row in rows
        )
        prompt_exposures = sum(row["unauthorized_prompt_exposure"] for row in rows)
        invalid_influence = sum(row["invalid_claim_changed_action"] for row in rows)
        valid_opportunities = len(rows)
        valid_retained = sum(row["valid_matched_knowledge_retained"] for row in rows)
        summaries[system] = {
            "decision_count": len(rows),
            "unauthorized_prompt_exposure_count": prompt_exposures,
            "unauthorized_prompt_exposure_rate": _ratio(
                prompt_exposures, len(rows)
            ),
            "invalid_claim_gateway_exposure_count": invalid_opportunities,
            "invalid_influence_count": invalid_influence,
            "invalid_influence_opportunity_count": invalid_opportunities,
            "invalid_influence_rate": _ratio(
                invalid_influence, invalid_opportunities
            ),
            "prompt_conditional_invalid_influence_rate": _ratio(
                invalid_influence, prompt_exposures
            ),
            "valid_knowledge_opportunity_count": valid_opportunities,
            "valid_knowledge_retained_count": valid_retained,
            "valid_knowledge_retention": _ratio(valid_retained, valid_opportunities),
            "oracle_action_rate": _ratio(
                sum(row["oracle_action_selected"] for row in rows), len(rows)
            ),
            "mean_controlled_action_utility": (
                sum(float(row["controlled_action_utility"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "stage_mismatch_influence_count": sum(
                row["stage_mismatch_changed_action"] for row in rows
            ),
        }
    return composed, summaries


def _primary_metrics(
    receipts: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_request = {row["request_id"]: row for row in receipts}
    invalid = [row for row in counterfactuals if row["condition"] in {"F00", "F10"}]
    invalid_influence = sum(row["action_changed"] for row in invalid)
    f11 = [row for row in receipts if row["condition"] == "F11"]
    valid_retained = sum(
        row["runtime_actuation_receipt"]["runtime_actuation_passed"] for row in f11
    )
    f01 = [row for row in counterfactuals if row["condition"] == "F01"]
    clean_unexposed = [
        row
        for row in receipts
        if row["condition"] == "NM"
        and row["stage"] == "governance"
        and row["episode_id"].split("::")[1] in {"natural-image", "medical-image"}
    ]
    return {
        "invalid_influence_count": invalid_influence,
        "invalid_influence_opportunity_count": len(invalid),
        "invalid_influence_rate": _ratio(invalid_influence, len(invalid)),
        "valid_knowledge_retained_count": valid_retained,
        "valid_knowledge_opportunity_count": len(f11),
        "valid_knowledge_retention": _ratio(valid_retained, len(f11)),
        "stage_mismatch_action_change_count": sum(row["action_changed"] for row in f01),
        "stage_mismatch_opportunity_count": len(f01),
        "stage_mismatch_action_change_rate": _ratio(
            sum(row["action_changed"] for row in f01), len(f01)
        ),
        "independent_result_path_retained_count": sum(
            row["current_run_node"]["recordable"]
            and row["static_actuation_receipt"]["static_actuation_passed"] is False
            and row["runtime_actuation_receipt"]["runtime_actuation_passed"] is False
            and row["promote_result_path"]["historical_actuation_required"] is False
            for row in clean_unexposed
        ),
        "independent_result_path_opportunity_count": len(clean_unexposed),
        "governance_clean_result_oracle_selection_count": sum(
            row["oracle_action_selected"] for row in clean_unexposed
        ),
        "governance_clean_result_oracle_selection_rate": _ratio(
            sum(row["oracle_action_selected"] for row in clean_unexposed),
            len(clean_unexposed),
        ),
        "host_receipt_count": len(by_request),
    }


def _gate_report(
    receipts: Sequence[Mapping[str, Any]],
    system_summaries: Mapping[str, Mapping[str, Any]],
    *,
    matrix_complete: bool,
) -> dict[str, Any]:
    if not matrix_complete:
        return {
            f"kill_gate_{index}": {
                "status": "not_assessed_partial_matrix",
                "passed": None,
            }
            for index in range(1, 5)
        }
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in receipts:
        by_pair[(row["episode_id"], int(row["agent_replicate_id"]))][
            row["condition"]
        ] = row
    stage_pairs = [
        (conditions["F01"], conditions["F11"])
        for conditions in by_pair.values()
        if {"F01", "F11"}.issubset(conditions)
    ]
    stage_action_diff = sum(left["selected_action_id"] != right["selected_action_id"] for left, right in stage_pairs)
    stage_code_diff = sum(
        left["code_artifact"]["source_sha256"]
        != right["code_artifact"]["source_sha256"]
        for left, right in stage_pairs
    )
    full = system_summaries["full_decision_admissibility"]
    global_bit = system_summaries["global_validity_bit"]
    post_tags = system_summaries["post_prompt_claim_tags"]
    gate2_passed = bool(
        full["invalid_influence_opportunity_count"] > 0
        and full["invalid_influence_opportunity_count"]
        == global_bit["invalid_influence_opportunity_count"]
        and full["invalid_influence_rate"]
        <= global_bit["invalid_influence_rate"]
        and full["valid_knowledge_retention"]
        > global_bit["valid_knowledge_retention"]
    )
    gate3_passed = bool(stage_pairs and stage_action_diff > 0 and stage_code_diff > 0)
    gate4_passed = bool(
        full["unauthorized_prompt_exposure_count"] == 0
        and post_tags["unauthorized_prompt_exposure_count"] > 0
        and post_tags["invalid_influence_count"] > 0
    )
    return {
        "kill_gate_1": {
            "name": "problem_prevalence",
            "status": "pending_real_decision_prevalence_audit",
            "passed": None,
            "reason": "Controlled episodes establish mechanism, not real-corpus prevalence.",
        },
        "kill_gate_2": {
            "name": "claim_level_vs_global_bit",
            "status": "pass" if gate2_passed else "fail",
            "passed": gate2_passed,
            "iir_denominator_layer": "gateway_challenged_decision",
            "full_iir_numerator": full["invalid_influence_count"],
            "full_iir_denominator": full["invalid_influence_opportunity_count"],
            "full_iir": full["invalid_influence_rate"],
            "global_bit_iir_numerator": global_bit["invalid_influence_count"],
            "global_bit_iir_denominator": global_bit[
                "invalid_influence_opportunity_count"
            ],
            "global_bit_iir": global_bit["invalid_influence_rate"],
            "full_vkr": full["valid_knowledge_retention"],
            "global_bit_vkr": global_bit["valid_knowledge_retention"],
        },
        "kill_gate_3": {
            "name": "stage_utility",
            "status": "pass" if gate3_passed else "fail",
            "passed": gate3_passed,
            "paired_decision_count": len(stage_pairs),
            "action_difference_count": stage_action_diff,
            "code_difference_count": stage_code_diff,
        },
        "kill_gate_4": {
            "name": "pre_prompt_visibility_necessity",
            "status": "pass" if gate4_passed else "fail",
            "passed": gate4_passed,
            "full_unauthorized_prompt_exposure_count": full[
                "unauthorized_prompt_exposure_count"
            ],
            "post_prompt_tag_exposure_count": post_tags[
                "unauthorized_prompt_exposure_count"
            ],
            "post_prompt_tag_invalid_influence_count": post_tags[
                "invalid_influence_count"
            ],
        },
    }


def evaluate(
    packet_root: str | Path,
    generation_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
    require_full_matrix: bool = True,
    legacy_episode_root: str | Path = DEFAULT_LEGACY_EPISODE_ROOT,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    generation_root = Path(generation_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse Tier-1 evaluation root: {output_root}")
    (
        packet_manifest,
        episodes,
        request_plan,
        generation_report,
        responses,
    ) = _load_bound_inputs(
        packet_root,
        generation_root,
        legacy_episode_root=legacy_episode_root,
    )
    expected_full_count = int(packet_manifest["validation"]["planned_agent_run_count"])
    matrix_complete = bool(
        request_plan["is_full_matrix"]
        and len(responses) == expected_full_count
        and request_plan["conditions"] == list(CONDITIONS)
        and request_plan["agent_replicate_ids"] == list(AGENT_SEEDS)
    )
    if require_full_matrix and not matrix_complete:
        raise ValueError(
            f"Tier-1 evaluation requires the full {expected_full_count}-request matrix"
        )
    episodes_by_id = {row["episode_id"]: row for row in episodes}
    requests_by_id = {
        row["request_id"]: row for row in request_plan["requests"]
    }
    receipts: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for response in sorted(responses, key=lambda row: row["request_id"]):
        request = requests_by_id[response["request_id"]]
        episode = episodes_by_id[request["episode_id"]]
        receipt, artifact = _base_decision_receipt(
            episode=episode,
            request=request,
            response=response,
        )
        if not receipt["code_execution_receipt"]["execution_passed"]:
            raise ValueError(f"Host canonical execution failed: {request['request_id']}")
        receipts.append(receipt)
        artifacts[request["request_id"]] = artifact
    counterfactuals = _attach_counterfactuals(
        receipts,
        require_full_matrix=require_full_matrix,
    )
    composed, system_summaries = _compose_systems(
        receipts,
        episodes_by_id,
        require_full_matrix=require_full_matrix,
    )
    cell_metrics = _cell_metrics(receipts, counterfactuals)
    primary = _primary_metrics(receipts, counterfactuals)
    gates = _gate_report(
        receipts,
        system_summaries,
        matrix_complete=matrix_complete,
    )

    output_root.mkdir(parents=True)
    code_hashes: dict[str, str] = {}
    for request_id, artifact in sorted(artifacts.items()):
        relative = Path("code") / _safe_code_name(request_id)
        path = output_root / relative
        _write_text_exclusive(path, artifact["source"])
        code_hashes[str(relative)] = _sha256_file(path)
    decision_path = output_root / "decision_receipts.jsonl"
    counterfactual_path = output_root / "counterfactual_receipts.jsonl"
    systems_path = output_root / "system_composition.jsonl"
    _write_jsonl_exclusive(decision_path, receipts)
    _write_jsonl_exclusive(counterfactual_path, counterfactuals)
    _write_jsonl_exclusive(systems_path, composed)
    report: dict[str, Any] = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "created_at": str(created_at),
        "packet_manifest_hash": packet_manifest["manifest_hash"],
        "generation_run_hash": generation_report["run_hash"],
        "request_plan_hash": request_plan["request_plan_hash"],
        "matrix_complete": matrix_complete,
        "expected_full_request_count": expected_full_count,
        "evaluated_decision_count": len(receipts),
        "code_execution_pass_count": sum(
            row["code_execution_receipt"]["execution_passed"] for row in receipts
        ),
        "code_file_count": len(code_hashes),
        "code_file_hashes_hash": sha256_json(code_hashes),
        "memory_metadata_embedded_in_code_count": sum(
            row["code_artifact"]["memory_metadata_embedded"] for row in receipts
        ),
        "decision_receipts_file": decision_path.name,
        "decision_receipts_file_sha256": _sha256_file(decision_path),
        "counterfactual_receipts_file": counterfactual_path.name,
        "counterfactual_receipts_file_sha256": _sha256_file(counterfactual_path),
        "system_composition_file": systems_path.name,
        "system_composition_file_sha256": _sha256_file(systems_path),
        "cell_metrics": cell_metrics,
        "primary_metrics": primary,
        "system_summaries": system_summaries,
        "kill_gates_1_to_4": gates,
        "writeback_semantics": {
            "current_node_is_memory_subject": True,
            "code_execution_receipt_is_target_node_evidence": True,
            "static_runtime_actuation_are_historical_influence_evidence": True,
            "promote_result_requires_historical_actuation": False,
            "promote_result_derived_from_refs": [],
            "adoption_requires_static_and_runtime_actuation": True,
            "causal_requires_counterfactual_actuation": True,
            "production_memory_write_performed": False,
        },
        "metric_estimands": {
            "primary_cell_iir_denominator": "paired Prompt-exposed invalid-cell decisions",
            "system_iir_denominator": "paired decisions whose gateway candidate set contains an invalid Claim",
            "unauthorized_prompt_exposure_is_separate": True,
            "prompt_conditional_iir_is_null_when_prompt_exposure_count_is_zero": True,
        },
        "limitations": [
            "Controlled action utility is host-owned synthetic gold, not a real target-task metric.",
            "System baselines are host routing over frozen response cells, not additional end-to-end model calls.",
            "DeepSeek exposes no provider RNG seed in the bound API contract; 101/202/303 are paired replicate IDs.",
            "Kill Gate 1 requires a separate real-decision prevalence audit.",
            "System-level IIR uses gateway-challenged decisions as its denominator; Prompt exposure and Prompt-conditional influence are reported separately.",
        ],
        "evaluator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "runtime_source_sha256": _sha256_file(
            Path(__file__).resolve().with_name("tier1_controlled_runtime.py")
        ),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    _write_json_exclusive(output_root / "evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WP8 Tier-1 responses with host-owned code/runtime receipts."
    )
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--legacy-episode-root",
        type=Path,
        default=DEFAULT_LEGACY_EPISODE_ROOT,
    )
    args = parser.parse_args()
    report = evaluate(
        args.packet_root,
        args.generation_root,
        args.output_root,
        created_at=args.created_at,
        require_full_matrix=not args.allow_partial,
        legacy_episode_root=args.legacy_episode_root,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
