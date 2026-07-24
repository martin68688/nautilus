from __future__ import annotations

import copy
import json
import stat
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier1_controlled_episodes import build  # noqa: E402
from evaluate_tier1_controlled_decisions import evaluate  # noqa: E402
from run_tier1_controlled_decisions import (  # noqa: E402
    build_request_plan,
    run_generation,
)
from tier1_controlled_runtime import (  # noqa: E402
    canonical_action_program,
    execute_canonical_action,
    runtime_actuation_receipt,
    static_actuation_receipt,
)
from verify_tier1_controlled_evaluation import verify_evaluation  # noqa: E402


CREATED_AT = "2026-07-21T00:00:00+08:00"
MODEL = "deepseek-test"
BASE_URL = "https://api.deepseek.com"


def _prepare_packet(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "legacy.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "legacy::001",
                "query_text": "An unrelated superseded controlled-decision record. " * 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet = tmp_path / "packet"
    build(packet, created_at=CREATED_AT, legacy_episode_root=legacy)
    episodes = {
        row["episode_id"]: row
        for row in (
            json.loads(line)
            for line in (packet / "episodes.jsonl").read_text().splitlines()
            if line
        )
    }
    return packet, legacy, episodes


def _response_for_action(request: dict, action_id: str, *, adopted: bool) -> str:
    refs = list(request["visible_memory_ids"]) if adopted else []
    return json.dumps(
        {
            "selected_action_id": action_id,
            "config_patch": request["candidate_action_map"][action_id],
            "memory_refs_used": refs,
            "memory_influence": "adopted" if adopted else "none",
            "rationale": "Controlled test response.",
        }
    )


def _run_fixture_generation(
    tmp_path: Path,
    *,
    full: bool,
) -> tuple[Path, Path, Path, dict[str, dict]]:
    packet, legacy, episodes = _prepare_packet(tmp_path)
    plan = build_request_plan(
        packet,
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.7,
        max_tokens=512,
        created_at=CREATED_AT,
        max_requests=None if full else 1,
        legacy_episode_root=legacy,
    )

    def caller(request):
        episode = episodes[request["episode_id"]]
        if request["condition"] == "NM":
            action_id = episode["neutral_action_id"]
            adopted = False
        else:
            memory = episode["memory_cells"][request["condition"]]
            action_id = memory["recommended_action_id"]
            adopted = True
        return _response_for_action(request, action_id, adopted=adopted), {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    generation = tmp_path / "generation"
    run_generation(
        plan,
        generation,
        allow_network=True,
        api_key="test-only-key",
        base_url=BASE_URL,
        workers=4 if full else 1,
        network_caller=caller,
        sleep_fn=lambda _seconds: None,
    )
    return packet, legacy, generation, episodes


def test_host_code_execution_is_distinct_from_historical_actuation() -> None:
    action = {
        "action_id": "action::selected",
        "config_patch": {"repair": "align rows by sample_id"},
    }
    memory = {
        "memory_id": "memory::source",
        "recommended_action_id": action["action_id"],
    }
    artifact = canonical_action_program(
        episode_id="episode::1",
        stage="debug",
        protocol_id="random-classification",
        selected_action_id=action["action_id"],
        config_patch=action["config_patch"],
        protocol_legal=True,
    )
    execution = execute_canonical_action(artifact)
    adopted_static = static_actuation_receipt(
        request_id="request::memory",
        condition="F11",
        memory=memory,
        recommended_action=action,
        selected_action=action,
        artifact=artifact,
    )
    adopted_runtime = runtime_actuation_receipt(
        static_receipt=adopted_static,
        code_execution_receipt=execution,
        artifact=artifact,
    )
    cold_static = static_actuation_receipt(
        request_id="request::cold",
        condition="NM",
        memory=None,
        recommended_action=None,
        selected_action=action,
        artifact=artifact,
    )
    cold_runtime = runtime_actuation_receipt(
        static_receipt=cold_static,
        code_execution_receipt=execution,
        artifact=artifact,
    )

    assert execution["execution_passed"] is True
    assert adopted_static["static_actuation_passed"] is True
    assert adopted_runtime["runtime_actuation_passed"] is True
    assert cold_static["static_actuation_passed"] is False
    assert cold_runtime["runtime_actuation_passed"] is False
    assert cold_runtime["code_execution_is_not_historical_actuation"] is True
    assert artifact["memory_metadata_embedded"] is False

    tampered = copy.deepcopy(artifact)
    tampered["source"] += "TAMPERED = True\n"
    assert execute_canonical_action(tampered)["execution_passed"] is False


def test_full_evaluator_emits_code_actuation_counterfactual_and_gate_evidence(
    tmp_path: Path,
) -> None:
    packet, legacy, generation, _episodes = _run_fixture_generation(
        tmp_path,
        full=True,
    )
    output = tmp_path / "evaluation"
    report = evaluate(
        packet,
        generation,
        output,
        created_at=CREATED_AT,
        require_full_matrix=True,
        legacy_episode_root=legacy,
    )

    assert report["matrix_complete"] is True
    assert report["evaluated_decision_count"] == 360
    assert report["code_execution_pass_count"] == 360
    assert report["code_file_count"] == 360
    assert report["memory_metadata_embedded_in_code_count"] == 0
    assert report["primary_metrics"]["invalid_influence_rate"] == 1.0
    assert report["primary_metrics"]["valid_knowledge_retention"] == 1.0
    assert report["primary_metrics"]["independent_result_path_retained_count"] == 6
    assert report["primary_metrics"]["independent_result_path_opportunity_count"] == 6
    assert report["system_summaries"]["full_decision_admissibility"][
        "invalid_influence_rate"
    ] == 0.0
    assert report["system_summaries"]["full_decision_admissibility"][
        "invalid_influence_opportunity_count"
    ] == 72
    assert report["system_summaries"]["full_decision_admissibility"][
        "unauthorized_prompt_exposure_count"
    ] == 0
    assert report["system_summaries"]["full_decision_admissibility"][
        "prompt_conditional_invalid_influence_rate"
    ] is None
    assert report["system_summaries"]["full_decision_admissibility"][
        "valid_knowledge_retention"
    ] == 1.0
    assert report["system_summaries"]["global_validity_bit"][
        "valid_knowledge_retention"
    ] == 0.0
    assert report["system_summaries"]["post_prompt_claim_tags"][
        "invalid_influence_rate"
    ] == 1.0
    assert report["system_summaries"]["post_prompt_claim_tags"][
        "prompt_conditional_invalid_influence_rate"
    ] == 1.0
    assert report["kill_gates_1_to_4"]["kill_gate_1"]["passed"] is None
    assert report["kill_gates_1_to_4"]["kill_gate_2"]["passed"] is True
    assert report["kill_gates_1_to_4"]["kill_gate_2"][
        "full_iir_denominator"
    ] == 72
    assert report["kill_gates_1_to_4"]["kill_gate_2"][
        "global_bit_iir_denominator"
    ] == 72
    assert report["kill_gates_1_to_4"]["kill_gate_3"]["passed"] is True
    assert report["kill_gates_1_to_4"]["kill_gate_4"]["passed"] is True
    assert report["writeback_semantics"][
        "promote_result_requires_historical_actuation"
    ] is False
    assert report["writeback_semantics"]["production_memory_write_performed"] is False

    decisions = [
        json.loads(line)
        for line in (output / "decision_receipts.jsonl").read_text().splitlines()
        if line
    ]
    assert sum(row["publish_adoption_path"]["eligible"] for row in decisions) == 72
    assert sum(row["publish_causal_path"]["eligible"] for row in decisions) == 72
    assert all(
        row["promote_result_path"]["historical_actuation_required"] is False
        and row["promote_result_path"]["derived_from_refs"] == []
        for row in decisions
    )
    assert stat.S_IMODE((output / "evaluation_report.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((output / "decision_receipts.jsonl").stat().st_mode) == 0o444
    verification = verify_evaluation(packet, generation, output)
    assert verification["verified"] is True
    assert verification["errors"] == []
    assert verification["decision_count"] == 360
    assert verification["independent_cold_result_evidence_count"] == 72
    assert verification["code_execution_schema_distinct_from_static_actuation"] is True
    assert verification["code_execution_schema_distinct_from_runtime_actuation"] is True

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        evaluate(
            packet,
            generation,
            output,
            created_at=CREATED_AT,
            legacy_episode_root=legacy,
        )


def test_partial_evaluation_preserves_cold_result_without_actuation(
    tmp_path: Path,
) -> None:
    packet, legacy, generation, _episodes = _run_fixture_generation(
        tmp_path,
        full=False,
    )
    output = tmp_path / "partial-evaluation"
    report = evaluate(
        packet,
        generation,
        output,
        created_at=CREATED_AT,
        require_full_matrix=False,
        legacy_episode_root=legacy,
    )
    row = json.loads(
        next(
            line
            for line in (output / "decision_receipts.jsonl").read_text().splitlines()
            if line
        )
    )

    assert report["matrix_complete"] is False
    assert report["evaluated_decision_count"] == 1
    assert report["kill_gates_1_to_4"]["kill_gate_2"]["passed"] is None
    assert row["code_execution_receipt"]["execution_passed"] is True
    assert row["static_actuation_receipt"]["static_actuation_passed"] is False
    assert row["runtime_actuation_receipt"]["runtime_actuation_passed"] is False
    assert row["current_run_node"]["recordable"] is True
    assert row["promote_result_path"]["historical_actuation_required"] is False
    assert row["promote_result_path"]["derived_from_refs"] == []


def test_evaluator_rejects_tampered_generation_artifact(tmp_path: Path) -> None:
    packet, legacy, generation, _episodes = _run_fixture_generation(
        tmp_path,
        full=False,
    )
    response_path = generation / "responses.jsonl"
    response_path.chmod(0o644)
    response_path.write_text(
        response_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="response JSONL hash mismatch"):
        evaluate(
            packet,
            generation,
            tmp_path / "rejected",
            created_at=CREATED_AT,
            require_full_matrix=False,
            legacy_episode_root=legacy,
        )


def test_evaluation_verifier_rejects_tampered_current_node_code(
    tmp_path: Path,
) -> None:
    packet, legacy, generation, _episodes = _run_fixture_generation(
        tmp_path,
        full=False,
    )
    output = tmp_path / "evaluation"
    evaluate(
        packet,
        generation,
        output,
        created_at=CREATED_AT,
        require_full_matrix=False,
        legacy_episode_root=legacy,
    )
    code_path = next((output / "code").glob("*.py"))
    code_path.chmod(0o644)
    code_path.write_text(
        code_path.read_text(encoding="utf-8") + "TAMPERED = True\n",
        encoding="utf-8",
    )

    verification = verify_evaluation(packet, generation, output)

    assert verification["verified"] is False
    assert any(error.startswith("code_source_hash:") for error in verification["errors"])
    assert "code_file_hashes_hash" in verification["errors"]
