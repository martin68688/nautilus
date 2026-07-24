from __future__ import annotations

import copy
import json
import os
import stat
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier1_controlled_episodes import build  # noqa: E402
from run_tier1_controlled_decisions import (  # noqa: E402
    AgentResponseValidationError,
    _safe_base_url_origin,
    build_request_plan,
    execute_request,
    load_deepseek_env_file,
    parse_agent_response,
    run_generation,
)


CREATED_AT = "2026-07-21T00:00:00+08:00"
MODEL = "deepseek-test"
BASE_URL = "https://api.deepseek.com"


def _write_legacy_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "legacy.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "legacy::001",
                "query_text": "A deliberately unrelated superseded benchmark string. " * 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _packet(tmp_path: Path) -> tuple[Path, Path]:
    legacy_root = _write_legacy_fixture(tmp_path / "legacy")
    packet = tmp_path / "packet"
    build(
        packet,
        created_at=CREATED_AT,
        legacy_episode_root=legacy_root,
    )
    return packet, legacy_root


def _plan(
    tmp_path: Path,
    *,
    max_requests: int | None = None,
) -> dict:
    packet, legacy_root = _packet(tmp_path)
    return build_request_plan(
        packet,
        model=MODEL,
        base_url=BASE_URL,
        temperature=0.7,
        max_tokens=512,
        created_at=CREATED_AT,
        max_requests=max_requests,
        legacy_episode_root=legacy_root,
    )


def _valid_payload(request: dict, *, adopt_memory: bool = False) -> dict:
    action_id = next(iter(request["candidate_action_map"]))
    visible = list(request["visible_memory_ids"])
    if adopt_memory and visible:
        refs = visible
        influence = "adopted"
    else:
        refs = []
        influence = "none"
    return {
        "selected_action_id": action_id,
        "config_patch": request["candidate_action_map"][action_id],
        "memory_refs_used": refs,
        "memory_influence": influence,
        "rationale": "This action best fits the supplied state.",
    }


def test_full_request_plan_freezes_360_requests_and_120_paired_prompts(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert plan["request_count"] == 360
    assert plan["full_matrix_request_count"] == 360
    assert plan["is_full_matrix"] is True
    assert plan["paired_prompt_mismatch_count"] == 0
    assert len({row["request_id"] for row in plan["requests"]}) == 360
    assert len(
        {row["identity"]["user_prompt_sha256"] for row in plan["requests"]}
    ) == 120
    assert plan["provider_seed_parameter_sent"] is False
    assert plan["agent_replicate_id_exposed_to_agent"] is False
    assert len(plan["base_url_endpoint_sha256"]) == 64

    paired = {}
    for request in plan["requests"]:
        key = (request["episode_id"], request["condition"])
        paired.setdefault(key, set()).add(request["user_prompt"])
        prompt = request["user_prompt"]
        assert "agent_seed" not in prompt
        assert "agent_replicate_id" not in prompt
        assert request["condition"] not in prompt
        assert "authority_valid" not in prompt
        assert "protocol_legal" not in prompt
        assert "oracle_action_id" not in prompt
    assert all(len(prompts) == 1 for prompts in paired.values())


def test_base_url_metadata_strips_paths_and_rejects_embedded_secrets() -> None:
    assert _safe_base_url_origin("https://api.example.test/v1") == (
        "https://api.example.test"
    )
    with pytest.raises(ValueError, match="must not contain credentials"):
        _safe_base_url_origin("https://user:secret@api.example.test/v1")
    with pytest.raises(ValueError, match="must not contain credentials"):
        _safe_base_url_origin("https://api.example.test/v1?token=secret")


def test_env_file_loader_whitelists_deepseek_keys_without_returning_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "UNRELATED_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=test-secret\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        "DEEPSEEK_MODEL=deepseek-test\n"
        "UNRELATED_SECRET=must-not-load\n",
        encoding="utf-8",
    )

    loaded = load_deepseek_env_file(env_file)

    assert loaded == {
        "DEEPSEEK_API_KEY": True,
        "DEEPSEEK_BASE_URL": True,
        "DEEPSEEK_MODEL": True,
    }
    assert "test-secret" not in json.dumps(loaded)
    assert "UNRELATED_SECRET" not in os.environ


def test_response_validator_requires_supplied_action_exact_patch_and_memory_refs(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, max_requests=4)
    memory_off = plan["requests"][0]
    memory_on = next(row for row in plan["requests"] if row["visible_memory_ids"])

    payload = _valid_payload(memory_off)
    parsed = parse_agent_response(json.dumps(payload), memory_off)
    assert parsed["selected_action_id"] == payload["selected_action_id"]
    assert parsed["memory_influence"] == "none"

    wrong_patch = copy.deepcopy(payload)
    wrong_patch["config_patch"] = {"forged": True}
    with pytest.raises(
        AgentResponseValidationError,
        match="config_patch_does_not_match_selected_action",
    ):
        parse_agent_response(wrong_patch, memory_off)

    unseen = _valid_payload(memory_on)
    unseen["memory_refs_used"] = ["memory::not-visible"]
    unseen["memory_influence"] = "adopted"
    with pytest.raises(
        AgentResponseValidationError,
        match="memory_refs_used_contains_unseen_memory",
    ):
        parse_agent_response(unseen, memory_on)

    extra = _valid_payload(memory_off)
    extra["hidden_gold"] = "F11"
    with pytest.raises(AgentResponseValidationError, match="response_keys"):
        parse_agent_response(extra, memory_off)


def test_request_retries_parse_failures_redacts_key_and_resumes_from_cache(
    tmp_path: Path,
) -> None:
    request = _plan(tmp_path, max_requests=1)["requests"][0]
    run_root = tmp_path / "run"
    (run_root / "raw_responses").mkdir(parents=True)
    (run_root / "attempt_logs").mkdir()
    secret = "super-secret-api-key"
    calls = []
    sleeps = []

    def caller(row):
        calls.append(row["request_id"])
        if len(calls) == 1:
            raise RuntimeError(f"temporary provider failure containing {secret}")
        if len(calls) == 2:
            invalid = _valid_payload(row)
            invalid["config_patch"] = {"wrong": True}
            return json.dumps(invalid), {"finish_reason": "stop"}
        return json.dumps(_valid_payload(row)), {"finish_reason": "stop"}

    result = execute_request(
        request,
        run_root,
        allow_network=True,
        network_caller=caller,
        api_key_for_redaction=secret,
        sleep_fn=sleeps.append,
    )

    assert result["source"] == "network"
    assert result["record"]["attempt_count"] == 3
    assert result["record"]["retry_count"] == 2
    assert sleeps == [1, 2]
    attempt_text = (
        run_root / "attempt_logs" / f"{request['request_id']}.json"
    ).read_text(encoding="utf-8")
    assert secret not in attempt_text
    assert "[REDACTED_API_KEY]" in attempt_text

    cached = execute_request(
        request,
        run_root,
        allow_network=False,
        network_caller=None,
    )
    assert cached["source"] == "saved"
    assert cached["record"] == result["record"]


def test_generation_root_is_resumable_then_finalized_exclusively(tmp_path: Path) -> None:
    plan = _plan(tmp_path, max_requests=4)
    run_root = tmp_path / "generation"

    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_generation(
            plan,
            run_root,
            allow_network=False,
            base_url=BASE_URL,
            workers=1,
        )
    progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
    assert progress["failure_count"] == 4
    assert not (run_root / "run_report.json").exists()

    def caller(request):
        return json.dumps(_valid_payload(request)), {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    report = run_generation(
        plan,
        run_root,
        allow_network=True,
        api_key="not-written-to-artifacts",
        base_url=BASE_URL,
        workers=2,
        network_caller=caller,
        sleep_fn=lambda _seconds: None,
    )

    assert report["request_count"] == report["response_count"] == 4
    assert report["network_response_count"] == 4
    assert report["error_count"] == 0
    assert len(report["run_hash"]) == 64
    assert sum(
        bool(line)
        for line in (run_root / "responses.jsonl").read_text().splitlines()
    ) == 4
    all_artifacts = "".join(
        path.read_text(encoding="utf-8")
        for path in run_root.rglob("*.json*")
    )
    assert "not-written-to-artifacts" not in all_artifacts
    assert stat.S_IMODE((run_root / "request_plan.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((run_root / "responses.jsonl").stat().st_mode) == 0o444
    assert stat.S_IMODE((run_root / "run_report.json").stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="finalized"):
        run_generation(
            plan,
            run_root,
            allow_network=False,
            base_url=BASE_URL,
            workers=1,
        )


def test_resume_rejects_request_plan_or_endpoint_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path, max_requests=1)
    run_root = tmp_path / "generation"
    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_generation(
            plan,
            run_root,
            allow_network=False,
            base_url=BASE_URL,
            workers=1,
        )

    drifted = copy.deepcopy(plan)
    drifted["temperature"] = 0.1
    with pytest.raises(ValueError, match="request plan identity mismatch"):
        run_generation(
            drifted,
            run_root,
            allow_network=False,
            base_url=BASE_URL,
            workers=1,
        )

    with pytest.raises(ValueError, match="base URL does not match"):
        run_generation(
            plan,
            tmp_path / "other-run",
            allow_network=False,
            base_url="https://different.example.test",
            workers=1,
        )
