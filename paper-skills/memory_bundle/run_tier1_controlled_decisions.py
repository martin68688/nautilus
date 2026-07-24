from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from build_tier1_controlled_episodes import (
    AGENT_SEEDS,
    CONDITIONS,
    DEFAULT_LEGACY_EPISODE_ROOT,
    HIDDEN_AGENT_KEYS,
    project_agent_view,
)
from schema import sha256_json, write_json_atomic
from verify_tier1_controlled_episodes import verify_packet


REQUEST_PLAN_SCHEMA = "decision_admissibility_tier1_request_plan_v1"
RAW_RESPONSE_SCHEMA = "decision_admissibility_tier1_raw_response_v1"
ATTEMPT_LOG_SCHEMA = "decision_admissibility_tier1_attempt_log_v1"
RUN_REPORT_SCHEMA = "decision_admissibility_tier1_generation_report_v1"
RESPONSE_SCHEMA = "decision_admissibility_tier1_agent_response_v1"
NETWORK_ATTEMPTS_PER_INVOCATION = 3
MAX_TOTAL_NETWORK_ATTEMPTS = 9
INFLUENCE_VALUES = {"none", "considered_not_adopted", "adopted"}
RESPONSE_KEYS = {
    "selected_action_id",
    "config_patch",
    "memory_refs_used",
    "memory_influence",
    "rationale",
}
DEEPSEEK_ENV_KEYS = {
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
}

SYSTEM_PROMPT = """You are the decision agent in a controlled machine-learning engineering experiment.
Select exactly one supplied action candidate for the current decision state. A memory item, when present, comes from a different task in the same domain; decide for yourself whether it is useful. Do not invent action IDs or edit a candidate's config_patch. Return one JSON object only, with no Markdown or surrounding commentary.

The JSON object must contain exactly these keys:
- selected_action_id: one supplied action_id
- config_patch: an exact copy of that selected candidate's config_patch
- memory_refs_used: a JSON list containing only supplied memory_id values that affected the choice
- memory_influence: one of "none", "considered_not_adopted", or "adopted"
- rationale: a concise explanation of the decision

If no memory is supplied, memory_refs_used must be [] and memory_influence must be "none". If memory_influence is "adopted", memory_refs_used must name the supplied memory item. Produce valid JSON."""


class AgentResponseValidationError(ValueError):
    """Raised when a provider response violates the controlled output contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_base_url_origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("DeepSeek base URL must be an HTTP(S) URL with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "DeepSeek base URL must not contain credentials, query parameters, or fragments"
        )
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _base_url_endpoint_hash(base_url: str) -> str:
    _safe_base_url_origin(base_url)
    return _sha256_text(base_url.rstrip("/"))


def _redact_error(error: BaseException, *, api_key: str = "") -> str:
    message = f"{type(error).__name__}: {error}"
    if api_key:
        message = message.replace(api_key, "[REDACTED_API_KEY]")
    return message[:2000]


def load_deepseek_env_file(path: str | Path) -> dict[str, bool]:
    """Load only the three DeepSeek settings without logging their values."""

    from dotenv import dotenv_values

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DeepSeek env file does not exist: {source}")
    values = dotenv_values(source)
    loaded: dict[str, bool] = {}
    for key in sorted(DEEPSEEK_ENV_KEYS):
        value = values.get(key)
        should_load = bool(value) and not os.environ.get(key)
        if should_load:
            os.environ[key] = str(value)
        loaded[key] = should_load
    return loaded


def _walk_mapping_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_mapping_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mapping_keys(item)


def build_user_prompt(agent_view: Mapping[str, Any]) -> str:
    payload = {
        "instruction": "Choose one action and return the required JSON object.",
        "decision": dict(agent_view),
        "response_example_shape": {
            "selected_action_id": "action::<supplied-id>",
            "config_patch": {"copy": "the selected candidate patch exactly"},
            "memory_refs_used": [],
            "memory_influence": "none",
            "rationale": "brief reason",
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)


def _request_spec(
    episode: Mapping[str, Any],
    *,
    condition: str,
    agent_seed: int,
    packet_manifest_hash: str,
    packet_episode_file_sha256: str,
    base_url_origin: str,
    base_url_endpoint_sha256: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    view = project_agent_view(
        episode,
        condition=condition,
        agent_seed=agent_seed,
    )
    user_prompt = build_user_prompt(view)
    candidates = {
        str(action["action_id"]): dict(action["config_patch"])
        for action in view["action_candidates"]
    }
    visible_memory_ids = [
        str(memory["memory_id"]) for memory in view["memory_context"]
    ]
    identity = {
        "packet_manifest_hash": packet_manifest_hash,
        "packet_episode_file_sha256": packet_episode_file_sha256,
        "episode_id": episode["episode_id"],
        "episode_hash": episode["episode_hash"],
        "condition": condition,
        "agent_replicate_id": agent_seed,
        "agent_view_hash": view["view_hash"],
        "system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
        "user_prompt_sha256": _sha256_text(user_prompt),
        "base_url_origin": base_url_origin,
        "base_url_endpoint_sha256": base_url_endpoint_sha256,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": "json_object",
        "network_attempts_per_invocation": NETWORK_ATTEMPTS_PER_INVOCATION,
        "max_total_network_attempts": MAX_TOTAL_NETWORK_ATTEMPTS,
        "provider_seed_parameter_sent": False,
    }
    identity_hash = sha256_json(identity)
    return {
        "request_id": f"tier1-request::{identity_hash[:24]}",
        "identity": identity,
        "identity_hash": identity_hash,
        "episode_id": episode["episode_id"],
        "stage": episode["stage"],
        "domain": episode["domain"],
        "condition": condition,
        "agent_replicate_id": agent_seed,
        "agent_view_hash": view["view_hash"],
        "candidate_action_map": candidates,
        "visible_memory_ids": visible_memory_ids,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "provider_seed_parameter_sent": False,
    }


def build_request_plan(
    packet_root: str | Path,
    *,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    created_at: str,
    conditions: Sequence[str] = CONDITIONS,
    agent_seeds: Sequence[int] = AGENT_SEEDS,
    episode_ids: Sequence[str] | None = None,
    max_requests: int | None = None,
    legacy_episode_root: str | Path = DEFAULT_LEGACY_EPISODE_ROOT,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    verification = verify_packet(
        packet_root,
        legacy_episode_root=legacy_episode_root,
    )
    if not verification["verified"]:
        raise ValueError(
            f"Tier-1 packet verification failed: {verification['errors']}"
        )
    manifest = _read_json(packet_root / "manifest.json")
    base_url_origin = _safe_base_url_origin(base_url)
    base_url_endpoint_sha256 = _base_url_endpoint_hash(base_url)
    episodes = _read_jsonl(packet_root / str(manifest["episode_file"]))
    selected_ids = set(episode_ids or [])
    if selected_ids:
        unknown = selected_ids - {str(row["episode_id"]) for row in episodes}
        if unknown:
            raise ValueError(f"Unknown Tier-1 episode IDs: {sorted(unknown)}")
        episodes = [row for row in episodes if row["episode_id"] in selected_ids]
    requested_conditions = tuple(conditions)
    requested_seeds = tuple(int(value) for value in agent_seeds)
    if not requested_conditions or any(value not in CONDITIONS for value in requested_conditions):
        raise ValueError("Conditions must be a non-empty subset of the Tier-1 matrix")
    if len(set(requested_conditions)) != len(requested_conditions):
        raise ValueError("Conditions must not contain duplicates")
    if not requested_seeds or any(value not in AGENT_SEEDS for value in requested_seeds):
        raise ValueError("Agent replicate IDs must be a non-empty subset of the packet")
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Agent replicate IDs must not contain duplicates")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")

    requests: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        for condition in requested_conditions:
            for seed in requested_seeds:
                requests.append(
                    _request_spec(
                        episode,
                        condition=condition,
                        agent_seed=seed,
                        packet_manifest_hash=manifest["manifest_hash"],
                        packet_episode_file_sha256=manifest["episode_file_sha256"],
                        base_url_origin=base_url_origin,
                        base_url_endpoint_sha256=base_url_endpoint_sha256,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
    if max_requests is not None:
        requests = requests[:max_requests]
    request_ids = [row["request_id"] for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Tier-1 request IDs are not unique")
    prompt_pairing: dict[str, set[str]] = {}
    for row in requests:
        key = f"{row['episode_id']}::{row['condition']}"
        prompt_pairing.setdefault(key, set()).add(row["identity"]["user_prompt_sha256"])
    paired_prompt_mismatch = sorted(
        key for key, hashes in prompt_pairing.items() if len(hashes) != 1
    )
    if paired_prompt_mismatch:
        raise ValueError(
            f"Agent replicate prompts are not frozen: {paired_prompt_mismatch[:5]}"
        )
    prompt_gold_leaks: list[str] = []
    for row in requests:
        prompt = str(row["user_prompt"])
        prompt_payload = json.loads(prompt)
        leaked_keys = sorted(set(_walk_mapping_keys(prompt_payload)) & HIDDEN_AGENT_KEYS)
        leaked_cell = row["condition"] in prompt
        if leaked_keys or leaked_cell:
            prompt_gold_leaks.append(
                f"{row['request_id']}:keys={leaked_keys}:cell={leaked_cell}"
            )
    if prompt_gold_leaks:
        raise ValueError(f"Gold leaked into Agent Prompt: {prompt_gold_leaks[:5]}")

    # Use the bound packet count instead of re-deriving it from profile internals.
    full_request_count = int(manifest["validation"]["planned_agent_run_count"])
    plan: dict[str, Any] = {
        "schema": REQUEST_PLAN_SCHEMA,
        "created_at": str(created_at),
        "packet_root_name": packet_root.name,
        "packet_manifest_hash": manifest["manifest_hash"],
        "packet_manifest_file_sha256": _sha256_file(packet_root / "manifest.json"),
        "packet_episode_file_sha256": manifest["episode_file_sha256"],
        "packet_verification_hash": verification["verification_hash"],
        "model": model,
        "base_url_origin": base_url_origin,
        "base_url_endpoint_sha256": base_url_endpoint_sha256,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": "json_object",
        "network_attempts_per_invocation": NETWORK_ATTEMPTS_PER_INVOCATION,
        "max_total_network_attempts": MAX_TOTAL_NETWORK_ATTEMPTS,
        "provider_seed_parameter_sent": False,
        "agent_seed_semantics": "host_paired_replicate_id_not_provider_rng_seed",
        "agent_replicate_id_exposed_to_agent": False,
        "conditions": list(requested_conditions),
        "agent_replicate_ids": list(requested_seeds),
        "episode_filter": sorted(selected_ids),
        "request_count": len(requests),
        "full_matrix_request_count": full_request_count,
        "is_full_matrix": len(requests) == full_request_count
        and set(requested_conditions) == set(CONDITIONS)
        and set(requested_seeds) == set(AGENT_SEEDS)
        and not selected_ids,
        "paired_prompt_mismatch_count": 0,
        "prompt_gold_leak_count": 0,
        "system_prompt_sha256": _sha256_text(SYSTEM_PROMPT),
        "requests": requests,
        "request_plan_hash": "",
    }
    plan["request_plan_hash"] = sha256_json(
        {key: value for key, value in plan.items() if key != "request_plan_hash"}
    )
    return plan


def parse_agent_response(
    raw_response: Any,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise AgentResponseValidationError("response_is_not_valid_json") from error
    else:
        payload = raw_response
    if not isinstance(payload, Mapping):
        raise AgentResponseValidationError("response_is_not_an_object")
    if set(payload) != RESPONSE_KEYS:
        missing = sorted(RESPONSE_KEYS - set(payload))
        extra = sorted(set(payload) - RESPONSE_KEYS)
        raise AgentResponseValidationError(
            f"response_keys:missing={missing}:extra={extra}"
        )
    selected = str(payload.get("selected_action_id") or "")
    candidates = request.get("candidate_action_map") or {}
    if selected not in candidates:
        raise AgentResponseValidationError("selected_action_id_not_supplied")
    patch = payload.get("config_patch")
    if not isinstance(patch, Mapping):
        raise AgentResponseValidationError("config_patch_is_not_an_object")
    normalized_patch = dict(patch)
    if normalized_patch != candidates[selected]:
        raise AgentResponseValidationError("config_patch_does_not_match_selected_action")
    refs = payload.get("memory_refs_used")
    if not isinstance(refs, list) or any(not isinstance(value, str) for value in refs):
        raise AgentResponseValidationError("memory_refs_used_is_not_a_string_list")
    if len(refs) != len(set(refs)):
        raise AgentResponseValidationError("memory_refs_used_contains_duplicates")
    visible_refs = set(request.get("visible_memory_ids") or [])
    if not set(refs).issubset(visible_refs):
        raise AgentResponseValidationError("memory_refs_used_contains_unseen_memory")
    influence = str(payload.get("memory_influence") or "")
    if influence not in INFLUENCE_VALUES:
        raise AgentResponseValidationError("invalid_memory_influence")
    if not visible_refs and (refs or influence != "none"):
        raise AgentResponseValidationError("memory_off_response_claims_memory_use")
    if influence == "none" and refs:
        raise AgentResponseValidationError("none_influence_has_memory_refs")
    if influence in {"considered_not_adopted", "adopted"} and not refs:
        raise AgentResponseValidationError("memory_influence_missing_memory_ref")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise AgentResponseValidationError("rationale_is_empty")
    if len(rationale) > 2000:
        raise AgentResponseValidationError("rationale_is_too_long")
    return {
        "schema": RESPONSE_SCHEMA,
        "selected_action_id": selected,
        "config_patch": normalized_patch,
        "memory_refs_used": list(refs),
        "memory_influence": influence,
        "rationale": rationale.strip(),
    }


def _default_network_caller(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
) -> Callable[[Mapping[str, Any]], tuple[Any, dict[str, Any]]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )

    def call(request: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        response = client.chat.completions.create(
            model=request["model"],
            messages=[
                {"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": request["user_prompt"]},
            ],
            temperature=request["temperature"],
            max_tokens=request["max_tokens"],
            response_format={"type": "json_object"},
            stream=False,
        )
        if not response.choices:
            raise RuntimeError("provider_response_has_no_choices")
        choice = response.choices[0]
        finish_reason = str(choice.finish_reason or "")
        if finish_reason != "stop":
            raise RuntimeError(f"provider_finish_reason:{finish_reason}")
        message = choice.message
        raw = message.content or ""
        usage = getattr(response, "usage", None)
        usage_payload = usage.model_dump() if hasattr(usage, "model_dump") else {}
        metadata = {
            "response_id": str(getattr(response, "id", "") or ""),
            "response_model": str(getattr(response, "model", "") or ""),
            "system_fingerprint": str(
                getattr(response, "system_fingerprint", "") or ""
            ),
            "finish_reason": finish_reason,
            "usage": usage_payload,
            "reasoning_content_present": bool(
                getattr(message, "reasoning_content", None)
            ),
            "reasoning_content_stored": False,
        }
        return raw, metadata

    return call


def _load_saved_response(
    path: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    record = _read_json(path)
    if record.get("schema") != RAW_RESPONSE_SCHEMA:
        raise ValueError(f"Saved response schema mismatch: {path}")
    if record.get("request_id") != request["request_id"]:
        raise ValueError(f"Saved response ID mismatch: {path}")
    if record.get("identity_hash") != request["identity_hash"]:
        raise ValueError(f"Saved response identity mismatch: {path}")
    parsed = parse_agent_response(record.get("raw_response"), request)
    if parsed != record.get("parsed_response"):
        raise ValueError(f"Saved response parse binding mismatch: {path}")
    expected_hash = sha256_json(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    if record.get("record_hash") != expected_hash:
        raise ValueError(f"Saved response hash mismatch: {path}")
    return record


def _load_attempts(path: Path, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    record = _read_json(path)
    if record.get("schema") != ATTEMPT_LOG_SCHEMA:
        raise ValueError(f"Attempt log schema mismatch: {path}")
    if record.get("request_id") != request["request_id"]:
        raise ValueError(f"Attempt log request mismatch: {path}")
    if record.get("identity_hash") != request["identity_hash"]:
        raise ValueError(f"Attempt log identity mismatch: {path}")
    attempts = record.get("attempts") or []
    if not isinstance(attempts, list):
        raise ValueError(f"Attempt log attempts are invalid: {path}")
    return [dict(value) for value in attempts]


def _write_attempts(
    path: Path,
    request: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    completed: bool,
) -> None:
    write_json_atomic(
        path,
        {
            "schema": ATTEMPT_LOG_SCHEMA,
            "request_id": request["request_id"],
            "identity_hash": request["identity_hash"],
            "attempts": [dict(value) for value in attempts],
            "completed": completed,
        },
    )


def execute_request(
    request: Mapping[str, Any],
    run_root: str | Path,
    *,
    allow_network: bool,
    network_caller: Callable[[Mapping[str, Any]], tuple[Any, dict[str, Any]]] | None,
    api_key_for_redaction: str = "",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    raw_path = root / "raw_responses" / f"{request['request_id']}.json"
    attempt_path = root / "attempt_logs" / f"{request['request_id']}.json"
    if raw_path.exists():
        record = _load_saved_response(raw_path, request)
        return {
            "request_id": request["request_id"],
            "source": "saved",
            "record": record,
            "error": "",
        }
    if not allow_network:
        return {
            "request_id": request["request_id"],
            "source": "cache_miss",
            "record": None,
            "error": "network_disabled_and_response_not_cached",
        }
    if network_caller is None:
        raise ValueError("A network caller is required when network access is enabled")
    attempts = _load_attempts(attempt_path, request)
    remaining_total = MAX_TOTAL_NETWORK_ATTEMPTS - len(attempts)
    if remaining_total <= 0:
        return {
            "request_id": request["request_id"],
            "source": "attempts_exhausted",
            "record": None,
            "error": "network_attempt_budget_exhausted",
        }
    invocation_budget = min(NETWORK_ATTEMPTS_PER_INVOCATION, remaining_total)
    for invocation_offset in range(invocation_budget):
        attempt_number = len(attempts) + 1
        started = time.perf_counter()
        try:
            raw_response, provider_metadata = network_caller(request)
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "request_status": "error",
                    "parse_status": "not_attempted",
                    "error": _redact_error(error, api_key=api_key_for_redaction),
                    "latency_seconds": time.perf_counter() - started,
                    "recorded_at": _utc_now(),
                }
            )
            _write_attempts(attempt_path, request, attempts, completed=False)
            if invocation_offset + 1 < invocation_budget:
                sleep_fn(min(2 ** (attempt_number - 1), 8))
            continue
        try:
            parsed = parse_agent_response(raw_response, request)
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "request_status": "ok",
                    "parse_status": "error",
                    "parse_error": _redact_error(error),
                    "raw_response": raw_response,
                    "provider_metadata": dict(provider_metadata),
                    "latency_seconds": time.perf_counter() - started,
                    "recorded_at": _utc_now(),
                }
            )
            _write_attempts(attempt_path, request, attempts, completed=False)
            if invocation_offset + 1 < invocation_budget:
                sleep_fn(min(2 ** (attempt_number - 1), 8))
            continue
        attempts.append(
            {
                "attempt": attempt_number,
                "request_status": "ok",
                "parse_status": "ok",
                "provider_metadata": dict(provider_metadata),
                "latency_seconds": time.perf_counter() - started,
                "recorded_at": _utc_now(),
            }
        )
        _write_attempts(attempt_path, request, attempts, completed=True)
        record: dict[str, Any] = {
            "schema": RAW_RESPONSE_SCHEMA,
            "request_id": request["request_id"],
            "identity_hash": request["identity_hash"],
            "episode_id": request["episode_id"],
            "stage": request["stage"],
            "domain": request["domain"],
            "condition": request["condition"],
            "agent_replicate_id": request["agent_replicate_id"],
            "model": request["model"],
            "temperature": request["temperature"],
            "provider_seed_parameter_sent": False,
            "raw_response": raw_response,
            "raw_response_sha256": _sha256_text(
                raw_response
                if isinstance(raw_response, str)
                else json.dumps(raw_response, sort_keys=True, ensure_ascii=False)
            ),
            "parsed_response": parsed,
            "provider_metadata": dict(provider_metadata),
            "attempt_count": len(attempts),
            "retry_count": max(0, len(attempts) - 1),
            "record_hash": "",
        }
        record["record_hash"] = sha256_json(
            {key: value for key, value in record.items() if key != "record_hash"}
        )
        _write_json_exclusive(raw_path, record)
        return {
            "request_id": request["request_id"],
            "source": "network",
            "record": record,
            "error": "",
        }
    return {
        "request_id": request["request_id"],
        "source": "failed",
        "record": None,
        "error": "network_or_parse_attempts_exhausted",
    }


def _prepare_run_root(run_root: Path, plan: Mapping[str, Any]) -> None:
    if (run_root / "run_report.json").exists() or (run_root / "responses.jsonl").exists():
        raise FileExistsError(f"Tier-1 generation root is finalized: {run_root}")
    allowed_root_entries = {
        "attempt_logs",
        "progress.json",
        "raw_responses",
        "request_plan.json",
    }
    if run_root.exists():
        unexpected_root_entries = {
            path.name for path in run_root.iterdir() if path.name not in allowed_root_entries
        }
        if unexpected_root_entries:
            raise FileExistsError(
                "Tier-1 generation root contains unexpected assets: "
                f"{sorted(unexpected_root_entries)}"
            )
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "raw_responses").mkdir(exist_ok=True)
    (run_root / "attempt_logs").mkdir(exist_ok=True)
    plan_path = run_root / "request_plan.json"
    if plan_path.exists():
        saved = _read_json(plan_path)
        if saved != plan:
            raise ValueError(f"Tier-1 request plan identity mismatch: {plan_path}")
    else:
        _write_json_exclusive(plan_path, plan)
    expected_names = {
        f"{request['request_id']}.json" for request in plan.get("requests") or []
    }
    for directory_name in ("raw_responses", "attempt_logs"):
        directory = run_root / directory_name
        unexpected = {
            path.name for path in directory.glob("*.json") if path.name not in expected_names
        }
        if unexpected:
            raise FileExistsError(
                f"Tier-1 {directory_name} contains responses outside the frozen plan: "
                f"{sorted(unexpected)[:5]}"
            )


def run_generation(
    plan: Mapping[str, Any],
    run_root: str | Path,
    *,
    allow_network: bool,
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
    timeout_seconds: float = 120.0,
    workers: int = 4,
    network_caller: Callable[[Mapping[str, Any]], tuple[Any, dict[str, Any]]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if workers <= 0:
        raise ValueError("workers must be positive")
    if _base_url_endpoint_hash(base_url) != plan.get("base_url_endpoint_sha256"):
        raise ValueError("DeepSeek base URL does not match the frozen request plan")
    _prepare_run_root(root, plan)
    if allow_network and network_caller is None:
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required with --allow-network")
        network_caller = _default_network_caller(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    requests = [dict(value) for value in plan.get("requests") or []]
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if workers == 1:
        for request in requests:
            results.append(
                execute_request(
                    request,
                    root,
                    allow_network=allow_network,
                    network_caller=network_caller,
                    api_key_for_redaction=api_key,
                    sleep_fn=sleep_fn,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    execute_request,
                    request,
                    root,
                    allow_network=allow_network,
                    network_caller=network_caller,
                    api_key_for_redaction=api_key,
                    sleep_fn=sleep_fn,
                ): request["request_id"]
                for request in requests
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        {
                            "request_id": futures[future],
                            "source": "runner_exception",
                            "record": None,
                            "error": _redact_error(error, api_key=api_key),
                        }
                    )
    results.sort(key=lambda row: row["request_id"])
    failures = [row for row in results if row.get("record") is None]
    progress = {
        "schema": "decision_admissibility_tier1_generation_progress_v1",
        "request_plan_hash": plan["request_plan_hash"],
        "request_count": len(requests),
        "completed_count": len(results) - len(failures),
        "failure_count": len(failures),
        "source_counts": dict(
            sorted(
                {
                    source: sum(row.get("source") == source for row in results)
                    for source in {str(row.get("source") or "") for row in results}
                }.items()
            )
        ),
        "failed_request_ids": [row["request_id"] for row in failures],
        "errors": [row.get("error") or "" for row in failures],
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at": _utc_now(),
    }
    write_json_atomic(root / "progress.json", progress)
    if failures:
        raise RuntimeError(
            f"Tier-1 generation incomplete: {len(failures)} of {len(requests)} requests failed; "
            f"resume with the same request plan at {root}"
        )

    response_rows = [
        {
            "request_id": row["request_id"],
            "source": row["source"],
            **dict(row["record"]),
        }
        for row in results
    ]
    response_path = root / "responses.jsonl"
    _write_jsonl_exclusive(response_path, response_rows)
    raw_hashes = {
        path.name: _sha256_file(path)
        for path in sorted((root / "raw_responses").glob("*.json"))
    }
    report: dict[str, Any] = {
        "schema": RUN_REPORT_SCHEMA,
        "completed_at": _utc_now(),
        "request_plan_hash": plan["request_plan_hash"],
        "packet_manifest_hash": plan["packet_manifest_hash"],
        "model": plan["model"],
        "base_url_origin": plan["base_url_origin"],
        "base_url_endpoint_sha256": plan["base_url_endpoint_sha256"],
        "temperature": plan["temperature"],
        "max_tokens": plan["max_tokens"],
        "network_attempts_per_invocation": plan[
            "network_attempts_per_invocation"
        ],
        "max_total_network_attempts": plan["max_total_network_attempts"],
        "provider_seed_parameter_sent": False,
        "agent_seed_semantics": plan["agent_seed_semantics"],
        "request_count": len(requests),
        "response_count": len(response_rows),
        "network_response_count": sum(row["source"] == "network" for row in results),
        "saved_response_count": sum(row["source"] == "saved" for row in results),
        "error_count": 0,
        "responses_file": response_path.name,
        "responses_file_sha256": _sha256_file(response_path),
        "raw_response_file_hashes": raw_hashes,
        "raw_response_file_count": len(raw_hashes),
        "python_version": platform.python_version(),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "run_hash": "",
    }
    report["run_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "run_hash"}
    )
    _write_json_exclusive(root / "run_report.json", report)
    return report


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.env_file is not None:
        load_deepseek_env_file(pre_args.env_file)
    parser = argparse.ArgumentParser(
        description="Run the held-out WP8 Tier-1 controlled decisions via DeepSeek."
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", ""))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--agent-seed", action="append", type=int)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--legacy-episode-root",
        type=Path,
        default=DEFAULT_LEGACY_EPISODE_ROOT,
    )
    args = parser.parse_args()
    if not args.model:
        raise RuntimeError("--model or DEEPSEEK_MODEL is required")
    plan = build_request_plan(
        args.packet_root,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        created_at=args.created_at,
        conditions=args.condition or CONDITIONS,
        agent_seeds=args.agent_seed or AGENT_SEEDS,
        episode_ids=args.episode_id,
        max_requests=args.max_requests,
        legacy_episode_root=args.legacy_episode_root,
    )
    report = run_generation(
        plan,
        args.run_root,
        allow_network=args.allow_network,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        workers=args.workers,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
