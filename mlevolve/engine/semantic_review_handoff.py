"""Host-owned, process-local handoff of a completed semantic review.

This is not an admission judge or a persistent/candidate-writable cache. Only
the reviewer can insert a complete clean result. A new process starts empty.
"""
from copy import deepcopy
import hashlib
import json
import os
from urllib.parse import urlsplit


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def input_identity(agent, node_id, prompt, spec, model, temperature, max_tokens,
                   contract_context=None):
    """Bind the complete semantic dossier; never inspect credential fields.

    Repair permissions/count affect whether a patch may be emitted, not whether
    unchanged code is clean. Normalize just those two action controls. All rules,
    source, static findings, ABI, task and Planner-plan content remain covered.
    """
    dossier = deepcopy(prompt)
    dossier["Review mode"] = "judge-only: revised_code must be empty; return revise plus concrete findings"
    dossier["Repair attempt already used"] = "0"
    config = getattr(agent.cfg, "agent", None)
    routes = {}
    for name in ("code", "feedback"):
        stage = getattr(config, name, None)
        endpoint = str(getattr(stage, "base_url", "") or "")
        parsed = urlsplit(endpoint)
        # A credential-bearing URL is not safe to record/hash. Disable reuse.
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return None
        routes[name] = {"model": str(getattr(stage, "model", "") or ""),
                        "base_url": endpoint}
    return digest({"schema": "semantic-review-input-v1", "candidate_id": str(node_id),
                   "prompt": dossier, "function": {"name": spec.name,
                   "description": spec.description, "json_schema": spec.json_schema},
                   "model": model, "temperature": temperature, "max_tokens": max_tokens,
                   "reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", ""),
                   "routes": routes, "candidate_contract_context": contract_context})


def complete_audit(record, hard_rules):
    audited = record.get("audited_rule_ids") or []
    blocked = record.get("blocking_rule_ids") or []
    status = record.get("status")
    return (record.get("audit_complete") is True
            and len(audited) == len(set(audited)) == len(hard_rules)
            and set(audited) == set(hard_rules)
            and len(blocked) == len(set(blocked))
            and set(blocked) <= set(hard_rules)
            and status in {"clean", "revise", "uncertain"}
            and bool(blocked) == (status == "revise")
            and (status != "clean" or not record.get("findings")))


def _complete_clean(report, source_sha, hard_rules):
    records = report.get("attempts") or []
    if not records:
        return False
    final = records[-1]
    return (report.get("enabled") is True and report.get("reviewer") == "llm_agent"
            and report.get("final_status") == final.get("status") == "clean"
            and report.get("final_code_sha256") == final.get("code_sha256") == source_sha
            and complete_audit(final, hard_rules)
            and not final.get("host_sdk_static_observations")
            and not any(final.get(k) for k in ("consistency_error", "patch_supplied",
                       "patch_applied", "patch_ignored_in_observe_only_mode")))


def remember(agent, node_id, identity, report, hard_rules):
    if not identity or not _complete_clean(report, report.get("final_code_sha256"), hard_rules):
        return
    cache = getattr(agent, "_host_semantic_review_handoff", None)
    if cache is None:
        cache = {}
        agent._host_semantic_review_handoff = cache
    saved = deepcopy(report)
    cache[str(node_id)] = {"input_sha256": identity, "report": saved,
                           "receipt_sha256": digest(saved)}


def reuse(agent, node_id, identity, source_sha, hard_rules, allow_repair):
    cached = getattr(agent, "_host_semantic_review_handoff", {}).get(str(node_id))
    if not identity or not cached or cached.get("input_sha256") != identity:
        return None
    previous = cached.get("report") or {}
    if (cached.get("receipt_sha256") != digest(previous)
            or not _complete_clean(previous, source_sha, hard_rules)):
        return None
    result = deepcopy(previous)
    result["handoff"] = {"schema": "semantic-review-handoff-v1", "reused": True,
                         "input_sha256": identity,
                         "origin_receipt_sha256": cached["receipt_sha256"],
                         "origin": "same_agent_process_completed_review",
                         "new_semantic_llm_calls": 0,
                         "requested_allow_repair": bool(allow_repair)}
    return result
