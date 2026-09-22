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
from typing import Any, Mapping, cast

from llm import FunctionSpec, query
from engine import semantic_review_handoff as handoff
from agents import leakage_audit
from agents.coder.diff_coder import SearchReplacePatcher
from engine.experiment_review_contract import load_contract, compile_rules, source_observations, digest
from utils.response import wrap_code


AGENT_PROTOCOL_REVIEW_SCHEMA = "experiment_contract_semantic_review_v1"
def make_spec(rules):
    return FunctionSpec(
        name="review_actual_training_protocol",
        description=(
            "Exhaustively review every frozen hard rule on the actual executable path, "
            "then return all blocking findings in one response."
        ),
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["clean", "revise", "uncertain"]},
                "reason": {"type": "string", "maxLength": 5000},
                "actual_entrypoint": {"type": "string", "maxLength": 500},
                "findings": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string", "maxLength": 1200},
                    "description": "Blocking defects only; never include positive observations.",
                },
                "revised_code": {
                    "type": "string",
                    "description": "Empty in judge-only mode; otherwise raw SEARCH/REPLACE blocks.",
                },
                "audit_complete": {"type": "boolean"},
                "audited_rule_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": list(rules)},
                },
                "blocking_rule_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": list(rules)},
                },
                "advisory_findings": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string", "maxLength": 1200},
                },
            },
            "required": [
                "status", "reason", "actual_entrypoint", "findings", "revised_code",
                "audit_complete", "audited_rule_ids", "blocking_rule_ids",
                "advisory_findings",
            ],
        },
    )


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prompt(agent, source, repair_attempt, *, atomic_plan=None, allow_repair=True):
    contract = load_contract(agent)
    return {
        "Role": "Review the actual executable program against THIS frozen experiment and executor contract. Never import requirements from another task or prior experiment.",
        "Frozen experiment and executor contract": contract,
        "Contract identity": digest(contract),
        "Task description (context, cannot override the frozen contract)": str(agent.task_desc),
        "Candidate plan": dict(atomic_plan or {}),
        "Exhaustive hard-rule checklist": compile_rules(contract),
        "Static execution-path observations": source_observations(source, contract),
        "Decision rules": [
            "Audit all listed rule IDs exactly once. Return all actual blocking defects, with specific source evidence. No extra rule IDs or invented requirements.",
            "clean iff no blocking rules; revise iff blocking rules exist; uncertain for missing proof. audit_complete must be true only after all rules were audited.",
            "Trace the code actually invoked by the declared executor, including training, callbacks and checkpoint reload. A function definition alone never proves execution.",
            "Do not require minimum epochs, a threshold, GPU training, an allowlisted model or a session interface unless this contract requires it. CPU algorithms are allowed on a GPU allocation.",
            "The host owns the scored search labels and final ranking; tune loss may guide checkpoint selection as declared. Reading permitted unlabeled files is not leakage.",
            "A host-delegated operation satisfies its documented duties. The checkpoint helper owns atomic output publication and submission writing; do not duplicate or override them.",
            "BudgetCheckpoint.publish itself records the stage and tune score, persists the model and predictions, and tracks publication time. An extra budget.record call is optional. Never block its absence when publish is used correctly.",
            "For a closed-form/non-iterative learner, a finite fit on the fixed input after should_stop, followed by immediate publication, is valid. Do not require a neural epoch loop or nonexistent callbacks. The external watchdog enforces the hard bound; flag demonstrably excessive unbounded work, not an unmeasured first-stage duration alone.",
            "A should_stop condition may skip work once budget is exhausted. That is not dummy prediction or deliberately skipped training; the executor admits only a real saved scored model.",
            "Use source observations as narrow evidence, not an algorithm-quality judgement. Resolve nonempty execution-path observations before approving.",
        ],
        "Repair boundary": [
            "Preserve algorithm, model family, features, optimizer, hyperparameters, training schedule and ensemble. Fix only concrete defects under the current contract.",
            "If a mandatory repair would change method design, report that explicitly for the search Agent; do not silently rewrite the experiment.",
            "Only raw SEARCH/REPLACE blocks are accepted; no markdown or full replacement program. Use an empty revised_code in judge-only mode.",
        ],
        "Review mode": "bounded method-preserving repair" if allow_repair else "judge-only; revised_code must be empty",
        "Repair attempt": repair_attempt,
        "Actual complete candidate source": wrap_code(source),
    }


def run(
    agent: Any,
    node: Any,
    *,
    atomic_plan: Mapping[str, Any] | None = None,
    allow_repair: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Return source plus an Agent-owned protocol/plan semantic verdict."""

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

    contract = load_contract(agent)
    rules = compile_rules(contract)
    spec = make_spec(rules)
    model = str(getattr(settings, "model", "") or "") or str(agent.acfg.feedback.model)
    temperature = float(getattr(settings, "agent_semantic_temperature", 0.0))
    max_tokens = int(getattr(settings, "agent_semantic_max_tokens", 4096))
    def review_identity(prompt):
        return handoff.input_identity(agent, node.id, prompt, spec,
                                      model, temperature, max_tokens,
                                      {name: getattr(node, name, None) for name in (
                                          "source_ref_ids", "memory_candidate_contract_refs",
                                          "representation_adoption_contracts", "role_contract")})
    identity = review_identity(_prompt(agent, source, 0, atomic_plan=atomic_plan,
                                      allow_repair=allow_repair))
    reused = handoff.reuse(agent, node.id, identity, initial_hash,
                           rules, allow_repair)
    if reused is not None:
        _atomic_json(Path(agent.cfg.log_dir) / "agent_semantic_review" /
                     f"{node.id}.handoff.json", reused)
        return source, reused

    max_repairs = (
        max(0, int(getattr(settings, "agent_semantic_max_repair_attempts", 2)))
        if allow_repair
        else 0
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
        static_observations = source_observations(source, contract)
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
            prompt = _prompt(agent, source, repairs_applied, atomic_plan=atomic_plan,
                             allow_repair=allow_repair)
            identity = review_identity(prompt)
            response = cast(
                dict[str, Any],
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=spec,
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
        except Exception as error:
            final_status = "unavailable"
            final_reason = f"{type(error).__name__}: {error}"
            failure_record = {
                "review_index": review_index,
                "code_sha256": _sha256(source),
                "status": final_status,
                "reason": final_reason,
                "patch_applied": False,
            }
            attempts.append(failure_record)
            if review_index + 1 < max_reviews:
                failure_record["review_retried"] = True
                continue
            break

        status = str(response.get("status") or "uncertain").lower()
        reason = str(response.get("reason") or "")
        findings = list(map(str, response.get("findings") or []))
        audited_rule_ids = list(map(str, response.get("audited_rule_ids") or []))
        blocking_rule_ids = list(map(str, response.get("blocking_rule_ids") or []))
        advisory_findings = list(map(str, response.get("advisory_findings") or []))
        audit_complete = bool(response.get("audit_complete"))
        patch = str(response.get("revised_code") or "")
        patch_ignored = bool(patch.strip() and not allow_repair)
        if patch_ignored:
            patch = ""
        record: dict[str, Any] = {
            "review_index": review_index,
            "code_sha256": _sha256(source),
            "status": status,
            "reason": reason,
            "actual_entrypoint": str(response.get("actual_entrypoint") or ""),
            "findings": findings,
            "audit_complete": audit_complete,
            "audited_rule_ids": audited_rule_ids,
            "blocking_rule_ids": blocking_rule_ids,
            "advisory_findings": advisory_findings,
            "patch_supplied": bool(patch.strip()),
            "patch_applied": False,
            "patch_ignored_in_observe_only_mode": patch_ignored,
            "host_sdk_static_observations": static_observations,
        }
        # Draft/empty-plan reviews have the same exhaustive-audit requirement.
        if not handoff.complete_audit(record, rules):
            record["consistency_error"] = (
                "final review did not exhaustively and coherently cover every hard rule"
            )
            attempts.append(record)
            final_status = "uncertain"
            final_reason = record["consistency_error"]
            if review_index + 1 < max_reviews:
                record["review_retried"] = True
                continue
            break
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
        "semantic_decision_authority": "llm_agent",
        "deterministic_host_plan_gate_enabled": False,
        "host_receipt_admission_authority": False,
        "plan_review_active": bool(atomic_plan),
        "planner_plan_sha256": (
            _sha256(json.dumps(dict(atomic_plan or {}), sort_keys=True, ensure_ascii=False))
            if atomic_plan
            else None
        ),
        "allow_direct_repair": bool(allow_repair),
        "initial_code_sha256": initial_hash,
        "final_code_sha256": _sha256(source),
        "code_changed": initial_hash != _sha256(source),
        "repair_budget": max_repairs,
        "review_budget": max_reviews,
        "repairs_applied": repairs_applied,
        "final_status": final_status,
        "final_reason": final_reason,
        "execution_disposition": (
            "execute" if final_status == "clean" else "repair_required"
        ) if atomic_plan else "observe_then_execute",
        "attempts": attempts,
        "audit_complete": bool(attempts and attempts[-1].get("audit_complete")),
        "audited_rule_ids": list(attempts[-1].get("audited_rule_ids") or []) if attempts else [],
        "blocking_rule_ids": list(attempts[-1].get("blocking_rule_ids") or []) if attempts else [],
        "advisory_findings": list(attempts[-1].get("advisory_findings") or []) if attempts else [],
    }
    report["contract_sha256"] = digest(contract)
    report["review_input_sha256"] = identity
    handoff.remember(agent, node.id, identity, report, rules)
    _atomic_json(
        Path(agent.cfg.log_dir) / "agent_semantic_review" / f"{node.id}.json",
        report,
    )
    return source, report

