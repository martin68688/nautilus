"""Dedicated code agent for one stage of a protocol-repair transaction."""

from __future__ import annotations

import copy
import logging

from agents.adoption import log_adoption
from agents.coder import plan_and_code_query
from agents.leakage_audit import format_audit, format_repair_preservation_contract
from agents.planner import build_chat_prompt_for_model
from agents.protocol_repair import (
    begin_stage_generation,
    current_stage,
    finish_stage_generation,
    stage_instructions,
)
from agents.triggers import register_node
from engine.search_node import SearchNode
from utils.response import wrap_code

logger = logging.getLogger("MLEvolve")


def run(agent, parent_node: SearchNode) -> SearchNode:
    transaction = begin_stage_generation(parent_node.protocol_repair)
    parent_node.protocol_repair = copy.deepcopy(transaction)
    stage = current_stage(transaction)
    if not stage:
        raise ValueError("Protocol repair has no pending stage")

    plan = transaction.get("protocol_plan", {})
    contract = transaction.get("preservation_contract", {})
    instructions = stage_instructions(transaction)
    introduction = (
        "You are a protocol-repair engineer. The model direction is frozen and useful, but its "
        "evaluation protocol is not trustworthy. Implement exactly one cumulative protocol stage. "
        "Do not optimize, simplify, replace, or creatively redesign the solution."
    )
    user_prompt = (
        f"# Task\n{agent.task_desc}\n\n"
        f"# Protocol plan\n{plan}\n\n"
        f"# Current stage\n{stage}\n\n"
        f"# Frozen preservation contract\n{format_repair_preservation_contract(contract)}\n\n"
        f"# Audit evidence\n{format_audit(parent_node.leakage_audit)}\n\n"
        "# Mandatory stage instructions\n- " + "\n- ".join(instructions) + "\n\n"
        "# Previous complete program\n" + wrap_code(parent_node.code) + "\n\n"
        "Return a concise repair description followed by one complete Python code block."
    )
    prompt = build_chat_prompt_for_model(
        agent.acfg.code.model,
        introduction,
        user_prompt,
        f"I will implement only the `{stage}` protocol stage and preserve the full model design.",
    )

    parent_node.add_expected_child_count()
    config = getattr(agent.acfg, "protocol_repair", None)
    repair_plan, code = plan_and_code_query(
        agent,
        prompt,
        retries=1,
        generation_retries=int(getattr(config, "stage_generation_backend_retries", 2)),
        request_timeout=float(getattr(config, "stage_generation_timeout_seconds", 300)),
    )
    if not repair_plan or not code:
        raise RuntimeError(f"Protocol repair code generation returned no usable program for {stage}")
    transaction = finish_stage_generation(transaction)
    child = SearchNode(
        plan=f"[staged_protocol_repair:{stage}] {repair_plan}",
        code=code,
        parent=parent_node,
        stage="debug" if parent_node.is_buggy or parent_node.is_valid is False else "improve",
        local_best_node=parent_node.local_best_node,
        draft_role=parent_node.draft_role,
        protocol_repair=transaction,
        # Ordinary code review has no protocol/preservation context and can
        # silently redesign a model.  The dedicated stage + preservation
        # audits are the only reviewers for this repair-only child.
        skip_code_review=True,
    )
    register_node(agent, child, prompt, parent_node=parent_node)
    child.protocol_repair = transaction
    child.leakage_repair_context = {
        "source_node_id": transaction.get("source_node_id"),
        "source_code_sha256": transaction.get("source_code_sha256"),
        "status": "staged_protocol_repair",
        "issues": copy.deepcopy(parent_node.leakage_audit.get("issues") or []),
        "preservation_contract": copy.deepcopy(contract),
        "protocol_transaction_id": transaction.get("transaction_id"),
        "protocol_stage": stage,
    }
    child.audit_repair_required = True
    if child.replay_source:
        child.replay_source["repair_seed_only"] = False
        child.replay_source["repair_parent_node_id"] = parent_node.id
        child.replay_status = "staged_protocol_repair"
    log_adoption(
        child,
        agent,
        "leakage_failure_memory",
        [transaction.get("source_node_id")],
        "protocol_repair",
        adoption_mode=f"staged_protocol_repair:{stage}",
    )
    logger.warning(
        "[protocol-repair] transaction=%s stage=%s parent=%s child=%s",
        transaction.get("transaction_id"), stage, parent_node.id, child.id,
    )
    return child
