"""Dedicated code agent for one stage of a protocol-repair transaction."""

from __future__ import annotations

import copy
import ast
import json
import logging
import re

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
from utils.response import extract_code, wrap_code

logger = logging.getLogger("MLEvolve")


def _protected_constructor_snippets(code: str, contract: dict) -> list[str]:
    """Return exact protected constructor expressions for prompt pinning."""
    protected = set((contract.get("component_calls") or {}).keys())
    if not protected:
        return []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    snippets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name not in protected:
            continue
        snippet = ast.get_source_segment(code, node)
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    return snippets


def _anchor_missing_model_literals(code: str, contract: dict) -> str:
    """Keep frozen identity literals visible when protocol code derives fold paths."""
    tree = ast.parse(code)
    present_literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing = [
        str(value) for value in (contract.get("model_literals") or [])
        if str(value) not in present_literals
    ]
    if not missing:
        return code
    insert_after = 0
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        insert_after = int(body[0].end_lineno or body[0].lineno)
        body = body[1:]
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_after = int(node.end_lineno or node.lineno)
        else:
            break
    lines = code.splitlines()
    anchor = "_MLEVOLVE_PRESERVED_MODEL_LITERALS = " + repr(tuple(missing))
    lines.insert(insert_after, anchor)
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _source_transaction_code(parent_node: SearchNode, transaction: dict) -> str:
    source_id = transaction.get("source_node_id")
    node = parent_node
    while node is not None:
        if node.id == source_id:
            return node.code
        node = node.parent
    return parent_node.code


def _stage_generation_base(parent_node: SearchNode, stage: str) -> SearchNode:
    """Continue a same-stage retry from the latest usable candidate.

    Static-audit failures still contain useful protocol edits.  Starting each
    retry from the last clean stage discarded those edits and forced the model
    to rediscover them.  Generation failures do not create a child, so the
    supplied parent remains the latest usable program in both cases.
    """
    return parent_node


def _same_stage_rejections(transaction: dict, stage: str) -> list[dict]:
    feedback_stages = {stage}
    if stage == "final_holdout":
        feedback_stages.add("final_holdout_runtime")
    feedback = []
    seen = set()
    for entry in reversed(list(transaction.get("history") or [])):
        if entry.get("stage") not in feedback_stages or entry.get("status") not in {
            "failed", "generation_failed"
        }:
            continue
        for item in entry.get("feedback") or []:
            key = (
                item.get("issue_code"), item.get("evidence"), item.get("remediation")
            )
            if key in seen:
                continue
            seen.add(key)
            feedback.append(item)
    return feedback


def _actionable_rejection_contract(transaction: dict, stage: str) -> str:
    """Return a machine-readable checklist for the next generation attempt."""
    rejections = _same_stage_rejections(transaction, stage)[:32]
    stage_calls = {
        "data_scope": [
            'protocol_guard.register_partition("outer_train", outer_train_ids)',
            'protocol_guard.register_partition("outer_holdout", outer_holdout_ids)',
        ],
        "cross_fit": [
            'protocol_guard.record_fit(component, inner_train_ids, purpose="cross_fit")',
            'protocol_guard.record_prediction(component, inner_train_ids, inner_valid_ids, purpose="oof")',
            "protocol_guard.record_global_oof(oof_predictions, outer_train_ids)",
        ],
        "selection_freeze": [
            'protocol_guard.record_selection("protocol_state", outer_train_ids)',
            "protocol_guard.freeze()",
        ],
        "final_holdout": [
            'protocol_guard.record_prediction("final_predictor", outer_train_ids, outer_holdout_ids, purpose="final")',
            "protocol_guard.record_final_evaluation(outer_holdout_ids)",
            "protocol_guard.assert_clean()",
            "protocol_guard.emit()",
        ],
    }
    contract = {
        "stage": stage,
        "retry_mode": "edit_latest_candidate_in_place",
        "rejections_to_fix": [
            {
                "issue_code": item.get("issue_code") or "PROTOCOL_REJECTION",
                "evidence": item.get("evidence") or "unspecified violation",
                "required_fix": item.get("remediation") or "remove the reported violation",
            }
            for item in rejections
        ],
        "required_runtime_calls": stage_calls.get(stage, []),
        "acceptance_checks": [
            "preserve every previously repaired item present in the supplied program",
            "do not add, remove, or reconfigure protected model components",
            "use only the shipped ProtocolProvenanceGuard API",
        ],
    }
    if stage == "cross_fit":
        missing_fit_labels = sorted({
            match.group(1)
            for item in rejections
            for match in re.finditer(
                r"fold-local preprocessor\s+([A-Za-z_][A-Za-z0-9_]*)\s+lacks record_fit",
                str(item.get("evidence") or ""),
            )
        })
        contract["required_component_fit_calls"] = [
            f'protocol_guard.record_fit("{label}", inner_train_ids, purpose="fold_preprocess")'
            for label in missing_fit_labels
        ]
        contract["acceptance_checks"].extend([
            "assign each outer_train row exactly once through its fold validation indices",
            "record every learned fit and every fold prediction with stable component names",
            "record_global_oof must occur after complete OOF assignment and before any selection",
            "do not read, transform, predict, score, or tune on outer_holdout",
        ])
    return json.dumps(contract, indent=2, sort_keys=True)


def _rejection_feedback(parent_node: SearchNode, transaction: dict, stage: str) -> str:
    """Format all same-stage rejections as a cumulative next-attempt contract."""
    feedback = _same_stage_rejections(transaction, stage)
    if not feedback:
        feedback = [
            {
                "issue_code": item.get("issue_code"),
                "evidence": item.get("evidence"),
                "remediation": item.get("remediation"),
                "line": item.get("line", 0),
            }
            for item in (parent_node.leakage_audit or {}).get("issues", [])
            if isinstance(item, dict)
        ]
    if not feedback:
        return "- No previous rejection for this stage. Implement the mandatory contract directly."
    lines = [
        "Earlier candidates for this same stage were rejected. Fix every item below in this attempt; do not merely rename variables or suppress the guard. All earlier constraints remain mandatory even when a later attempt failed for a different reason:"
    ]
    for item in feedback[:32]:
        location = f" (line {item.get('line')})" if item.get("line") else ""
        lines.append(f"- [{item.get('issue_code') or 'PROTOCOL_REJECTION'}]{location} {item.get('evidence') or 'unspecified violation'}")
        if item.get("remediation"):
            lines.append(f"  Required fix: {item['remediation']}")
    lines.append("Do not regress any previously repaired item. Keep every already-passed protocol stage and every protected model component unchanged.")
    return "\n".join(lines)


def _restore_protected_component_calls(code: str, source_code: str, contract: dict) -> str:
    """Restore frozen constructor ASTs while retaining generated protocol flow."""
    protected = set((contract.get("component_calls") or {}).keys())
    if not protected:
        return code
    try:
        source_tree = ast.parse(source_code)
        generated_tree = ast.parse(code)
    except SyntaxError:
        return code
    source_calls: dict[str, list[ast.Call]] = {}
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Call) and _call_name(node) in protected:
            source_calls.setdefault(_call_name(node), []).append(node)
    for nodes in source_calls.values():
        nodes.sort(key=lambda node: (node.lineno, node.col_offset))

    seen: dict[str, int] = {}

    class RestoreCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            name = _call_name(node)
            if name not in source_calls:
                return node
            index = seen.get(name, 0)
            seen[name] = index + 1
            if index >= len(source_calls[name]):
                return node
            return ast.copy_location(copy.deepcopy(source_calls[name][index]), node)

    restored = RestoreCalls().visit(generated_tree)
    ast.fix_missing_locations(restored)
    return ast.unparse(restored) + "\n"


def _normalize_protocol_guard_calls(code: str) -> str:
    """Rewrite common invented guard aliases to the supported runtime API."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def partition_call(receiver: ast.expr, name: ast.expr, ids: ast.expr, source: ast.AST) -> ast.Expr:
        call = ast.Call(
            func=ast.Attribute(value=copy.deepcopy(receiver), attr="register_partition", ctx=ast.Load()),
            args=[name, copy.deepcopy(ids)],
            keywords=[],
        )
        return ast.copy_location(ast.Expr(value=call), source)

    class NormalizeGuardCalls(ast.NodeTransformer):
        _partition_aliases = {
            "register_outer_train": "outer_train",
            "register_outer_holdout": "outer_holdout",
            "record_outer_train": "outer_train",
            "record_outer_holdout": "outer_holdout",
        }

        def __init__(self):
            self.changed = False

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if _call_name(node) == "ProtocolProvenanceGuard" and (node.args or node.keywords):
                self.changed = True
                node.args = []
                node.keywords = []
                return node
            if not isinstance(node.func, ast.Attribute):
                return node
            if node.func.attr in {
                "register_global_oof",
                "record_global_oof_coverage",
                "register_global_oof_coverage",
            }:
                # These names are unambiguous aliases for the shipped API.
                # Do not insert a missing call: coverage must still be present
                # in the generated program and pass static/runtime audits.
                self.changed = True
                node.func.attr = "record_global_oof"
                return node
            if node.func.attr in {"verify_no_leak", "assert_no_overlap"}:
                self.changed = True
                node.func.attr = "check_no_overlap"
                return node
            if node.func.attr == "register_partition" and len(node.args) == 2:
                # The runtime API is register_partition(name, sample_ids).  LLMs
                # occasionally reverse those arguments even when they use the
                # correct method name, which otherwise fails only after a long
                # final training run.
                if (
                    isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and not (
                        isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    )
                ):
                    self.changed = True
                    node.args = [node.args[1], node.args[0]]
                return node
            if node.func.attr == "register" and len(node.args) == 2:
                self.changed = True
                node.func.attr = "register_partition"
                return node
            if node.func.attr == "register_outer_partition" and node.args:
                purpose = next(
                    (
                        keyword.value.value
                        for keyword in node.keywords
                        if keyword.arg == "purpose"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ),
                    "",
                )
                partition_name = {
                    "train": "outer_train",
                    "training": "outer_train",
                    "holdout": "outer_holdout",
                    "test": "outer_holdout",
                    "eval": "outer_holdout",
                    "evaluation": "outer_holdout",
                }.get(purpose.lower())
                if partition_name:
                    self.changed = True
                    node.func.attr = "register_partition"
                    node.args = [ast.Constant(partition_name), node.args[0]]
                    node.keywords = []
                    return node
            partition_name = self._partition_aliases.get(node.func.attr)
            if partition_name is None or not node.args:
                return node
            self.changed = True
            node.func.attr = "register_partition"
            node.args = [ast.Constant(partition_name), node.args[0]]
            node.keywords = []
            return node

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {
                    "register", "register_outer_split", "record_outer_split", "register_dataset"
                }
                and len(call.args) >= 2
            ):
                self.changed = True
                train_index, holdout_index = (1, 2) if len(call.args) >= 3 else (0, 1)
                return [
                    partition_call(
                        call.func.value,
                        ast.Constant("outer_train"),
                        call.args[train_index],
                        node,
                    ),
                    partition_call(
                        call.func.value,
                        ast.Constant("outer_holdout"),
                        call.args[holdout_index],
                        node,
                    ),
                ]
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "register_inner_split"
                and len(call.args) >= 2
            ):
                fold = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "fold"),
                    ast.Constant(0),
                )
                self.changed = True
                train_name = ast.JoinedStr(values=[
                    ast.Constant("fold_"),
                    ast.FormattedValue(copy.deepcopy(fold), conversion=-1),
                    ast.Constant("_train"),
                ])
                valid_name = ast.JoinedStr(values=[
                    ast.Constant("fold_"),
                    ast.FormattedValue(copy.deepcopy(fold), conversion=-1),
                    ast.Constant("_valid"),
                ])
                return [
                    partition_call(call.func.value, train_name, call.args[0], node),
                    partition_call(call.func.value, valid_name, call.args[1], node),
                ]
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "register_fold"
                or len(call.args) < 3
            ):
                return node
            fold, train_ids, valid_ids = call.args[:3]
            self.changed = True
            train_name = ast.JoinedStr(values=[
                ast.Constant("fold_"), ast.FormattedValue(copy.deepcopy(fold), conversion=-1), ast.Constant("_train")
            ])
            valid_name = ast.JoinedStr(values=[
                ast.Constant("fold_"), ast.FormattedValue(copy.deepcopy(fold), conversion=-1), ast.Constant("_valid")
            ])
            return [
                partition_call(call.func.value, train_name, train_ids, node),
                partition_call(call.func.value, valid_name, valid_ids, node),
            ]

        def visit_ClassDef(self, node):
            if node.name != "ProtocolProvenanceGuard":
                return self.generic_visit(node)
            # Generated local imitations cannot provide trusted runtime
            # provenance. Drop the shadow class and use the shipped guard.
            self.changed = True
            return None

    transformer = NormalizeGuardCalls()
    normalized = transformer.visit(tree)
    if transformer.changed and not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "agents.protocol_repair_runtime"
        and any(alias.name == "ProtocolProvenanceGuard" for alias in node.names)
        for node in normalized.body
    ):
        normalized.body.insert(
            0,
            ast.ImportFrom(
                module="agents.protocol_repair_runtime",
                names=[ast.alias(name="ProtocolProvenanceGuard")],
                level=0,
            ),
        )
    if not transformer.changed:
        return code
    ast.fix_missing_locations(normalized)
    return ast.unparse(normalized) + "\n"


def run(agent, parent_node: SearchNode) -> SearchNode:
    transaction = begin_stage_generation(parent_node.protocol_repair)
    parent_node.protocol_repair = copy.deepcopy(transaction)
    stage = current_stage(transaction)
    if not stage:
        raise ValueError("Protocol repair has no pending stage")

    plan = transaction.get("protocol_plan", {})
    contract = transaction.get("preservation_contract", {})
    generation_base = _stage_generation_base(parent_node, stage)
    source_code = _source_transaction_code(parent_node, transaction)
    protected_snippets = _protected_constructor_snippets(source_code, contract)
    protected_literals = [str(value) for value in (contract.get("model_literals") or [])]
    instructions = stage_instructions(transaction)
    rejection_feedback = _rejection_feedback(parent_node, transaction, stage)
    rejection_contract = _actionable_rejection_contract(transaction, stage)
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
        "# Exact protected constructors\n"
        + ("\n".join(f"- `{snippet}`" for snippet in protected_snippets) or "- none detected")
        + "\nCopy every expression above exactly; do not alter, remove, or recreate its arguments.\n\n"
        "# Exact protected model/checkpoint literals\n"
        + ("\n".join(f"- `{literal}`" for literal in protected_literals) or "- none detected")
        + "\nEvery literal above must still appear verbatim in the complete program. Preserve it as a named constant even if a later stage will consume it.\n\n"
        f"# Audit evidence\n{format_audit(parent_node.leakage_audit)}\n\n"
        f"# Previous rejection feedback (mandatory)\n{rejection_feedback}\n\n"
        f"# Executable retry contract (mandatory JSON)\n{rejection_contract}\n\n"
        "# Mandatory stage instructions\n- " + "\n- ".join(instructions) + "\n\n"
        "# Latest candidate program\n" + wrap_code(generation_base.code) + "\n\n"
        "Edit the latest candidate cumulatively. Keep its correct partial repairs and change only what the rejection contract requires.\n\n"
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
    if not repair_plan and code:
        # base_coder returns ("", raw_response) when the model emits a valid
        # code block without prose before it. The protocol plan is already
        # frozen, so recover this formatting-only case without weakening any
        # leakage or preservation gate.
        recovered_code = extract_code(code)
        if recovered_code:
            code = recovered_code
            repair_plan = f"Implement the frozen `{stage}` protocol stage."
            logger.warning(
                "[protocol-repair] accepted code-only response for stage=%s",
                stage,
            )
    if not repair_plan or not code:
        raise RuntimeError(f"Protocol repair code generation returned no usable program for {stage}")
    code = _restore_protected_component_calls(code, source_code, contract)
    code = _normalize_protocol_guard_calls(code)
    code = _anchor_missing_model_literals(code, contract)
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
