import logging
from typing import Any, List, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.prompts import prompt_resp_fmt, get_impl_guideline_from_agent
from agents.planner import build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.memory.external_skill_memory import fetch_external_skill_memory, external_memory_section_title, external_memory_section_intro

from engine.conditions import cross_role_synthesis_allowed
from agents.triggers import register_node

logger = logging.getLogger("MLEvolve")


def _coverage_fusion_enabled(agent) -> bool:
    """Whether this aggregation is the protected two-role Fusion milestone."""

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    return bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "cross_role_synthesis_after_balance", False)
        and getattr(policy, "cross_role_synthesis_on_coverage", False)
    )


def _collect_branch_representatives(agent) -> List[SearchNode]:
    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    coverage_mode = _coverage_fusion_enabled(agent)
    if coverage_mode:
        from agents.leakage_audit import legacy_rank_eligible
        from authority.adapters.mlevolve.ranking_gate import (
            authorize_selection,
            filter_ranked_nodes,
        )
        from authority.models import DecisionStage
        from engine.role_balance import candidate_matches_protected_role

        all_successful = [
            node
            for nodes in agent.branch_successful_nodes.values()
            for node in nodes
        ]
        maximize = agent.metric_maximize if agent.metric_maximize is not None else True
        representatives: list[SearchNode] = []
        for role in list(getattr(policy, "roles", []) or []):
            candidates = [
                node
                for node in all_successful
                if candidate_matches_protected_role(node, str(role))
                and node.metric is not None
                and node.metric.value is not None
            ]
            candidates = filter_ranked_nodes(
                agent,
                candidates,
                component=(
                    "agents.aggregation_agent."
                    f"_collect_two_role_representative.{role}"
                ),
            )
            candidates.sort(
                key=lambda node: node.metric.value,
                reverse=maximize,
            )
            representative = next(
                (
                    node
                    for node in candidates
                    if authorize_selection(
                        agent,
                        node,
                        legacy_allowed=legacy_rank_eligible(agent, node),
                        component=(
                            "agents.aggregation_agent."
                            f"_collect_two_role_representative.{role}"
                        ),
                        stage=DecisionStage.FUSION,
                    )
                ),
                None,
            )
            if representative is None:
                logger.info(
                    "Protected two-role synthesis has no authorized representative for role=%s",
                    role,
                )
                return []
            representatives.append(representative)
        logger.info(
            "Collected protected two-role representatives: %s",
            [
                {
                    "role": str(role),
                    "node_id": str(node.id),
                    "metric": node.metric.value,
                }
                for role, node in zip(list(getattr(policy, "roles", []) or []), representatives)
            ],
        )
        return representatives

    representatives = []

    for branch_id, successful_nodes in agent.branch_successful_nodes.items():
        if not successful_nodes or len(successful_nodes) == 0:
            logger.debug(f"Branch {branch_id} has no successful nodes, skipping")
            continue

        from authority.adapters.mlevolve.ranking_gate import filter_ranked_nodes
        successful_nodes = filter_ranked_nodes(
            agent,
            list(successful_nodes),
            component="agents.aggregation_agent._collect_branch_representatives",
        )
        if not successful_nodes:
            continue
        maximize = agent.metric_maximize if agent.metric_maximize is not None else True
        branch_best = max(
            successful_nodes,
            key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
                float("-inf") if maximize else float("inf")
            ),
        )

        if not branch_best.metric or branch_best.metric.value is None:
            logger.debug(f"Branch {branch_id} best node has no valid metric, skipping")
            continue

        from agents.leakage_audit import legacy_rank_eligible
        from authority.adapters.mlevolve.ranking_gate import authorize_selection
        from authority.models import DecisionStage
        if authorize_selection(
            agent,
            branch_best,
            legacy_allowed=legacy_rank_eligible(agent, branch_best),
            component="agents.aggregation_agent._collect_branch_representatives",
            stage=DecisionStage.FUSION,
        ):
            representatives.append(branch_best)

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True
    representatives.sort(
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float("-inf") if maximize else float("inf")
        ),
        reverse=maximize,
    )

    logger.info(
        f"Collected {len(representatives)} branch representatives "
        f"from {len(agent.branch_successful_nodes)} successful solutions"
    )
    return representatives


def run(
    agent,
    mode: str = "node",
    parent_node: Optional[SearchNode] = None,
) -> Optional[SearchNode]:

    if parent_node and not agent.is_root(parent_node):
        logger.error(
            f"_aggregation() should only be called from root node! Got parent_node: {parent_node.id}"
        )
        return None

    if not cross_role_synthesis_allowed(
        agent,
        component="agents.aggregation_agent.run",
    ):
        return None

    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        logger.info(
            f"Max fusion drafts ({agent.max_fusion_drafts}) reached, skipping aggregation"
        )
        return None

    branch_representatives = _collect_branch_representatives(agent)
    if len(branch_representatives) < 2:
        logger.info("Not enough successful branches for aggregation")
        return None

    coverage_fusion = _coverage_fusion_enabled(agent)
    if coverage_fusion:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You are provided with the complete executed source programs and results from the best "
            "authorized Replay and independent Novel branches. "
            "Create an independent cross-role Fusion implementation by inspecting both programs and "
            "freely combining, adapting, refactoring, or selectively replacing their compatible components. "
            "The Fusion branch is exploratory: it may outperform or underperform either parent, and it is "
            "not required to preserve Replay performance."
        )
    else:
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You are provided with multiple successful solutions from different independent branches below. "
            "Your task is to synthesize these diverse approaches and create a completely NEW solution "
            "that draws inspiration from their strengths. "
            "This is a fresh start to spark new ideas by combining insights from different successful directions."
        )

    reference_summaries = []
    if mode == "node":
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=coverage_fusion)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            if coverage_fusion:
                role = str(getattr(node, "draft_role", "") or "unknown")
                branch_info = (
                    f"**Fusion Parent {i + 1}: role={role}, branch={branch_id}, "
                    f"node={node.id}, metric={metric_val:.12g}**\n{trajectory}"
                )
            else:
                branch_info = (
                    f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
                )
            reference_summaries.append(branch_info)
    elif mode == "trajectory":
        for i, node in enumerate(branch_representatives):
            trajectory = node.get_root_to_current_trajectory(max_steps=6)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Evolution Path** (Best Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)
    else:
        logger.warning(f"Unknown aggregation mode: {mode}, using node mode as default")
        for i, node in enumerate(branch_representatives):
            trajectory = node.generate_node_trajectory(need_code=False)
            branch_id = node.branch_id if hasattr(node, "branch_id") else i + 1
            metric_val = node.metric.value if node.metric else 0
            branch_info = (
                f"**Branch {branch_id} Best Solution** (Metric: {metric_val:.4f}):\n{trajectory}"
            )
            reference_summaries.append(branch_info)

    reference_experiences = "\n" + "-" * 80 + "\n".join(reference_summaries)

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Branch Experiences": reference_experiences,
        "Instructions": {},
    }

    prompt["Instructions"] |= prompt_resp_fmt()

    if mode == "node" and coverage_fusion:
        prompt["Instructions"] |= {
            "Protected cross-role Fusion guideline": [
                "- The complete executed source code for both parent branches is included above; inspect it directly rather than relying only on prose summaries.",
                "- Build one independent Fusion candidate that concretely uses compatible strengths from both Replay and Novel.",
                "- You may copy, adapt, refactor, combine, or discard incompatible components from either parent; there is no non-degradation requirement and no requirement to preserve Replay performance.",
                "- Do not ignore the parent implementations and substitute an unrelated third method merely to appear novel.",
                "- In the plan, identify the specific code-level elements taken or adapted from each parent and explain their compatibility.",
                "- The Fusion candidate must remain a single self-contained runnable Python script and must satisfy the same validation and official-submission contract.",
                "- Do not suggest to do EDA.",
            ],
        }
    elif mode == "node":
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Node Mode)": [
                "- You are provided with the BEST solutions from different independent branches.",
                "- Analyze what makes each branch's final solution successful - their key techniques and approaches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: how can you synthesize the strengths of different final solutions into an innovative approach?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, combining the best ideas in a novel way.",
                "- Focus on discovering new synergies between successful techniques from different branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "Multi-branch aggregation guideline (Trajectory Mode)": [
                "- You are provided with the EVOLUTION PATHS of different independent branches.",
                "- Analyze how each branch evolved from initial ideas to their best solutions - what worked and what didn't.",
                "- Learn from the successful improvement patterns and evolution strategies across branches.",
                "- This is NOT about improving a current solution - this is about creating a FRESH NEW approach.",
                "- Think creatively: what new directions emerge from understanding these different evolution paths?",
                "- Write a brief natural language description of your NEW synthesized approach.",
                "- The solution should be distinct and innovative, inspired by successful evolution patterns.",
                "- Focus on discovering unexplored directions suggested by the evolution insights from multiple branches.",
                "- The final code should be a single, runnable Python script.",
                "- Do not suggest to do EDA.",
            ],
        }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)

    external_skill_text, external_skill_ref_ids, external_skill_source = fetch_external_skill_memory(
        agent,
        "fusion_draft",
        branch_experiences=reference_experiences,
        draft_role="novel_exploration",
    )
    if external_skill_text:
        prompt["External Skill Memory"] = external_skill_text

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    data_preview = getattr(agent, "data_preview", "") or ""
    if coverage_fusion:
        assistant_prefix = (
            "Let me approach this systematically.\n"
            f"First, I'll examine the dataset:\n{data_preview}\n"
            "I have the complete executed code and results from the Replay and Novel parents. "
            "I'll inspect both implementations and build an independent Fusion candidate by "
            "combining compatible code-level strengths while freely resolving conflicts."
        )
    else:
        assistant_prefix = (
            "Let me approach this systematically.\n"
            f"First, I'll examine the dataset:\n{data_preview}\n"
            "I have access to multiple successful approaches from different independent branches. "
            "I'll synthesize these diverse insights and create a completely new solution "
            "that combines the best ideas in an innovative way."
        )

    external_skill_section = ""
    if prompt.get("External Skill Memory", "").strip():
        section_title = external_memory_section_title(external_skill_source)
        section_intro = external_memory_section_intro(external_skill_source, "this multi-branch synthesis")
        external_skill_section = (
            f"\n# {section_title}\n"
            f"{section_intro}\n"
            f"{prompt['External Skill Memory']}\n"
        )

    user_prompt = (
        f"\n# Task description\n{prompt['Task description']}\n\n"
        f"{external_skill_section}\n\n"
        f"# Branch Experiences\n{prompt['Branch Experiences']}\n\n{instructions}"
    )
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    plan, code = plan_and_code_query(agent, prompt_complete)

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    explicit_cross_role_provenance = bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "cross_role_synthesis_after_balance", False)
    )
    source_node_ids = [str(node.id) for node in branch_representatives]
    source_roles = sorted(
        {
            str(getattr(node, "draft_role", "") or "")
            for node in branch_representatives
            if getattr(node, "draft_role", None)
        }
    )
    role_contract = {
        "role": "novel_exploration",
        "requirement": "Explore a distinct memory-informed direction.",
    }
    if explicit_cross_role_provenance:
        coverage_trigger = bool(
            getattr(policy, "cross_role_synthesis_on_coverage", False)
        )
        role_contract.update(
            {
                "behavioral_role": "cross_role_synthesis",
                "source_node_ids": source_node_ids,
                "source_draft_roles": source_roles,
                "coverage_gate": "all_configured_roles_minimum_valid_met",
                "synthesis_trigger": (
                    "two_role_coverage_milestone_v1"
                    if coverage_trigger
                    else "balanced_cross_role_synthesis"
                ),
                "protected_first_execution": coverage_trigger,
            }
        )
    aggregation_node = SearchNode(
        plan=plan,
        code=code,
        parent=agent.virtual_root,
        stage="fusion_draft",
        local_best_node=agent.virtual_root,
        draft_role="novel_exploration",
        role_contract=role_contract,
        source_ref_ids=source_node_ids if explicit_cross_role_provenance else [],
    )
    register_node(agent, aggregation_node, prompt_complete, new_branch=True)

    from agents.adoption import log_adoption
    log_adoption(aggregation_node, agent, external_skill_source, external_skill_ref_ids, "fusion_draft")

    agent.fusion_draft_count += 1

    logger.info(f"[aggregation] → node {aggregation_node.id} (branch={aggregation_node.branch_id})")
    return aggregation_node
