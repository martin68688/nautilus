"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging

from agents.leakage_audit import rank_eligible
from engine.search_node import SearchNode
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def validate_executed_node(agent, node: SearchNode):
    """Check submission.csv exists, metric=0.0 anomaly; register successful node to branch."""
    if node.is_buggy:
        return

    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
    if not submission_path.exists():
        node.is_buggy = True
        node.metric = WorstMetricValue()
        logger.info(f"Node {node.id} did not produce a submission.csv")
        return

    try:
        from official_submission import (
            enabled as official_submission_enabled,
            validate_candidate_submission,
        )

        if official_submission_enabled(agent.cfg):
            node.official_submission_receipt = (
                validate_candidate_submission(agent.cfg, node) or {}
            )
            logger.info(
                "Node %s produced a validated native official submission (%s rows)",
                node.id,
                node.official_submission_receipt.get("row_count"),
            )
    except Exception as error:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            "Native official-test submission validation failed: "
            f"{type(error).__name__}: {error}"
        )
        logger.warning(
            "Node %s is not rank-eligible because its official submission failed: %s",
            node.id,
            error,
        )
        return

    if node.metric.maximize and node.metric.value == 0.0:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = _ZERO_METRIC_ANALYSIS
        logger.warning(
            f"Node {node.id} has metric=0.0 (maximize=True), marking as buggy for debugging."
        )
        return

    if hasattr(node, 'branch_id') and node.branch_id and rank_eligible(agent, node):
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        agent.branch_successful_nodes[node.branch_id].append(node)
    elif getattr(agent.acfg, "check_data_leakage", False):
        logger.warning(
            "Node %s remains in the journal for repair but is excluded from branch_successful_nodes",
            node.id,
        )
